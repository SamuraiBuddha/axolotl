"""Module for definition of GEGLU Triton kernels.

See "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""

# pylint: disable=invalid-name,unnecessary-lambda-assignment,duplicate-code

import torch

# Try to import triton, but make it optional for Windows
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    # Create dummy decorators and functions for compatibility
    class triton:
        @staticmethod
        def jit(fn):
            return fn
        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b
    
    class tl:
        constexpr = int

# PyTorch fallback implementations
def _geglu_pytorch_forward(gate, up):
    """PyTorch implementation of GEGLU forward pass."""
    # GELU activation on gate
    gelu_gate = gate * 0.5 * (1.0 + torch.erf(gate / torch.sqrt(torch.tensor(2.0))))
    # Element-wise multiply with up projection
    return gelu_gate * up

def _geglu_pytorch_backward(grad_output, gate, up):
    """PyTorch implementation of GEGLU backward pass."""
    # GELU activation and its derivative
    sqrt_2 = torch.sqrt(torch.tensor(2.0))
    erf_gate = torch.erf(gate / sqrt_2)
    gelu_partial = 0.5 * (erf_gate + 1.0)
    
    # GELU derivative
    exp_term = torch.exp(-0.5 * gate * gate)
    gelu_derivative = 0.5 * (erf_gate + 1.0) + gate * exp_term / torch.sqrt(torch.tensor(2.0 * 3.14159265359))
    
    # Gradients
    grad_gate = grad_output * up * gelu_derivative
    grad_up = grad_output * gate * gelu_partial
    
    # Forward output (h)
    h = gate * gelu_partial * up
    
    return h, grad_gate, grad_up


if HAS_TRITON:
    @triton.jit
    def _geglu_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """GEGLU forward kernel.

        Args:
            gate_ptr: Pointer to gate tensor [*, hidden_dim].
            up_ptr: Pointer to up-projection tensor [*, hidden_dim].
            out_ptr: Pointer to output tensor [*, hidden_dim].
            n_elements: Total number of elements in the input tensors.
            BLOCK_SIZE: Size of thread blocks for parallel computation.
        """
        block_idx = tl.program_id(0)
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0)

        # GELU(gate) * up
        gelu_partial = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * gate) + 1.0)
        gelu_gate = gelu_partial * gate
        gelu_gate = gelu_gate.to(up.dtype)

        out = gelu_gate * up
        tl.store(out_ptr + offsets, out, mask=mask)


def geglu_forward(gate, up, out=None):
    """Compute GEGLU activation: GELU(gate) * up.

    Args:
        gate: Gate tensor, shape `[batch, seq_len, hidden_dim]`.
        up: Up-projection tensor, shape `[batch, seq_len, hidden_dim]`.
        out: Optional output tensor. If None, a new tensor is allocated.

    Returns:
        GEGLU activation output, shape `[batch, seq_len, hidden_dim]`.
    """
    if out is None:
        out = torch.empty_like(gate)
        
    if not HAS_TRITON:
        # Use PyTorch fallback
        result = _geglu_pytorch_forward(gate, up)
        out.copy_(result)
        return out
    
    # Original Triton implementation
    n_elements = gate.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _geglu_fwd_kernel[grid](
        gate_ptr=gate,
        up_ptr=up,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )
    return out


if HAS_TRITON:
    @triton.jit
    def _geglu_bwd_kernel(
        grad_out_ptr,
        gate_ptr,
        up_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """GEGLU backward kernel. Stores gradient results in-place.

        Args:
            grad_out_ptr: Pointer to gradient output tensor [*, hidden_dim].
            gate_ptr: Pointer to gate tensor [*, hidden_dim].
            up_ptr: Pointer to up-projection tensor [*, hidden_dim].
            n_elements: Total number of elements in the input tensors.
            BLOCK_SIZE: Size of thread blocks for parallel computation.

        Note:
            After kernel execution, tensors are modified in-place:
            - `grad_out_ptr` contains GEGLU activation output (`h`)
            - `gate_ptr` contains gradient w.r.t gate (`grad_gate`)
            - `up_ptr` contains gradient w.r.t up (`grad_up`)
        """
        block_idx = tl.program_id(0)
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0)

        # Forward pass
        gelu_partial = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * gate) + 1.0)
        gelu_gate = gelu_partial * gate
        gelu_gate = gelu_gate.to(grad_out.dtype)

        # Forward output
        h = gelu_gate * up

        # Backward pass
        grad_up = grad_out * gelu_gate
        grad_gelu_gate = grad_out * up

        # Gradient of GELU w.r.t gate
        sqrt_2 = tl.math.sqrt(2.0)
        rsqrt_2pi = tl.math.rsqrt(2.0 * 3.14159265359)
        exp_term = tl.math.exp(-0.5 * gate * gate)
        grad_gate = grad_gelu_gate * (gelu_partial + gate * exp_term * rsqrt_2pi)

        # Store results in-place
        tl.store(grad_out_ptr + offsets, h, mask=mask)
        tl.store(gate_ptr + offsets, grad_gate.to(gate.dtype), mask=mask)
        tl.store(up_ptr + offsets, grad_up, mask=mask)


def geglu_backward(grad_output, gate, up):
    """Compute gradients for GEGLU activation.

    Args:
        grad_output: Gradient of loss with respect to output, shape `[batch, seq_len, hidden_dim]`.
        gate: Gate tensor from forward pass, shape `[batch, seq_len, hidden_dim]`.
        up: Up-projection tensor from forward pass, shape `[batch, seq_len, hidden_dim]`.

    Returns:
        Tuple containing:
            - GEGLU activation output (`h`)
            - Gradient with respect to gate (`grad_gate`)
            - Gradient with respect to up (`grad_up`)

    Note:
        This function modifies its input tensors in-place to store results when using Triton.
    """
    if not HAS_TRITON:
        # Use PyTorch fallback - create copies to simulate in-place behavior
        h, grad_gate, grad_up = _geglu_pytorch_backward(grad_output, gate, up)
        grad_output_copy = grad_output.clone()
        gate_copy = gate.clone()
        up_copy = up.clone()
        grad_output_copy.copy_(h)
        gate_copy.copy_(grad_gate)
        up_copy.copy_(grad_up)
        return grad_output_copy, gate_copy, up_copy
    
    # Original Triton implementation
    n_elements = grad_output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    _geglu_bwd_kernel[grid](
        grad_out_ptr=grad_output,
        gate_ptr=gate,
        up_ptr=up,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    return grad_output, gate, up
