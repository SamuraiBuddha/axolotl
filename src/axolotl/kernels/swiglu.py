"""
Module for definition of SwiGLU Triton kernels.

See "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).

Credit to `unsloth` (https://unsloth.ai/) for inspiration for this implementation.
"""

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
def _swiglu_pytorch_forward(gate, up):
    """PyTorch implementation of SwiGLU forward pass."""
    # SiLU activation on gate (x * sigmoid(x))
    silu_gate = gate * torch.sigmoid(gate)
    # Element-wise multiply with up projection
    return silu_gate * up

def _swiglu_pytorch_backward(grad_output, gate, up):
    """PyTorch implementation of SwiGLU backward pass."""
    # SiLU activation
    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    
    # SiLU derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    silu_derivative = sigmoid_gate + gate * sigmoid_gate * (1 - sigmoid_gate)
    
    # Gradients
    grad_gate = grad_output * up * silu_derivative
    grad_up = grad_output * silu_gate
    
    # Forward output (h)
    h = silu_gate * up
    
    return h, grad_gate, grad_up

if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        block_size: tl.constexpr,
    ):
        """SwiGLU forward kernel."""
        pid = tl.program_id(0)
        block_start = pid * block_size
        offsets = block_start + tl.arange(0, block_size)
        mask = offsets < n_elements

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0)

        # SiLU(gate) * up
        gate = gate / (1 + tl.exp(-gate))
        silu = gate * gate_ptr.dtype.element_ty(1)  # gate is already sigmoid(original_gate)
        
        out = silu * up
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        grad_ptr,
        gate_ptr,
        up_ptr,
        n_elements,
        block_size: tl.constexpr,
    ):
        """SwiGLU backward kernel."""
        pid = tl.program_id(0)
        block_start = pid * block_size
        offsets = block_start + tl.arange(0, block_size)
        mask = offsets < n_elements

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0)

        # Forward
        sigmoid_gate = 1 / (1 + tl.exp(-gate))
        silu = gate * sigmoid_gate
        h = silu * up

        # Backward
        grad_up = grad * silu
        grad_silu = grad * up
        grad_gate = grad_silu * (sigmoid_gate + gate * sigmoid_gate * (1 - sigmoid_gate))

        # Store in-place
        tl.store(grad_ptr + offsets, h, mask=mask)
        tl.store(gate_ptr + offsets, grad_gate.to(gate_ptr.dtype.element_ty), mask=mask)
        tl.store(up_ptr + offsets, grad_up, mask=mask)


def swiglu_forward(gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor = None):
    """Compute SwiGLU activation: SiLU(gate) * up.

    Args:
        gate: Gate tensor
        up: Up projection tensor  
        out: Optional output tensor

    Returns:
        SwiGLU activation output
    """
    if out is None:
        out = torch.empty_like(gate)
        
    if not HAS_TRITON:
        # Use PyTorch fallback
        result = _swiglu_pytorch_forward(gate, up)
        out.copy_(result)
        return out
    
    # Original Triton implementation
    n_elements = gate.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["block_size"]),)
    _swiglu_fwd_kernel[grid](
        gate_ptr=gate,
        up_ptr=up,
        out_ptr=out,
        n_elements=n_elements,
        block_size=1024,
    )
    return out


def swiglu_backward(grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor):
    """Compute gradients for SwiGLU activation.

    Args:
        grad_output: Gradient of loss w.r.t output
        gate: Gate tensor from forward pass
        up: Up projection tensor from forward pass

    Returns:
        Tuple of (h, grad_gate, grad_up)
    """
    if not HAS_TRITON:
        # Use PyTorch fallback - create copies to simulate in-place behavior
        h, grad_gate, grad_up = _swiglu_pytorch_backward(grad_output, gate, up)
        grad_output_copy = grad_output.clone()
        gate_copy = gate.clone()
        up_copy = up.clone()
        grad_output_copy.copy_(h)
        gate_copy.copy_(grad_gate)
        up_copy.copy_(grad_up)
        return grad_output_copy, gate_copy, up_copy
    
    # Original Triton implementation
    n_elements = grad_output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["block_size"]),)
    _swiglu_bwd_kernel[grid](
        grad_ptr=grad_output,
        gate_ptr=gate,
        up_ptr=up,
        n_elements=n_elements,
        block_size=1024,
    )
    return grad_output, gate, up
