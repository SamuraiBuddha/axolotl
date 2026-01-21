#!/usr/bin/env python3
"""
Axolotl GUI - Web-based interface for Axolotl training
"""

import os
import sys
import json
import yaml
import subprocess
import threading
import queue
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
import psutil

# Initialize Flask app
app = Flask(__name__, 
           template_folder='gui_templates',
           static_folder='gui_static')
CORS(app)

# Global variables for managing training processes
training_processes = {}
training_logs = {}
log_queues = {}

class TrainingManager:
    """Manages training processes and logs"""
    
    def __init__(self):
        self.processes = {}
        self.logs = {}
        self.queues = {}
        
    def start_training(self, config_path: str, training_id: str) -> bool:
        """Start a new training process"""
        if training_id in self.processes:
            return False
            
        # Create log queue
        log_queue = queue.Queue()
        self.queues[training_id] = log_queue
        self.logs[training_id] = []
        
        # Start training in a separate thread
        thread = threading.Thread(
            target=self._run_training,
            args=(config_path, training_id, log_queue)
        )
        thread.daemon = True
        thread.start()
        
        self.processes[training_id] = {
            'thread': thread,
            'status': 'running',
            'config': config_path,
            'start_time': datetime.now().isoformat()
        }
        
        return True
        
    def _run_training(self, config_path: str, training_id: str, log_queue: queue.Queue):
        """Run training process and capture output"""
        try:
            # Check if virtual environment exists
            venv_path = Path("venv_axolotl")
            if venv_path.exists():
                python_cmd = str(venv_path / "bin" / "python")
            else:
                python_cmd = "python3"
                
            # Run axolotl train command
            cmd = [python_cmd, "-m", "axolotl.cli.train", config_path]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to queue
            for line in process.stdout:
                log_queue.put(line.strip())
                self.logs[training_id].append({
                    'timestamp': datetime.now().isoformat(),
                    'message': line.strip()
                })
                
            process.wait()
            
            # Update status
            if training_id in self.processes:
                self.processes[training_id]['status'] = 'completed' if process.returncode == 0 else 'failed'
                self.processes[training_id]['end_time'] = datetime.now().isoformat()
                
        except Exception as e:
            log_queue.put(f"Error: {str(e)}")
            if training_id in self.processes:
                self.processes[training_id]['status'] = 'error'
                self.processes[training_id]['error'] = str(e)
                
    def stop_training(self, training_id: str) -> bool:
        """Stop a training process"""
        if training_id not in self.processes:
            return False
            
        # TODO: Implement proper process termination
        self.processes[training_id]['status'] = 'stopped'
        return True
        
    def get_logs(self, training_id: str, last_n: int = 100) -> List[Dict]:
        """Get recent logs for a training process"""
        if training_id not in self.logs:
            return []
        return self.logs[training_id][-last_n:]
        
    def get_status(self, training_id: str) -> Optional[Dict]:
        """Get status of a training process"""
        return self.processes.get(training_id)

# Initialize training manager
training_manager = TrainingManager()

# Routes

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/system/info')
def system_info():
    """Get system information"""
    try:
        # GPU info
        gpu_info = []
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_info.append({
                            'name': parts[0],
                            'memory_total': int(parts[1]),
                            'memory_used': int(parts[2]),
                            'memory_free': int(parts[3]),
                            'utilization': int(parts[4])
                        })
        except:
            pass
            
        # CPU and Memory info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return jsonify({
            'cpu': {
                'percent': cpu_percent,
                'cores': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total,
                'used': memory.used,
                'free': memory.free,
                'percent': memory.percent
            },
            'gpu': gpu_info,
            'python_version': sys.version.split()[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/configs/list')
def list_configs():
    """List available configuration files"""
    configs = []
    
    # Search for YAML configs in common locations
    search_paths = [
        Path('.'),
        Path('examples'),
        Path('configs')
    ]
    
    for base_path in search_paths:
        if base_path.exists():
            for yaml_file in base_path.glob('**/*.yml'):
                configs.append({
                    'path': str(yaml_file),
                    'name': yaml_file.stem,
                    'category': yaml_file.parent.name
                })
            for yaml_file in base_path.glob('**/*.yaml'):
                configs.append({
                    'path': str(yaml_file),
                    'name': yaml_file.stem,
                    'category': yaml_file.parent.name
                })
                
    return jsonify(configs)

@app.route('/api/configs/load', methods=['POST'])
def load_config():
    """Load a configuration file"""
    try:
        config_path = request.json.get('path')
        if not config_path or not Path(config_path).exists():
            return jsonify({'error': 'Config file not found'}), 404
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return jsonify({
            'path': config_path,
            'content': config,
            'raw': open(config_path, 'r').read()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/configs/save', methods=['POST'])
def save_config():
    """Save a configuration file"""
    try:
        data = request.json
        config_path = data.get('path')
        config_content = data.get('content')
        
        if not config_path or not config_content:
            return jsonify({'error': 'Invalid request'}), 400
            
        # Parse and validate YAML
        config = yaml.safe_load(config_content)
        
        # Save to file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        return jsonify({'success': True, 'path': config_path})
    except yaml.YAMLError as e:
        return jsonify({'error': f'Invalid YAML: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/configs/create', methods=['POST'])
def create_config():
    """Create a new configuration from template"""
    try:
        data = request.json
        template_type = data.get('template', 'lora')
        config_name = data.get('name', 'new_config.yml')
        
        # Base configuration templates
        templates = {
            'lora': {
                'base_model': 'NousResearch/Llama-3.2-1B',
                'load_in_8bit': True,
                'adapter': 'lora',
                'lora_r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.05,
                'lora_target_linear': True,
                'datasets': [{
                    'path': 'teknium/GPT4-LLM-Cleaned',
                    'type': 'alpaca'
                }],
                'dataset_prepared_path': './prepared_data',
                'val_set_size': 0.1,
                'output_dir': './outputs/lora_model',
                'micro_batch_size': 2,
                'gradient_accumulation_steps': 4,
                'num_epochs': 3,
                'learning_rate': 0.0003,
                'warmup_steps': 100,
                'gradient_checkpointing': True,
                'flash_attention': True,
                'logging_steps': 10,
                'eval_steps': 50,
                'save_steps': 200,
                'save_total_limit': 3
            },
            'qlora': {
                'base_model': 'NousResearch/Llama-3.2-1B',
                'load_in_4bit': True,
                'adapter': 'qlora',
                'lora_r': 32,
                'lora_alpha': 64,
                'lora_dropout': 0.05,
                'lora_target_linear': True,
                'datasets': [{
                    'path': 'teknium/GPT4-LLM-Cleaned',
                    'type': 'alpaca'
                }],
                'dataset_prepared_path': './prepared_data',
                'val_set_size': 0.1,
                'output_dir': './outputs/qlora_model',
                'micro_batch_size': 1,
                'gradient_accumulation_steps': 8,
                'num_epochs': 3,
                'learning_rate': 0.0002,
                'warmup_steps': 100,
                'gradient_checkpointing': True,
                'flash_attention': True,
                'logging_steps': 10,
                'eval_steps': 50,
                'save_steps': 200,
                'save_total_limit': 3
            },
            'full': {
                'base_model': 'NousResearch/Llama-3.2-1B',
                'datasets': [{
                    'path': 'teknium/GPT4-LLM-Cleaned',
                    'type': 'alpaca'
                }],
                'dataset_prepared_path': './prepared_data',
                'val_set_size': 0.1,
                'output_dir': './outputs/full_model',
                'micro_batch_size': 1,
                'gradient_accumulation_steps': 16,
                'num_epochs': 3,
                'learning_rate': 0.00001,
                'warmup_steps': 100,
                'gradient_checkpointing': True,
                'flash_attention': True,
                'logging_steps': 10,
                'eval_steps': 50,
                'save_steps': 200,
                'save_total_limit': 3
            }
        }
        
        template = templates.get(template_type, templates['lora'])
        
        # Save the new config
        config_path = Path('configs') / config_name
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)
            
        return jsonify({
            'success': True,
            'path': str(config_path),
            'content': template
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start a new training job"""
    try:
        data = request.json
        config_path = data.get('config_path')
        
        if not config_path or not Path(config_path).exists():
            return jsonify({'error': 'Config file not found'}), 404
            
        # Generate training ID
        training_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start training
        success = training_manager.start_training(config_path, training_id)
        
        if success:
            return jsonify({
                'success': True,
                'training_id': training_id
            })
        else:
            return jsonify({'error': 'Failed to start training'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/<training_id>/stop', methods=['POST'])
def stop_training(training_id):
    """Stop a training job"""
    success = training_manager.stop_training(training_id)
    return jsonify({'success': success})

@app.route('/api/training/<training_id>/status')
def training_status(training_id):
    """Get training job status"""
    status = training_manager.get_status(training_id)
    if status:
        return jsonify(status)
    else:
        return jsonify({'error': 'Training job not found'}), 404

@app.route('/api/training/<training_id>/logs')
def training_logs(training_id):
    """Get training logs"""
    last_n = request.args.get('last_n', 100, type=int)
    logs = training_manager.get_logs(training_id, last_n)
    return jsonify(logs)

@app.route('/api/training/list')
def list_trainings():
    """List all training jobs"""
    return jsonify(training_manager.processes)

@app.route('/api/training/<training_id>/stream')
def stream_logs(training_id):
    """Stream training logs in real-time"""
    def generate():
        if training_id not in training_manager.queues:
            yield f"data: {json.dumps({'error': 'Training not found'})}\n\n"
            return
            
        log_queue = training_manager.queues[training_id]
        
        while True:
            try:
                # Get log from queue with timeout
                log = log_queue.get(timeout=1)
                yield f"data: {json.dumps({'log': log})}\n\n"
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
                
                # Check if training is still running
                status = training_manager.get_status(training_id)
                if status and status['status'] not in ['running']:
                    yield f"data: {json.dumps({'status': status['status']})}\n\n"
                    break
                    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/models/list')
def list_models():
    """List trained models in output directory"""
    models = []
    output_dir = Path('outputs')
    
    if output_dir.exists():
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir():
                # Check for model files
                has_model = any([
                    (model_dir / 'adapter_config.json').exists(),
                    (model_dir / 'pytorch_model.bin').exists(),
                    (model_dir / 'model.safetensors').exists(),
                ])
                
                if has_model:
                    models.append({
                        'name': model_dir.name,
                        'path': str(model_dir),
                        'created': datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
                        'size': sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                    })
                    
    return jsonify(models)

@app.route('/api/datasets/list')
def list_datasets():
    """List available datasets"""
    datasets = []
    
    # Check for local dataset files
    dataset_extensions = ['.json', '.jsonl', '.csv', '.txt', '.parquet']
    search_dirs = [Path('.'), Path('data'), Path('datasets')]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for ext in dataset_extensions:
                for file in search_dir.glob(f'**/*{ext}'):
                    datasets.append({
                        'name': file.stem,
                        'path': str(file),
                        'type': 'local',
                        'format': ext[1:],
                        'size': file.stat().st_size
                    })
                    
    # Add some popular HuggingFace datasets
    hf_datasets = [
        {'name': 'teknium/GPT4-LLM-Cleaned', 'type': 'huggingface'},
        {'name': 'yahma/alpaca-cleaned', 'type': 'huggingface'},
        {'name': 'databricks/databricks-dolly-15k', 'type': 'huggingface'},
        {'name': 'OpenAssistant/oasst1', 'type': 'huggingface'},
    ]
    
    datasets.extend(hf_datasets)
    
    return jsonify(datasets)

if __name__ == '__main__':
    # Create necessary directories
    Path('gui_templates').mkdir(exist_ok=True)
    Path('gui_static').mkdir(exist_ok=True)
    Path('configs').mkdir(exist_ok=True)
    
    # Run the app
    print("Starting Axolotl GUI on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)