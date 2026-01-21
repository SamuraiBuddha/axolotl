// Axolotl GUI Application JavaScript

// Global variables
let currentConfig = null;
let currentTrainingId = null;
let activeTrainings = {};
let eventSource = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    loadSystemInfo();
    setInterval(loadSystemInfo, 5000); // Update every 5 seconds
});

// Initialize app
function initializeApp() {
    loadConfigs();
    loadModels();
    loadDatasets();
    updateTrainingList();
}

// Setup event listeners
function setupEventListeners() {
    // Tab navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            const tab = e.currentTarget.dataset.tab;
            switchTab(tab);
        });
    });

    // Quick train button
    document.getElementById('quickTrainBtn')?.addEventListener('click', showQuickTrainModal);

    // Config editor auto-save
    let saveTimeout;
    document.getElementById('configEditor')?.addEventListener('input', () => {
        clearTimeout(saveTimeout);
        saveTimeout = setTimeout(() => {
            if (currentConfig) {
                saveConfig(true); // Auto-save
            }
        }, 2000);
    });
}

// Tab switching
function switchTab(tabName) {
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.tab === tabName);
    });

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });

    // Load tab-specific data
    switch(tabName) {
        case 'training':
            updateTrainingList();
            break;
        case 'configs':
            loadConfigs();
            break;
        case 'models':
            loadModels();
            break;
        case 'datasets':
            loadDatasets();
            break;
    }
}

// Load system information
async function loadSystemInfo() {
    try {
        const response = await fetch('/api/system/info');
        const data = await response.json();
        
        // Update CPU info
        document.getElementById('cpuInfo').textContent = 
            `${data.cpu.cores} cores, ${data.cpu.percent}% usage`;
        
        // Update Memory info
        const memUsedGB = (data.memory.used / 1024 / 1024 / 1024).toFixed(1);
        const memTotalGB = (data.memory.total / 1024 / 1024 / 1024).toFixed(1);
        document.getElementById('memInfo').textContent = 
            `${memUsedGB} / ${memTotalGB} GB (${data.memory.percent.toFixed(1)}%)`;
        
        // Update GPU info
        if (data.gpu && data.gpu.length > 0) {
            const gpu = data.gpu[0];
            document.getElementById('gpuInfo').textContent = 
                `${gpu.name}, ${gpu.memory_used} / ${gpu.memory_total} MB`;
        } else {
            document.getElementById('gpuInfo').textContent = 'No GPU detected';
        }
        
        // Update Python info
        document.getElementById('pythonInfo').textContent = data.python_version;
        
        // Update system status
        updateSystemStatus('online');
        
    } catch (error) {
        console.error('Failed to load system info:', error);
        updateSystemStatus('error');
    }
}

// Update system status indicator
function updateSystemStatus(status) {
    const indicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.status-text');
    
    switch(status) {
        case 'online':
            indicator.style.background = 'var(--success)';
            statusText.textContent = 'System Ready';
            break;
        case 'training':
            indicator.style.background = 'var(--warning)';
            statusText.textContent = 'Training Active';
            break;
        case 'error':
            indicator.style.background = 'var(--danger)';
            statusText.textContent = 'Connection Error';
            break;
    }
}

// Load configurations
async function loadConfigs() {
    try {
        const response = await fetch('/api/configs/list');
        const configs = await response.json();
        
        // Update config select dropdown
        const configSelect = document.getElementById('configSelect');
        if (configSelect) {
            configSelect.innerHTML = '<option value="">-- Select a config file --</option>';
            configs.forEach(config => {
                const option = document.createElement('option');
                option.value = config.path;
                option.textContent = `${config.category}/${config.name}`;
                configSelect.appendChild(option);
            });
        }
        
        // Update config list in config manager
        const configList = document.getElementById('configList');
        if (configList) {
            configList.innerHTML = '';
            configs.forEach(config => {
                const item = document.createElement('div');
                item.className = 'config-item';
                item.textContent = config.name;
                item.onclick = () => loadConfigForEdit(config.path);
                configList.appendChild(item);
            });
        }
    } catch (error) {
        console.error('Failed to load configs:', error);
        showNotification('Failed to load configurations', 'error');
    }
}

// Load config for editing
async function loadConfigForEdit(path) {
    try {
        const response = await fetch('/api/configs/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        const data = await response.json();
        
        currentConfig = path;
        document.getElementById('configName').textContent = path.split('/').pop();
        document.getElementById('configEditor').value = data.raw;
        
    } catch (error) {
        console.error('Failed to load config:', error);
        showNotification('Failed to load configuration', 'error');
    }
}

// Save configuration
async function saveConfig(autoSave = false) {
    if (!currentConfig) {
        showNotification('No configuration selected', 'warning');
        return;
    }
    
    try {
        const content = document.getElementById('configEditor').value;
        const response = await fetch('/api/configs/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                path: currentConfig,
                content: content
            })
        });
        
        if (response.ok) {
            if (!autoSave) {
                showNotification('Configuration saved successfully', 'success');
            }
        } else {
            const error = await response.json();
            showNotification(`Failed to save: ${error.error}`, 'error');
        }
    } catch (error) {
        console.error('Failed to save config:', error);
        showNotification('Failed to save configuration', 'error');
    }
}

// Create new configuration
async function createNewConfig() {
    const modal = document.createElement('div');
    modal.className = 'modal active';
    modal.innerHTML = `
        <div class="modal-content">
            <h3>Create New Configuration</h3>
            <div class="form-group">
                <label>Configuration Name:</label>
                <input type="text" id="newConfigName" class="form-control" value="my_config.yml">
            </div>
            <div class="form-group">
                <label>Template:</label>
                <select id="configTemplate" class="form-control">
                    <option value="lora">LoRA Training</option>
                    <option value="qlora">QLoRA Training</option>
                    <option value="full">Full Fine-tuning</option>
                </select>
            </div>
            <div class="modal-actions">
                <button class="btn btn-primary" onclick="executeCreateConfig()">Create</button>
                <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    document.getElementById('modalOverlay').classList.add('active');
}

// Execute config creation
async function executeCreateConfig() {
    const name = document.getElementById('newConfigName').value;
    const template = document.getElementById('configTemplate').value;
    
    try {
        const response = await fetch('/api/configs/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, template })
        });
        
        const data = await response.json();
        if (data.success) {
            showNotification('Configuration created successfully', 'success');
            loadConfigs();
            closeModal();
            loadConfigForEdit(data.path);
        }
    } catch (error) {
        console.error('Failed to create config:', error);
        showNotification('Failed to create configuration', 'error');
    }
}

// Start training
async function startTraining() {
    const configPath = document.getElementById('configSelect').value;
    if (!configPath) {
        showNotification('Please select a configuration', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config_path: configPath })
        });
        
        const data = await response.json();
        if (data.success) {
            currentTrainingId = data.training_id;
            showNotification('Training started successfully', 'success');
            startLogStreaming(data.training_id);
            updateTrainingList();
            updateSystemStatus('training');
        }
    } catch (error) {
        console.error('Failed to start training:', error);
        showNotification('Failed to start training', 'error');
    }
}

// Start log streaming
function startLogStreaming(trainingId) {
    if (eventSource) {
        eventSource.close();
    }
    
    eventSource = new EventSource(`/api/training/${trainingId}/stream`);
    const logContainer = document.getElementById('trainingLogs');
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.log) {
            const logLine = document.createElement('div');
            logLine.textContent = data.log;
            logContainer.appendChild(logLine);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        if (data.status && data.status !== 'running') {
            showNotification(`Training ${data.status}`, 
                data.status === 'completed' ? 'success' : 'warning');
            eventSource.close();
            updateTrainingList();
            updateSystemStatus('online');
        }
    };
    
    eventSource.onerror = () => {
        console.error('Log streaming error');
        eventSource.close();
    };
}

// Update training list
async function updateTrainingList() {
    try {
        const response = await fetch('/api/training/list');
        const trainings = await response.json();
        
        const activeContainer = document.getElementById('activeTrainings');
        const selectElement = document.getElementById('activeTrainingSelect');
        
        if (Object.keys(trainings).length === 0) {
            activeContainer.innerHTML = '<p class="no-data">No active trainings</p>';
            if (selectElement) {
                selectElement.innerHTML = '<option value="">-- No active trainings --</option>';
            }
        } else {
            activeContainer.innerHTML = '';
            if (selectElement) {
                selectElement.innerHTML = '<option value="">-- Select active training --</option>';
            }
            
            Object.entries(trainings).forEach(([id, training]) => {
                // Update active trainings display
                const item = document.createElement('div');
                item.className = 'training-item';
                item.innerHTML = `
                    <div style="padding: 0.5rem; background: var(--surface-light); border-radius: 6px; margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span>${id}</span>
                            <span class="badge ${training.status === 'running' ? 'badge-success' : 'badge-warning'}">
                                ${training.status}
                            </span>
                        </div>
                        <small style="color: var(--text-secondary);">${training.config}</small>
                    </div>
                `;
                activeContainer.appendChild(item);
                
                // Update select dropdown
                if (selectElement) {
                    const option = document.createElement('option');
                    option.value = id;
                    option.textContent = `${id} (${training.status})`;
                    selectElement.appendChild(option);
                }
            });
        }
    } catch (error) {
        console.error('Failed to update training list:', error);
    }
}

// Load models
async function loadModels() {
    try {
        const response = await fetch('/api/models/list');
        const models = await response.json();
        
        const modelsGrid = document.getElementById('modelsGrid');
        const recentModels = document.getElementById('recentModels');
        
        if (models.length === 0) {
            modelsGrid.innerHTML = '<p class="no-data">No trained models found</p>';
            if (recentModels) {
                recentModels.innerHTML = '<p class="no-data">No models found</p>';
            }
        } else {
            modelsGrid.innerHTML = '';
            if (recentModels) {
                recentModels.innerHTML = '';
            }
            
            models.forEach((model, index) => {
                const card = document.createElement('div');
                card.className = 'model-card';
                card.innerHTML = `
                    <div class="model-name">${model.name}</div>
                    <div class="model-info">
                        <div>Path: ${model.path}</div>
                        <div>Size: ${formatBytes(model.size)}</div>
                        <div>Created: ${new Date(model.created).toLocaleDateString()}</div>
                    </div>
                    <div class="button-group">
                        <button class="btn btn-sm btn-primary" onclick="useModel('${model.path}')">Use</button>
                        <button class="btn btn-sm btn-secondary" onclick="exportModel('${model.path}')">Export</button>
                    </div>
                `;
                modelsGrid.appendChild(card);
                
                // Add to recent models (max 3)
                if (recentModels && index < 3) {
                    const recentItem = document.createElement('div');
                    recentItem.style = "padding: 0.5rem; background: var(--surface-light); border-radius: 6px; margin-bottom: 0.5rem;";
                    recentItem.innerHTML = `
                        <div>${model.name}</div>
                        <small style="color: var(--text-secondary);">${formatBytes(model.size)}</small>
                    `;
                    recentModels.appendChild(recentItem);
                }
            });
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        showNotification('Failed to load models', 'error');
    }
}

// Load datasets
async function loadDatasets() {
    try {
        const response = await fetch('/api/datasets/list');
        const datasets = await response.json();
        
        const localContainer = document.getElementById('localDatasets');
        const hfContainer = document.getElementById('hfDatasets');
        
        localContainer.innerHTML = '';
        hfContainer.innerHTML = '';
        
        datasets.forEach(dataset => {
            const item = document.createElement('div');
            item.style = "padding: 0.75rem; background: var(--surface-light); border-radius: 6px; margin-bottom: 0.5rem;";
            
            if (dataset.type === 'local') {
                item.innerHTML = `
                    <div style="display: flex; justify-content: space-between;">
                        <span>${dataset.name}.${dataset.format}</span>
                        <span style="color: var(--text-secondary);">${formatBytes(dataset.size)}</span>
                    </div>
                    <small style="color: var(--text-secondary);">${dataset.path}</small>
                `;
                localContainer.appendChild(item);
            } else {
                item.innerHTML = `
                    <div>${dataset.name}</div>
                    <button class="btn btn-sm btn-secondary" style="margin-top: 0.5rem;" 
                            onclick="useDataset('${dataset.name}')">Use Dataset</button>
                `;
                hfContainer.appendChild(item);
            }
        });
        
        if (localContainer.children.length === 0) {
            localContainer.innerHTML = '<p class="no-data">No local datasets found</p>';
        }
    } catch (error) {
        console.error('Failed to load datasets:', error);
        showNotification('Failed to load datasets', 'error');
    }
}

// Quick train modal
function showQuickTrainModal() {
    document.getElementById('quickTrainModal').classList.add('active');
    document.getElementById('modalOverlay').classList.add('active');
}

// Execute quick train
async function executeQuickTrain() {
    const modelSize = document.getElementById('quickModelSize').value;
    const trainType = document.getElementById('quickTrainType').value;
    
    // Create a quick config based on selections
    const configName = `quick_${trainType}_${modelSize}_${Date.now()}.yml`;
    
    try {
        const response = await fetch('/api/configs/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: configName,
                template: trainType
            })
        });
        
        const data = await response.json();
        if (data.success) {
            // Start training with the new config
            const trainResponse = await fetch('/api/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config_path: data.path })
            });
            
            const trainData = await trainResponse.json();
            if (trainData.success) {
                currentTrainingId = trainData.training_id;
                showNotification('Quick training started!', 'success');
                closeModal();
                switchTab('training');
                startLogStreaming(trainData.training_id);
            }
        }
    } catch (error) {
        console.error('Failed to start quick training:', error);
        showNotification('Failed to start quick training', 'error');
    }
}

// Helper functions
function closeModal() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('active');
        if (modal.id !== 'quickTrainModal') {
            modal.remove();
        }
    });
    document.getElementById('modalOverlay').classList.remove('active');
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        background: var(--${type === 'success' ? 'success' : type === 'error' ? 'danger' : type === 'warning' ? 'warning' : 'info'});
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 2000;
        animation: slideIn 0.3s;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Additional functions for features
function reloadConfigs() {
    loadConfigs();
    showNotification('Configurations reloaded', 'success');
}

function validateConfig() {
    // TODO: Implement config validation
    showNotification('Config validation not yet implemented', 'info');
}

function deleteConfig() {
    if (confirm('Are you sure you want to delete this configuration?')) {
        // TODO: Implement config deletion
        showNotification('Config deletion not yet implemented', 'info');
    }
}

function stopTraining() {
    const trainingId = document.getElementById('activeTrainingSelect').value;
    if (!trainingId) {
        showNotification('Please select a training to stop', 'warning');
        return;
    }
    
    // TODO: Implement training stop
    showNotification('Stop training not yet implemented', 'info');
}

function refreshModels() {
    loadModels();
    showNotification('Models list refreshed', 'success');
}

function fetchExamples() {
    // TODO: Implement fetch examples
    showNotification('Fetching examples...', 'info');
}

function openDocs() {
    window.open('https://docs.axolotl.ai/', '_blank');
}

function preprocessOnly() {
    // TODO: Implement preprocess only
    showNotification('Preprocessing not yet implemented', 'info');
}

function useModel(path) {
    showNotification(`Model ${path} selected`, 'info');
}

function exportModel(path) {
    showNotification('Model export not yet implemented', 'info');
}

function useDataset(name) {
    showNotification(`Dataset ${name} selected`, 'info');
}

function addDataset() {
    showNotification('Add dataset feature coming soon', 'info');
}

function importConfig() {
    showNotification('Import config feature coming soon', 'info');
}