# PowerShell script to launch swarm training in WSL2 Ubuntu
# Run this from Windows to automatically enter Ubuntu and start setup

Write-Host "=== Swarm Architecture WSL2 Launcher ===" -ForegroundColor Cyan
Write-Host ""

# Check if Ubuntu WSL instance exists
$wslList = wsl --list --quiet
if ($wslList -notcontains "Ubuntu") {
    Write-Host "ERROR: Ubuntu WSL instance not found!" -ForegroundColor Red
    Write-Host "Available WSL instances:" -ForegroundColor Yellow
    wsl --list
    Write-Host ""
    Write-Host "Please install Ubuntu from Microsoft Store first." -ForegroundColor Yellow
    exit 1
}

# Create the command to run in WSL
$projectPath = "/mnt/c/Users/$env:USERNAME/Documents/GitHub/axolotl/training_approaches/swarm_architecture"
$wslCommand = @"
cd $projectPath && \
if [ ! -f check_wsl_env.sh ]; then \
    echo 'Error: Project files not found at $projectPath'; \
    exit 1; \
fi && \
chmod +x check_wsl_env.sh wsl_setup.sh && \
echo '=== Checking WSL2 Environment ===' && \
./check_wsl_env.sh && \
echo '' && \
echo 'To start setup, run: ./wsl_setup.sh' && \
echo 'To train after setup: source swarm_env/bin/activate && python train_swarm_wsl.py' && \
exec bash
"@

Write-Host "Launching Ubuntu WSL instance..." -ForegroundColor Green
Write-Host "Project path: C:\Users\$env:USERNAME\Documents\GitHub\axolotl\training_approaches\swarm_architecture" -ForegroundColor Gray
Write-Host ""

# Launch WSL with Ubuntu and run commands
wsl -d Ubuntu -e bash -c $wslCommand
