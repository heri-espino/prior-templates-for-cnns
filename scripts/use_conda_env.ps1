# Native Windows PowerShell helper for the paper experiments.
# Finds a user-space Conda installation, creates the project environment if
# needed, installs dependencies, and exposes Invoke-CnnPython / Invoke-CnnPip.
# No administrator privileges and no `conda activate` are required.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (Get-Variable -Name CnnCondaHelperLoaded -Scope Script -ErrorAction SilentlyContinue) { return }
$script:CnnCondaHelperLoaded = $true

$script:CnnEnvName = if ($env:CNN_ENV_NAME) { $env:CNN_ENV_NAME } else { 'prior-templates-cnns' }
$script:CnnDevice = if ($env:DEVICE) { $env:DEVICE } else { 'cuda' }
$script:TorchIndexUrl = if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { 'https://download.pytorch.org/whl/cu128' }

function Find-UserConda {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return (Resolve-Path $env:CONDA_EXE).Path
    }

    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if (-not $cmd) { $cmd = Get-Command conda -ErrorAction SilentlyContinue }
    if ($cmd) { return $cmd.Source }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'miniconda3\Scripts\conda.exe'),
        (Join-Path $env:LOCALAPPDATA 'anaconda3\Scripts\conda.exe'),
        (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'),
        (Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    throw 'Could not find Conda. Install Miniconda for the current user or set CONDA_EXE. No administrator install is required.'
}

$script:CondaExe = Find-UserConda

function Invoke-CnnPython {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    & $script:CondaExe run --no-capture-output -n $script:CnnEnvName python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

function Invoke-CnnPip {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    & $script:CondaExe run --no-capture-output -n $script:CnnEnvName python -m pip @Arguments
    if ($LASTEXITCODE -ne 0) { throw "pip command failed with exit code $LASTEXITCODE" }
}

# Create the named user environment when absent or incompatible.
& $script:CondaExe run -n $script:CnnEnvName python -c "import sys; assert sys.version_info[:2] == (3, 11)" *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating user Conda environment: $script:CnnEnvName"
    & $script:CondaExe create -y -n $script:CnnEnvName python=3.11 pip
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
}

# Install scientific dependencies only when missing.
$depsCheck = 'import numpy, pandas, scipy, matplotlib, PIL, psutil'
& $script:CondaExe run -n $script:CnnEnvName python -c $depsCheck *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing project dependencies into Conda environment: $script:CnnEnvName"
    Invoke-CnnPip install --upgrade pip
    Invoke-CnnPip install 'numpy>=1.26,<3' 'pandas>=2.1,<4' 'scipy>=1.11,<2' 'matplotlib>=3.8,<4' 'Pillow>=10,<13' 'psutil>=5.9,<8'
}

# Require a CUDA-visible PyTorch wheel whenever GPU evaluation is requested.
if ($script:CnnDevice -eq 'cuda') {
    & $script:CondaExe run -n $script:CnnEnvName python -c "import torch; assert torch.cuda.is_available()" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing CUDA-enabled PyTorch into $script:CnnEnvName from $script:TorchIndexUrl"
        Invoke-CnnPip install --upgrade torch --index-url $script:TorchIndexUrl
    }
} else {
    & $script:CondaExe run -n $script:CnnEnvName python -c "import torch" *> $null
    if ($LASTEXITCODE -ne 0) { Invoke-CnnPip install --upgrade torch }
}

$probe = @'
import sys, torch
print('Conda Python:', sys.executable)
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU VRAM GiB:', round(p.total_memory / 2**30, 2))
'@
$probe | & $script:CondaExe run --no-capture-output -n $script:CnnEnvName python -
if ($LASTEXITCODE -ne 0) { throw 'Conda/PyTorch probe failed.' }

if ($script:CnnDevice -eq 'cuda') {
    & $script:CondaExe run -n $script:CnnEnvName python -c "import torch; assert torch.cuda.is_available()"
    if ($LASTEXITCODE -ne 0) { throw 'CUDA was requested but PyTorch cannot access the GPU.' }
}
