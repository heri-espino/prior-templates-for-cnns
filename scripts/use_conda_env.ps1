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

    $candidates = @()
    if ($env:LOCALAPPDATA) {
        $candidates += (Join-Path $env:LOCALAPPDATA 'miniconda3\Scripts\conda.exe')
        $candidates += (Join-Path $env:LOCALAPPDATA 'anaconda3\Scripts\conda.exe')
    }
    if ($env:USERPROFILE) {
        $candidates += (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe')
        $candidates += (Join-Path $env:USERPROFILE 'anaconda3\Scripts\conda.exe')
    }
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    throw 'Could not find Conda. Install Miniconda for the current user or set CONDA_EXE. No administrator install is required.'
}

$script:CondaExe = Find-UserConda

# Windows PowerShell 5.1 turns stderr written by native programs into ErrorRecord
# objects. With ErrorActionPreference=Stop, an expected non-zero probe (for
# example, checking whether an environment exists) can therefore terminate the
# script before $LASTEXITCODE is inspected. Keep native Conda calls under
# ErrorActionPreference=Continue and decide success explicitly from exit codes.
function Test-CondaCommand {
    param([Parameter(Mandatory=$true)][string[]]$Arguments)

    $previousPreference = $ErrorActionPreference
    $code = 1
    try {
        $ErrorActionPreference = 'Continue'
        & $script:CondaExe @Arguments 1>$null 2>$null
        $code = $LASTEXITCODE
    }
    catch {
        $code = 1
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    return ($code -eq 0)
}

function Invoke-CondaChecked {
    param(
        [Parameter(Mandatory=$true)][string[]]$Arguments,
        [string]$Description = 'Conda command'
    )

    $previousPreference = $ErrorActionPreference
    $code = 1
    try {
        $ErrorActionPreference = 'Continue'
        & $script:CondaExe @Arguments
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($code -ne 0) {
        throw "$Description failed with exit code $code."
    }
}

function Invoke-CnnPython {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    $condaArgs = @('run','--no-capture-output','-n',$script:CnnEnvName,'python') + $Arguments
    Invoke-CondaChecked -Arguments $condaArgs -Description 'Python command'
}

function Invoke-CnnPip {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    $condaArgs = @('run','--no-capture-output','-n',$script:CnnEnvName,'python','-m','pip') + $Arguments
    Invoke-CondaChecked -Arguments $condaArgs -Description 'pip command'
}

Write-Host "Conda executable: $script:CondaExe"
Write-Host "Project environment: $script:CnnEnvName"

# Create the named user environment when absent or incompatible. This probe is
# intentionally allowed to fail and therefore must go through Test-CondaCommand.
$python311Probe = @('run','-n',$script:CnnEnvName,'python','-c','import sys; assert sys.version_info[:2] == (3, 11)')
if (-not (Test-CondaCommand -Arguments $python311Probe)) {
    Write-Host "Creating user Conda environment: $script:CnnEnvName"
    Invoke-CondaChecked -Arguments @('create','-y','-n',$script:CnnEnvName,'python=3.11','pip') -Description 'Conda environment creation'
}

# Install scientific dependencies only when missing.
$depsCheck = 'import numpy, pandas, scipy, matplotlib, PIL, psutil'
$depsProbe = @('run','-n',$script:CnnEnvName,'python','-c',$depsCheck)
if (-not (Test-CondaCommand -Arguments $depsProbe)) {
    Write-Host "Installing project dependencies into Conda environment: $script:CnnEnvName"
    Invoke-CnnPip 'install' '--upgrade' 'pip'
    Invoke-CnnPip 'install' 'numpy>=1.26,<3' 'pandas>=2.1,<4' 'scipy>=1.11,<2' 'matplotlib>=3.8,<4' 'Pillow>=10,<13' 'psutil>=5.9,<8'
}

# Require a CUDA-visible PyTorch wheel whenever GPU evaluation is requested.
if ($script:CnnDevice -eq 'cuda') {
    $cudaProbe = @('run','-n',$script:CnnEnvName,'python','-c','import torch; assert torch.cuda.is_available()')
    if (-not (Test-CondaCommand -Arguments $cudaProbe)) {
        Write-Host "Installing CUDA-enabled PyTorch into $script:CnnEnvName from $script:TorchIndexUrl"
        Invoke-CnnPip 'install' '--upgrade' 'torch' '--index-url' $script:TorchIndexUrl
    }
    if (-not (Test-CondaCommand -Arguments $cudaProbe)) {
        throw 'CUDA was requested but PyTorch cannot access the GPU after installation.'
    }
}
else {
    $torchProbe = @('run','-n',$script:CnnEnvName,'python','-c','import torch')
    if (-not (Test-CondaCommand -Arguments $torchProbe)) {
        Invoke-CnnPip 'install' '--upgrade' 'torch'
    }
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
$probeFile = Join-Path ([System.IO.Path]::GetTempPath()) ("cnn_probe_{0}.py" -f [guid]::NewGuid().ToString('N'))
try {
    [System.IO.File]::WriteAllText($probeFile, $probe, [System.Text.Encoding]::UTF8)
    Invoke-CnnPython $probeFile
}
finally {
    Remove-Item $probeFile -Force -ErrorAction SilentlyContinue
}
