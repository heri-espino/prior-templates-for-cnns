param([ValidateSet('smoke','pilot','main','report')][string]$Mode='main', [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArgs)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
    & .venv/Scripts/python.exe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
}
$RunArgs = switch ($Mode) {
    'smoke' { @('experiment.py','--output','outputs/smoke','--epochs','2','--blocks','1','--tasks','two_concepts','--architectures','TinyCNN','--conditions','template_release','--release-start','0','--release-end','2','--checkpoints','1','2') }
    'pilot' { @('experiment.py','--output','outputs/pilot','--blocks','1') }
    'main' { @('experiment.py') }
    'report' { @('summarize.py') }
}
& .venv/Scripts/python.exe @RunArgs @ExtraArgs
exit $LASTEXITCODE
