param(
    [string]$ResultsRoot = $(Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'prior-templates-cnns\results')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ConfirmationSource = Join-Path $ResultsRoot 'budget_confirmation_001'
$ExhaustiveSource = Join-Path $ResultsRoot 'exhaustive_robustness_001'

$ConfirmationDest = Join-Path $RepoRoot 'analysis\budget_confirmation_001'
$ExhaustiveDest = Join-Path $RepoRoot 'analysis\exhaustive_robustness_001'

if (-not (Test-Path $ConfirmationSource)) {
    throw "Confirmation results not found: $ConfirmationSource"
}
if (-not (Test-Path $ExhaustiveSource)) {
    throw "Exhaustive results not found: $ExhaustiveSource"
}

New-Item -ItemType Directory -Force -Path $ConfirmationDest | Out-Null
New-Item -ItemType Directory -Force -Path $ExhaustiveDest | Out-Null

# Copy only compact, paper-facing outputs. Training checkpoints and raw model
# trees remain in LocalAppData and are deliberately not committed.
$confirmationAnalysis = Join-Path $ConfirmationSource 'analysis\budget_confirmation'
if (-not (Test-Path $confirmationAnalysis)) {
    throw "Confirmation analysis directory not found: $confirmationAnalysis"
}
Copy-Item -Path (Join-Path $confirmationAnalysis '*') -Destination $ConfirmationDest -Recurse -Force
Copy-Item -Path (Join-Path $ConfirmationSource 'execution_manifest.json') -Destination (Join-Path $ConfirmationDest 'execution_manifest.json') -Force

$exhaustiveAnalysis = Join-Path $ExhaustiveSource 'analysis'
if (-not (Test-Path $exhaustiveAnalysis)) {
    throw "Exhaustive analysis directory not found: $exhaustiveAnalysis"
}
Copy-Item -Path (Join-Path $exhaustiveAnalysis '*') -Destination $ExhaustiveDest -Recurse -Force
Copy-Item -Path (Join-Path $ExhaustiveSource 'execution_manifest.json') -Destination (Join-Path $ExhaustiveDest 'execution_manifest.json') -Force

Write-Host ''
Write-Host 'Imported completed paper results into the repository:'
Write-Host "  $ConfirmationDest"
Write-Host "  $ExhaustiveDest"
Write-Host ''
Write-Host 'Review with:'
Write-Host '  git status'
Write-Host '  git diff --stat'
Write-Host ''
Write-Host 'Then commit with:'
Write-Host '  git add analysis/budget_confirmation_001 analysis/exhaustive_robustness_001'
Write-Host '  git commit -m "results: add prospective confirmation and exhaustive robustness outputs"'
Write-Host '  git push origin paper_suggestions'
