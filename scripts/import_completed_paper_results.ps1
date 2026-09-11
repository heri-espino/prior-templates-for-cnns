param(
    [string]$ResultsRoot = $(Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'prior-templates-cnns\results'),
    [string]$Branch = 'paper_suggestions',
    [string]$ImportClone = $(Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'prior-templates-cnns\paper-results-import')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$SourceRepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ConfirmationSource = Join-Path $ResultsRoot 'budget_confirmation_001'
$ExhaustiveSource = Join-Path $ResultsRoot 'exhaustive_robustness_001'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git is required.'
}
if (-not (Test-Path $ConfirmationSource)) {
    throw "Confirmation results not found: $ConfirmationSource"
}
if (-not (Test-Path $ExhaustiveSource)) {
    throw "Exhaustive results not found: $ExhaustiveSource"
}

$confirmationAnalysis = Join-Path $ConfirmationSource 'analysis\budget_confirmation'
$exhaustiveAnalysis = Join-Path $ExhaustiveSource 'analysis'
if (-not (Test-Path $confirmationAnalysis)) {
    throw "Confirmation analysis directory not found: $confirmationAnalysis"
}
if (-not (Test-Path $exhaustiveAnalysis)) {
    throw "Exhaustive analysis directory not found: $exhaustiveAnalysis"
}

# The managed Windows workstation protects Documents from PowerShell writes.
# Work around that without administrator privileges by cloning the repository
# into LocalAppData, which is user-writable, copying only compact paper-facing
# outputs there, committing, and pushing the requested branch.
$origin = (& git -C $SourceRepoRoot remote get-url origin).Trim()
if (-not $origin) { throw 'Could not determine the origin remote.' }

$importParent = Split-Path -Parent $ImportClone
New-Item -ItemType Directory -Force -Path $importParent | Out-Null
if (Test-Path $ImportClone) {
    Write-Host "Removing previous import clone: $ImportClone"
    Remove-Item -Recurse -Force $ImportClone
}

Write-Host "Creating user-writable import clone under LocalAppData..."
& git clone --single-branch --branch $Branch $origin $ImportClone
if ($LASTEXITCODE -ne 0) { throw "git clone failed with exit code $LASTEXITCODE" }

$ConfirmationDest = Join-Path $ImportClone 'analysis\budget_confirmation_001'
$ExhaustiveDest = Join-Path $ImportClone 'analysis\exhaustive_robustness_001'
New-Item -ItemType Directory -Force -Path $ConfirmationDest | Out-Null
New-Item -ItemType Directory -Force -Path $ExhaustiveDest | Out-Null

# Copy only compact, paper-facing outputs. Training checkpoints and raw model
# trees remain in LocalAppData and are deliberately not committed.
Copy-Item -Path (Join-Path $confirmationAnalysis '*') -Destination $ConfirmationDest -Recurse -Force
Copy-Item -Path (Join-Path $ConfirmationSource 'execution_manifest.json') -Destination (Join-Path $ConfirmationDest 'execution_manifest.json') -Force
Copy-Item -Path (Join-Path $exhaustiveAnalysis '*') -Destination $ExhaustiveDest -Recurse -Force
Copy-Item -Path (Join-Path $ExhaustiveSource 'execution_manifest.json') -Destination (Join-Path $ExhaustiveDest 'execution_manifest.json') -Force

& git -C $ImportClone add analysis/budget_confirmation_001 analysis/exhaustive_robustness_001
if ($LASTEXITCODE -ne 0) { throw "git add failed with exit code $LASTEXITCODE" }

& git -C $ImportClone diff --cached --quiet
$diffCode = $LASTEXITCODE
if ($diffCode -eq 0) {
    Write-Host ''
    Write-Host 'The paper-facing result files are already committed on the branch; nothing new to push.'
} elseif ($diffCode -eq 1) {
    Write-Host ''
    Write-Host 'Files staged for commit:'
    & git -C $ImportClone status --short

    & git -C $ImportClone commit -m 'results: add prospective confirmation and exhaustive robustness outputs'
    if ($LASTEXITCODE -ne 0) {
        throw 'git commit failed. If Git reports missing identity, configure your normal user.name/user.email and rerun this script.'
    }

    & git -C $ImportClone push origin "HEAD:$Branch"
    if ($LASTEXITCODE -ne 0) { throw "git push failed with exit code $LASTEXITCODE" }
} else {
    throw "git diff --cached --quiet failed with exit code $diffCode"
}

Write-Host ''
Write-Host 'Completed result import without writing into the protected Documents checkout.'
Write-Host "Writable clone: $ImportClone"
Write-Host "Remote branch:  $Branch"
Write-Host ''
Write-Host 'The results are now available on GitHub. To make them appear in the original VS Code checkout, try:'
Write-Host "  git pull origin $Branch"
Write-Host ''
Write-Host 'If that managed Documents folder also blocks Git from materializing the new files, open this writable clone in VS Code instead:'
Write-Host "  code `"$ImportClone`""
