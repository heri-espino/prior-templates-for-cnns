param(
    [string]$OutputRoot = $(if ($env:OUTPUT_ROOT) { $env:OUTPUT_ROOT } else { Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'prior-templates-cnns\results\budget_confirmation_001' }),
    [ValidateSet('cuda','cpu')][string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { 'cuda' }),
    [int]$BatchSize = $(if ($env:BATCH_SIZE) { [int]$env:BATCH_SIZE } else { 64 }),
    [int]$Threads = $(if ($env:THREADS) { [int]$env:THREADS } else { 2 }),
    [string]$CondaEnv = $(if ($env:CNN_ENV_NAME) { $env:CNN_ENV_NAME } else { 'prior-templates-cnns' }),
    [string]$TorchIndexUrl = $(if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { 'https://download.pytorch.org/whl/cu128' })
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location $RepoRoot
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git is required for provenance recording.' }

    $env:DEVICE = $Device
    $env:CNN_ENV_NAME = $CondaEnv
    $env:TORCH_INDEX_URL = $TorchIndexUrl
    . (Join-Path $RepoRoot 'scripts\use_conda_env.ps1')

    $TrainRoot = Join-Path $OutputRoot 'training'
    $PatchRoot = Join-Path $OutputRoot 'patch_eval'
    $EnergyRoot = Join-Path $OutputRoot 'energy_eval'
    $AnalysisRoot = Join-Path $OutputRoot 'analysis\budget_confirmation'
    $Manifest = Join-Path $OutputRoot 'execution_manifest.json'
    $Protocol = 'studies/cnn_budget_confirmation/PROTOCOL.md'

    Write-Host "Output root: $OutputRoot"
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

    function Get-GitBlobId([string]$Path) {
        $value = (& git rev-parse "HEAD:$Path").Trim()
        if ($LASTEXITCODE -ne 0 -or -not $value) { throw "Could not resolve committed Git blob for: $Path" }
        return $value
    }

    $repoHead = (& git rev-parse HEAD).Trim()
    $protocolCommit = (& git log -n 1 --format=%H -- $Protocol).Trim()
    $protocolBlob = Get-GitBlobId $Protocol
    $sources = @(
        $Protocol,
        'studies/cnn_release_experiment/core.py',
        'studies/cnn_release_experiment/experiment.py',
        'studies/cnn_patch_robustness/evaluate_gpu.py',
        'studies/cnn_patch_energy_control/evaluate_gpu.py',
        'studies/cnn_budget_confirmation/analyze.py',
        'studies/cnn_budget_confirmation/run_confirmation.ps1',
        'scripts/use_conda_env.ps1'
    )
    $sourceGitBlobs = [ordered]@{}
    foreach ($p in $sources) { $sourceGitBlobs[$p] = Get-GitBlobId $p }

    $design = [ordered]@{
        task = 'two_concepts'
        architectures = @('TinyCNN','TwoLayerCNN')
        conditions = @('template_retention_1','template_release')
        blocks = @(4000..4019)
        epochs = 200
        release_start = 10
        release_end = 80
        checkpoint_epochs = @(1,5,10,20,40,80,120,160,200)
        intervention_budgets = @(1,2,4,8)
        rankings = @('contrast','auroc','validation_patch')
        primary = 'TinyCNN / contrast / selected-fidelity budget contrast B'
    }
    $record = [ordered]@{
        created_utc = [DateTime]::UtcNow.ToString('o')
        repo_head_at_launch = $repoHead
        protocol_commit = $protocolCommit
        protocol_git_blob = $protocolBlob
        device = $Device
        batch_size = $BatchSize
        training_threads = $Threads
        conda_environment = $CondaEnv
        source_git_blobs = $sourceGitBlobs
        frozen_design = $design
    }

    if (Test-Path $Manifest) {
        $old = Get-Content $Manifest -Raw | ConvertFrom-Json
        if ($old.protocol_commit -ne $protocolCommit -or $old.protocol_git_blob -ne $protocolBlob -or
            $old.device -ne $Device -or [int]$old.batch_size -ne $BatchSize -or
            [int]$old.training_threads -ne $Threads -or $old.conda_environment -ne $CondaEnv) {
            throw 'Existing confirmation manifest is incompatible. Use a new OutputRoot instead of mixing runs.'
        }
        foreach ($p in $sources) {
            if ($old.source_git_blobs.$p -ne $sourceGitBlobs[$p]) { throw "Source changed since this run started: $p. Use a new OutputRoot." }
        }
    } else {
        $record | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $Manifest
    }

    Write-Host "Protocol commit: $protocolCommit"
    Write-Host "Protocol Git blob: $protocolBlob"
    Write-Host ''
    Write-Host '[1/4] Training 80 fresh models on blocks 4000-4019...'
    Invoke-CnnPython 'studies/cnn_release_experiment/experiment.py' `
        '--output' $TrainRoot '--epochs' '200' '--blocks' '20' '--start-block' '4000' `
        '--tasks' 'two_concepts' '--architectures' 'TinyCNN' 'TwoLayerCNN' `
        '--conditions' 'template_retention_1' 'template_release' `
        '--release-start' '10' '--release-end' '80' `
        '--checkpoints' '1' '5' '10' '20' '40' '80' '120' '160' '200' '--threads' "$Threads"

    Write-Host ''
    Write-Host '[2/4] Evaluating selected fidelity, random-control U, rankings, and k={1,2,4,8}...'
    Invoke-CnnPython 'studies/cnn_patch_robustness/evaluate_gpu.py' `
        '--input' $TrainRoot '--output' $PatchRoot '--device' $Device '--batch-size' "$BatchSize" '--epochs' '200'

    Write-Host ''
    Write-Host '[3/4] Evaluating validation-energy-matched controls...'
    Invoke-CnnPython 'studies/cnn_patch_energy_control/evaluate_gpu.py' `
        '--input' $TrainRoot '--output' $EnergyRoot '--device' $Device '--batch-size' "$BatchSize" '--epochs' '200'

    Write-Host ''
    Write-Host '[4/4] Applying the frozen prospective analysis and decision rule...'
    Invoke-CnnPython 'studies/cnn_budget_confirmation/analyze.py' $OutputRoot '--out' $AnalysisRoot

    Write-Host ''
    Write-Host 'Confirmation run complete.'
    Write-Host "Report:   $AnalysisRoot\REPORT.md"
    Write-Host "Decision: $AnalysisRoot\decision.json"
    Write-Host "Manifest: $Manifest"
}
finally {
    Pop-Location
}
