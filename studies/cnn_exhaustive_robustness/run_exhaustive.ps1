param(
    [string]$OutputRoot = $(if ($env:OUTPUT_ROOT) { $env:OUTPUT_ROOT } else { Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'prior-templates-cnns\results\exhaustive_robustness_001' }),
    [ValidateSet('cuda','cpu')][string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { 'cuda' }),
    [int]$Workers = $(if ($env:WORKERS) { [int]$env:WORKERS } else { 8 }),
    [int]$ThreadsPerWorker = $(if ($env:THREADS_PER_WORKER) { [int]$env:THREADS_PER_WORKER } else { 2 }),
    [int]$BatchSize = $(if ($env:BATCH_SIZE) { [int]$env:BATCH_SIZE } else { 256 }),
    [string]$CondaEnv = $(if ($env:CNN_ENV_NAME) { $env:CNN_ENV_NAME } else { 'prior-templates-cnns' }),
    [string]$TorchIndexUrl = $(if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { 'https://download.pytorch.org/whl/cu128' })
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($Workers -lt 1 -or $ThreadsPerWorker -lt 1 -or $BatchSize -lt 1) { throw 'Workers, ThreadsPerWorker and BatchSize must be positive.' }

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Push-Location $RepoRoot
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git is required for provenance recording.' }

    $env:DEVICE = $Device
    $env:CNN_ENV_NAME = $CondaEnv
    $env:TORCH_INDEX_URL = $TorchIndexUrl
    . (Join-Path $RepoRoot 'scripts\use_conda_env.ps1')

    $TrainRoot = Join-Path $OutputRoot 'training'
    $EvalRoot = Join-Path $OutputRoot 'evaluation'
    $AnalysisRoot = Join-Path $OutputRoot 'analysis'
    $Manifest = Join-Path $OutputRoot 'execution_manifest.json'
    $Protocol = 'studies/cnn_exhaustive_robustness/PROTOCOL.md'

    Write-Host "Output root: $OutputRoot"
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

    $repoHead = (& git rev-parse HEAD).Trim()
    $protocolCommit = (& git log -n 1 --format=%H -- $Protocol).Trim()
    $protocolSha = (Get-FileHash $Protocol -Algorithm SHA256).Hash.ToLowerInvariant()
    $sources = @(
        $Protocol,
        'studies/cnn_exhaustive_robustness/train_grid.py',
        'studies/cnn_exhaustive_robustness/evaluate_grid.py',
        'studies/cnn_exhaustive_robustness/analyze_grid.py',
        'studies/cnn_exhaustive_robustness/run_exhaustive.ps1',
        'studies/cnn_release_experiment/core.py',
        'scripts/use_conda_env.ps1'
    )
    $sourceHashes = [ordered]@{}
    foreach ($p in $sources) { $sourceHashes[$p] = (Get-FileHash $p -Algorithm SHA256).Hash.ToLowerInvariant() }

    $design = [ordered]@{
        blocks = @(5000..5049)
        n_blocks = 50
        tasks = @('single_shape','two_concepts')
        architectures = @('TinyCNN','TwoLayerCNN')
        profiles = @('template_init','retention_0p1','retention_1','release_early','release_default','release_late')
        epochs = 200
        selected_fidelity_budgets = @(1..16)
        energy_matched_budgets = @(1,2,4,8)
        rankings = @('contrast','auroc','validation_patch')
        planned_models = 1200
    }
    $record = [ordered]@{
        created_utc = [DateTime]::UtcNow.ToString('o')
        repo_head_at_launch = $repoHead
        protocol_commit = $protocolCommit
        protocol_sha256 = $protocolSha
        device = $Device
        workers = $Workers
        threads_per_worker = $ThreadsPerWorker
        evaluation_batch_size = $BatchSize
        conda_environment = $CondaEnv
        source_sha256 = $sourceHashes
        frozen_design = $design
    }

    if (Test-Path $Manifest) {
        $old = Get-Content $Manifest -Raw | ConvertFrom-Json
        if ($old.protocol_commit -ne $protocolCommit -or $old.protocol_sha256 -ne $protocolSha -or
            $old.device -ne $Device -or [int]$old.threads_per_worker -ne $ThreadsPerWorker -or
            [int]$old.evaluation_batch_size -ne $BatchSize -or $old.conda_environment -ne $CondaEnv) {
            throw 'Existing exhaustive manifest is incompatible. Use a new OutputRoot instead of mixing runs.'
        }
        foreach ($p in $sources) {
            if ($old.source_sha256.$p -ne $sourceHashes[$p]) { throw "Source changed since this run started: $p. Use a new OutputRoot." }
        }
    } else {
        $record | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $Manifest
    }

    Write-Host "Protocol commit: $protocolCommit"
    Write-Host "Protocol SHA-256: $protocolSha"
    Write-Host "Workers: $Workers x $ThreadsPerWorker threads"

    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'

    Write-Host ''
    Write-Host '[1/3] Training the frozen 1200-model robustness grid...'
    Invoke-CnnPython 'studies/cnn_exhaustive_robustness/train_grid.py' `
        '--output' $TrainRoot '--start-block' '5000' '--blocks' '50' '--epochs' '200' `
        '--tasks' 'single_shape' 'two_concepts' '--architectures' 'TinyCNN' 'TwoLayerCNN' `
        '--profiles' 'template_init' 'retention_0p1' 'retention_1' 'release_early' 'release_default' 'release_late' `
        '--workers' "$Workers" '--threads-per-worker' "$ThreadsPerWorker"

    Write-Host ''
    Write-Host '[2/3] Evaluating k=1..16 and validation-energy-matched controls on the GPU...'
    Invoke-CnnPython 'studies/cnn_exhaustive_robustness/evaluate_grid.py' `
        '--input' $TrainRoot '--output' $EvalRoot '--device' $Device '--batch-size' "$BatchSize"

    Write-Host ''
    Write-Host '[3/3] Applying the frozen robustness analysis...'
    Invoke-CnnPython 'studies/cnn_exhaustive_robustness/analyze_grid.py' $OutputRoot '--out' $AnalysisRoot

    Write-Host ''
    Write-Host 'Exhaustive robustness study complete.'
    Write-Host "Report:   $AnalysisRoot\REPORT.md"
    Write-Host "Curves:   $AnalysisRoot\selected_fidelity_budget_curves.png"
    Write-Host "Manifest: $Manifest"
}
finally {
    Pop-Location
}
