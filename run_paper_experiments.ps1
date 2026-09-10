param(
    [ValidateSet('cuda','cpu')][string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { 'cuda' }),
    [string]$CondaEnv = $(if ($env:CNN_ENV_NAME) { $env:CNN_ENV_NAME } else { 'prior-templates-cnns' }),
    [int]$Workers = $(if ($env:WORKERS) { [int]$env:WORKERS } else { 8 }),
    [int]$ThreadsPerWorker = $(if ($env:THREADS_PER_WORKER) { [int]$env:THREADS_PER_WORKER } else { 2 }),
    [int]$ExhaustiveBatchSize = $(if ($env:EXHAUSTIVE_BATCH_SIZE) { [int]$env:EXHAUSTIVE_BATCH_SIZE } else { 256 }),
    [int]$ConfirmationBatchSize = $(if ($env:CONFIRMATION_BATCH_SIZE) { [int]$env:CONFIRMATION_BATCH_SIZE } else { 64 }),
    [int]$ConfirmationThreads = $(if ($env:CONFIRMATION_THREADS) { [int]$env:CONFIRMATION_THREADS } else { 2 })
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
Push-Location $RepoRoot
try {
    Write-Host '=== Stage D: frozen prospective confirmation (blocks 4000-4019) ==='
    & (Join-Path $RepoRoot 'studies\cnn_budget_confirmation\run_confirmation.ps1') `
        -Device $Device -CondaEnv $CondaEnv -BatchSize $ConfirmationBatchSize -Threads $ConfirmationThreads
    if ($LASTEXITCODE -ne 0) { throw "Confirmation stage failed with exit code $LASTEXITCODE" }

    Write-Host ''
    Write-Host '=== Stage E: separate exhaustive robustness map (blocks 5000-5049) ==='
    & (Join-Path $RepoRoot 'studies\cnn_exhaustive_robustness\run_exhaustive.ps1') `
        -Device $Device -CondaEnv $CondaEnv -Workers $Workers -ThreadsPerWorker $ThreadsPerWorker -BatchSize $ExhaustiveBatchSize
    if ($LASTEXITCODE -ne 0) { throw "Exhaustive stage failed with exit code $LASTEXITCODE" }

    Write-Host ''
    Write-Host 'All planned paper experiments finished.'
    Write-Host 'Primary confirmation report: results\budget_confirmation_001\analysis\budget_confirmation\REPORT.md'
    Write-Host 'Exhaustive robustness report: results\exhaustive_robustness_001\analysis\REPORT.md'
}
finally {
    Pop-Location
}
