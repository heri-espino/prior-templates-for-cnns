param([Parameter(ValueFromRemainingArguments=$true)][string[]]$RunArgs)
& "$PSScriptRoot/studies/cnn_release_experiment/run.ps1" @RunArgs
exit $LASTEXITCODE
