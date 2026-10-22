# Trains the four experiments from specification section 12, one after another.
#
# The machine has 8 logical cores and no GPU, so the experiments are run
# sequentially: running them in parallel would oversubscribe the CPU and make
# every run slower rather than finishing sooner.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File run_all_experiments.ps1
#   powershell -ExecutionPolicy Bypass -File run_all_experiments.ps1 -Experiments E1,E2

param(
    [string]$Experiments = "E1,E2,E3,E4",
    [int]$EpochsE4 = 20,
    [switch]$NoClassWeight
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:TF_CPP_MIN_LOG_LEVEL = "3"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
New-Item -ItemType Directory -Force -Path "results" | Out-Null

$classWeightArgs = @()
if ($NoClassWeight) { $classWeightArgs = @("--no-class-weight") }

$grid = @{ "E1" = 35; "E2" = 35; "E3" = 35; "E4" = $EpochsE4 }
$overall = Get-Date

foreach ($exp in $Experiments.Split(",")) {
    $exp = $exp.Trim().ToUpper()
    if (-not $grid.ContainsKey($exp)) { throw "Unknown experiment '$exp'" }

    $epochs = $grid[$exp]
    $log = "results\train_$exp.log"
    Write-Host ("=" * 68)
    Write-Host "Starting $exp ($epochs epochs) -> $log"
    Write-Host ("=" * 68)

    $started = Get-Date
    # Each run gets its own log file. Never pipe training output through a
    # PowerShell filter: the progress bar is carriage-return based and the
    # pipeline stalls until the process exits.
    & python -u -W ignore -m src.train --experiment $exp --epochs $epochs @classWeightArgs `
        *> $log
    $code = $LASTEXITCODE
    $minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)

    if ($code -ne 0) {
        Write-Host "FAILED $exp after $minutes min (exit $code). Tail of ${log}:"
        Get-Content $log -Tail 30
        exit $code
    }

    $best = (Get-Content "results\${exp}_train_summary.json" -Raw | ConvertFrom-Json).best_val_accuracy
    Write-Host "OK $exp in $minutes min, best val accuracy $([math]::Round($best * 100, 2))%"
}

$total = [math]::Round(((Get-Date) - $overall).TotalMinutes, 1)
Write-Host ("=" * 68)
Write-Host "All experiments finished in $total min"
Write-Host ("=" * 68)