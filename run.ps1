<#
.SYNOPSIS
    Run the Induction Dose Variability RSI pipeline (all 3 steps).
.DESCRIPTION
    PowerShell equivalent of run.sh for Windows environments.
    Executes cohort identification, dataset building, and Quarto
    statistical analysis sequentially, logging all output to logs/.
.EXAMPLE
    .\run.ps1
#>

$ErrorActionPreference = "Stop"

# --- Log setup ---------------------------------------------------------------
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir    = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$logFile = Join-Path $logDir "run_$timestamp.log"

function Write-Log {
    param([string]$Message)
    $Message | Tee-Object -FilePath $logFile -Append
}

function Invoke-Step {
    param(
        [string]$Command,
        [string[]]$Arguments
    )
    Write-Log "--- Running: $Command $($Arguments -join ' ') ---"
    & $Command @Arguments 2>&1 | Tee-Object -FilePath $logFile -Append
    if ($LASTEXITCODE -ne 0) {
        $msg = "FAILED (exit code $LASTEXITCODE): $Command $($Arguments -join ' ')"
        Write-Log $msg
        throw $msg
    }
}

# --- Pipeline ----------------------------------------------------------------
Write-Log "=== Run started: $(Get-Date) ==="

Invoke-Step "uv" @("run", "python", "code/01_cohort.py")
Invoke-Step "uv" @("run", "python", "code/02_dataset.py")
Invoke-Step "quarto" @("render", "code/03_Induction_Dose_Variability_site_analysis_6hr.qmd")

Write-Log "=== Run complete: $(Get-Date) ==="
