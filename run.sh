#!/bin/bash
set -e

LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== Run started: $(date) ===" | tee "$LOG"

uv run python code/01_cohort.py  2>&1 | tee -a "$LOG"
uv run python code/02_dataset.py 2>&1 | tee -a "$LOG"
quarto render code/03_Induction_Dose_Variability_site_analysis_6hr.qmd 2>&1 | tee -a "$LOG"

echo "=== Run complete: $(date) ===" | tee -a "$LOG"
