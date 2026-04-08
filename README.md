# Induction Dose Variability during RSI

**CLIF Version:** 2.1

Site-level pipeline for the **Induction Dose Variability during Rapid Sequence Intubation (RSI)** study. This multi-site CLIF Consortium project identifies adults who received RSI with etomidate or ketamine paired with rocuronium or succinylcholine, builds an analytical dataset with clinical covariates, and runs a pre-specified statistical analysis examining the relative contribution of patient, provider, and hospital factors to variability in induction dosing.

**1st Author:** Vaishvik | **Senior Author:** TBD

## Objective

Quantify the relative contribution of patient, provider, and hospital factors to variability in weight-normalized induction dose (mg/kg) of etomidate or ketamine during rapid sequence intubation. The study uses a linear mixed-effects model with variance decomposition (VPC) and median odds ratio across multi-site CLIF data.

## Important: ED + ICU Data Required

> **This study requires data from Emergency Department (ED) and ICU locations.** The cohort is restricted to patients intubated in the ED or ICU. Your CLIF tables must contain ADT records with `location_category` values for `ed` and `icu` locations.

Additional notes:

- The `billing_provider_id` field in `patient_procedures` is required (except for MIMIC, where it is imputed).
- The `crrt_therapy` table is used if available; the pipeline handles its absence gracefully.
- The pipeline produces both **PHI-containing** outputs (in `output_phi/`) and **de-identified shareable** outputs (in `output_to_share/` and `output_to_share_6hr/`). **Only share the `output_to_share*` directories** with the coordinating center.

## Required CLIF Tables and Fields

### 1. `patient`

| Column               | Description                        |
|----------------------|------------------------------------|
| `patient_id`         | Unique patient identifier          |
| `sex_category`       | Sex category                       |
| `race_category`      | Race category                      |
| `ethnicity_category` | Ethnicity category                 |

### 2. `hospitalization`

| Column                    | Description                       |
|---------------------------|-----------------------------------|
| `patient_id`              | Unique patient identifier         |
| `hospitalization_id`      | Unique hospitalization identifier |
| `admission_dttm`          | Admission date/time               |
| `discharge_dttm`          | Discharge date/time               |
| `age_at_admission`        | Age at admission (years)          |
| `admission_type_category` | Admission type (e.g., ED)         |
| `discharge_category`      | Discharge disposition             |

### 3. `adt`

| Column               | Description                                        |
|----------------------|----------------------------------------------------|
| `hospitalization_id` | Unique hospitalization identifier                  |
| `hospital_id`        | Hospital identifier                                |
| `hospital_type`      | Hospital type (academic, community; excludes LTACH) |
| `location_category`  | Location category (`ed`, `icu`, floor)             |
| `location_type`      | Location subtype (e.g., MICU, SICU)               |
| `in_dttm`            | Location entry date/time                           |
| `out_dttm`           | Location exit date/time                            |

### 4. `hospital_diagnosis`

| Column                 | Description                       |
|------------------------|-----------------------------------|
| `hospitalization_id`   | Unique hospitalization identifier |
| `diagnosis_code`       | ICD diagnosis code                |
| `diagnosis_code_format`| Diagnosis code format (ICD-9/10)  |

Used for Charlson Comorbidity Index (CCI) calculation via `clifpy`.

### 5. `patient_procedures`

| Column                  | Description                       |
|-------------------------|-----------------------------------|
| `hospitalization_id`    | Unique hospitalization identifier |
| `procedure_code`        | CPT/ICD procedure code            |
| `procedure_code_format` | Code format (CPT, ICD-9, ICD-10)  |
| `procedure_billed_dttm` | Procedure billing date/time       |
| `billing_provider_id`   | Billing provider identifier       |

**Required procedure codes:** `31500` (endotracheal intubation), `31600`-`31610` (tracheostomy, for exclusion)

### 6. `medication_admin_intermittent`

| Column                | Description                       |
|-----------------------|-----------------------------------|
| `hospitalization_id`  | Unique hospitalization identifier |
| `admin_dttm`          | Administration date/time          |
| `med_category`        | Medication category               |
| `med_dose`            | Dose amount                       |
| `med_dose_unit`       | Dose unit                         |
| `mar_action_category` | MAR action category               |

**Required `med_category` values:** `etomidate`, `ketamine`, `rocuronium`, `succinylcholine`, `midazolam`, `lorazepam`, `diazepam`, `propofol`, `phenylephrine`, `epinephrine`, `norepinephrine`

### 7. `medication_admin_continuous`

| Column               | Description                       |
|----------------------|-----------------------------------|
| `hospitalization_id` | Unique hospitalization identifier |
| `admin_dttm`         | Administration date/time          |
| `med_category`       | Medication category               |
| `med_dose`           | Dose amount                       |

**Required `med_category` values:** `vasoactives` (norepinephrine, epinephrine, phenylephrine, vasopressin, dopamine, etc.)

### 8. `respiratory_support`

| Column               | Description                       |
|----------------------|-----------------------------------|
| `hospitalization_id` | Unique hospitalization identifier |
| `recorded_dttm`      | Recorded date/time                |
| `device_category`    | Respiratory device category       |
| `tracheostomy`       | Tracheostomy flag                 |

**Required `device_category` values:** `IMV`, `NIPPV`, `High Flow NC`

### 9. `vitals`

| Column               | Description                       |
|----------------------|-----------------------------------|
| `hospitalization_id` | Unique hospitalization identifier |
| `recorded_dttm`      | Vital sign recorded date/time     |
| `vital_category`     | Vital sign category               |
| `vital_value`        | Vital sign value                  |

**Required `vital_category` values:** `heart_rate`, `sbp`, `dbp`, `map`, `spo2`, `weight`

### 10. `crrt_therapy` (optional)

| Column               | Description                       |
|----------------------|-----------------------------------|
| `hospitalization_id` | Unique hospitalization identifier |
| `recorded_dttm`      | Recorded date/time                |

Used to identify CRRT receipt in the 24 hours prior to RSI.

### 11. `microbiology_culture`

Used internally by `clifpy.compute_ase()` for Adult Sepsis Event calculation. No site-specific configuration needed beyond having the table available.

## Cohort Identification

### Inclusion Criteria

1. **Age** >= 18 years at admission
2. **Date range**: Admission and discharge between 2018-01-01 and 2025-12-31
3. **Intubation procedure**: CPT code 31500 with a non-missing `billing_provider_id`
4. **RSI medication pairing**: Induction agent (etomidate or ketamine) + paralytic (rocuronium or succinylcholine) administered within 5 minutes of each other, with `mar_action_category` = `"given"`
5. **Location**: Patient in ED or ICU at the time of RSI medication administration
6. **Mechanical ventilation**: Invasive mechanical ventilation (IMV) initiated within 6 hours of the index induction medication

### Exclusion Criteria

1. Tracheostomy within 24 hours of admission
2. Received both etomidate AND ketamine during the hospitalization
3. Benzodiazepine (lorazepam, diazepam, midazolam) or propofol administered within 60 minutes prior to RSI
4. Prior intubation during the same hospitalization (only first RSI event is retained)
5. Non-feasible induction dose (etomidate: <2 mg or >100 mg; ketamine: <15 mg or >300 mg)
6. Non-physiological weight (<20 kg or >300 kg)

## Expected Results

### `output_phi/` -- Contains PHI. Do NOT share.

| File | Description |
|------|-------------|
| `rsi_cohort.parquet` | Full cohort with patient identifiers |
| `rsi_analytical_dataset.parquet` | Wide analytical dataset (~100 columns) |
| `rsi_analysis_dataset_2.parquet` | Focused modeling dataset (~31 columns) |
| `subanalysis_rsi_timing_pairs.parquet` | Induction-paralytic timing pair data |
| `subanalysis_weight_timing.parquet` | Weight measurement timing data |

### `output_to_share/` -- De-identified. Share with coordinating center.

| File | Description |
|------|-------------|
| `consort_cohort.json` / `consort_cohort.csv` | CONSORT-style cohort flow diagram data |
| `subanalysis_rsi_timing_stats.csv` | Induction-paralytic timing statistics |
| `subanalysis_weight_timing_stats.csv` | Weight timing statistics |
| `plots/` | Distribution plots (PNG) |

### `output_to_share_6hr/` -- De-identified. Share with coordinating center.

| Directory | Description |
|-----------|-------------|
| `diagnostics/` | Data quality checks, missingness reports, exclusion waterfalls |
| `figures/` | Dose distributions, variance decomposition, trend plots, caterpillar plots |
| `tables/` | Table 1, dose summaries, provider summaries, stratification tables |
| `models/` | Model coefficients, ICC/VPC results, sensitivity analyses |

## Instructions

### 1. Configure `clif_config.json`

Copy the template and fill in site-specific values:

```bash
cp clif_config_template.json clif_config.json
```

```json
{
  "site_name": "your_site",
  "data_directory": "/path/to/clif/tables",
  "filetype": "parquet",
  "timezone": "America/Chicago",
  "output_directory": "./output_phi",
  "output_to_share_directory": "./output_to_share"
}
```

| Field | Description |
|-------|-------------|
| `site_name` | Your site identifier (e.g., `"site_a"`) |
| `data_directory` | Absolute path to directory containing CLIF 2.1 parquet files |
| `filetype` | File format of CLIF tables, typically `"parquet"` |
| `timezone` | IANA timezone string for your site (e.g., `"America/Chicago"`, `"US/Eastern"`) |
| `output_directory` | Path for PHI-containing outputs (default: `"./output_phi"`) |
| `output_to_share_directory` | Path for de-identified shareable outputs (default: `"./output_to_share"`) |

### 2. Set Up Environment

This is a **hybrid Python + R** pipeline:

- **Steps 1-2** are Python, managed by [uv](https://docs.astral.sh/uv/)
- **Step 3** is R/Quarto, requiring [R](https://cran.r-project.org/) and [Quarto](https://quarto.org/)

#### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [Python](https://www.python.org/) | >= 3.12 | Cohort identification and dataset building |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager (replaces pip) |
| [R](https://cran.r-project.org/) | >= 4.1 | Statistical analysis |
| [Quarto](https://quarto.org/) | latest | Renders the R analysis document |

#### macOS / Linux

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
uv sync

# Install R packages (run from an R console)
install.packages(c(
  "tidyverse", "dplyr", "lme4", "lmerTest", "broom.mixed",
  "boot", "gtsummary", "here", "arrow", "mice",
  "patchwork", "scales", "jsonlite"
))
```

#### Windows (PowerShell)

```powershell
# Install uv (if not already installed)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Python dependencies
uv sync

# Install R packages (run from an R console)
install.packages(c(
  "tidyverse", "dplyr", "lme4", "lmerTest", "broom.mixed",
  "boot", "gtsummary", "here", "arrow", "mice",
  "patchwork", "scales", "jsonlite"
))
```

> **Note:** If you encounter a PowerShell execution policy error when running scripts, run:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### 3. Run the Pipeline

#### macOS / Linux

```bash
bash run.sh
```

Or run each step manually:

```bash
uv run python code/01_cohort.py
uv run python code/02_dataset.py
quarto render code/03_Induction_Dose_Variability_site_analysis_6hr.qmd
```

#### Windows (PowerShell)

```powershell
.\run.ps1
```

Or run each step manually:

```powershell
uv run python code/01_cohort.py
uv run python code/02_dataset.py
quarto render code/03_Induction_Dose_Variability_site_analysis_6hr.qmd
```

All output is logged to `logs/run_YYYYMMDD_HHMMSS.log`.

> **Note:** If the Quarto step fails, you can open `code/03_Induction_Dose_Variability_site_analysis_6hr.qmd` in RStudio and render it directly from there.

## Pipeline Steps

| Step | Script | Language | Description |
|------|--------|----------|-------------|
| 1 | `code/01_cohort.py` | Python (marimo) | Cohort identification: applies inclusion/exclusion criteria, outputs `rsi_cohort.parquet` and CONSORT flow |
| 2 | `code/02_dataset.py` | Python (marimo) | Analytical dataset: joins clinical covariates (CCI, SOFA, ASE, vitals, respiratory support, vasopressors, outcomes) |
| 3 | `code/03_Induction_Dose_Variability_site_analysis_6hr.qmd` | R (Quarto) | Statistical analysis: descriptive tables, mixed-effects models, variance decomposition, sensitivity analyses |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `quarto render` fails to find R | Ensure R is on your system PATH, or set the `QUARTO_R` environment variable to point to your R installation |
| PowerShell execution policy error | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Quarto step fails with R package errors | Open the `.qmd` file in RStudio and render from there to see detailed R error messages |
| `uv` not found after install | Restart your terminal/PowerShell session so the updated PATH takes effect |
