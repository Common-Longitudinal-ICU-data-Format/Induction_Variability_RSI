import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import json
    from pathlib import Path
    from clifpy.tables import (
        HospitalDiagnosis,
        Vitals,
        RespiratorySupport,
        MedicationAdminContinuous,
        MedicationAdminIntermittent,
        CrrtTherapy,
        Adt,
        PatientProcedures,
    )
    from clifpy import calculate_cci, compute_sofa_polars
    from clifpy.utils.ase import compute_ase

    return (
        Adt,
        CrrtTherapy,
        HospitalDiagnosis,
        MedicationAdminContinuous,
        Path,
        PatientProcedures,
        RespiratorySupport,
        Vitals,
        calculate_cci,
        compute_ase,
        compute_sofa_polars,
        json,
        mo,
        pl,
    )


@app.cell
def _(mo):
    mo.md("""
    # 02 Analytical Dataset: RSI Induction Variability

    Build the full analytical dataset (1 row per hospitalization) with medications,
    demographics, clinical scores, vitals, respiratory support, vasopressors, and outcomes.
    """)
    return


@app.cell
def _(Path, json):
    config_path = Path(__file__).parent.parent / "clif_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    SITE = config["site_name"]
    DATA_DIR = config["data_directory"]
    FILETYPE = config["filetype"]
    TIMEZONE = config["timezone"]
    OUTPUT_DIR = Path(config["output_directory"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Site: {SITE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output (PHI): {OUTPUT_DIR}")
    return DATA_DIR, FILETYPE, OUTPUT_DIR, SITE, TIMEZONE


@app.cell
def _(OUTPUT_DIR, pl):
    # Load cohort from 01_cohort output
    cohort = pl.read_parquet(OUTPUT_DIR / "rsi_cohort.parquet")
    print(f"Cohort loaded: {cohort.height:,} hospitalizations")
    print(f"Columns: {cohort.columns}")
    cohort.head()
    return (cohort,)


@app.cell
def _(cohort, pl):
    # Basic columns & dose categories
    base = cohort.select([
        "hospitalization_id",
        "patient_id",
        "index_dttm",
        "imv_dttm",
        "admin_dttm_par",
        "admission_dttm",
        "discharge_dttm",
        "age_at_admission",
        "sex_category",
        "race_category",
        "ethnicity_category",
        "admission_type_category",
        "discharge_category",
        "med_category_ind",
        "med_dose_ind",
        "med_dose_unit_ind",
        "med_category_par",
        "med_dose_par",
        "med_dose_unit_par",
        "weight_kg",
        "weight_source",
        "weight_recorded_dttm",
        "weight_to_rsi_hours",
        "post_rsi_weight_kg",
        "post_rsi_weight_recorded_dttm",
        "post_rsi_weight_to_rsi_hours",
        "induction_dose_per_kg",
    ])

    # NMB dose per kg
    base = base.with_columns(
        (pl.col("med_dose_par") / pl.col("weight_kg")).alias("nmb_dose_mg_kg")
    )

    # Induction categorical dose (mg/kg)
    base = base.with_columns(
        pl.when(pl.col("med_category_ind") == "etomidate")
        .then(
            pl.when(pl.col("induction_dose_per_kg") < 0.05).then(pl.lit("<0.05"))
            .when(pl.col("induction_dose_per_kg") < 0.10).then(pl.lit("0.05-0.09"))
            .when(pl.col("induction_dose_per_kg") < 0.15).then(pl.lit("0.10-0.14"))
            .when(pl.col("induction_dose_per_kg") < 0.20).then(pl.lit("0.15-0.19"))
            .when(pl.col("induction_dose_per_kg") < 0.25).then(pl.lit("0.20-0.24"))
            .when(pl.col("induction_dose_per_kg") < 0.30).then(pl.lit("0.25-0.29"))
            .when(pl.col("induction_dose_per_kg") < 0.35).then(pl.lit("0.30-0.34"))
            .when(pl.col("induction_dose_per_kg") < 0.40).then(pl.lit("0.35-0.39"))
            .when(pl.col("induction_dose_per_kg") < 0.45).then(pl.lit("0.40-0.44"))
            .when(pl.col("induction_dose_per_kg") < 0.50).then(pl.lit("0.45-0.49"))
            .when(pl.col("induction_dose_per_kg") < 0.55).then(pl.lit("0.50-0.54"))
            .when(pl.col("induction_dose_per_kg") < 0.60).then(pl.lit("0.55-0.59"))
            .otherwise(pl.lit(">0.60"))
        )
        .when(pl.col("med_category_ind") == "ketamine")
        .then(
            pl.when(pl.col("induction_dose_per_kg") < 0.3).then(pl.lit("<0.3"))
            .when(pl.col("induction_dose_per_kg") < 0.5).then(pl.lit("0.3-0.49"))
            .when(pl.col("induction_dose_per_kg") < 0.7).then(pl.lit("0.5-0.69"))
            .when(pl.col("induction_dose_per_kg") < 0.9).then(pl.lit("0.7-0.89"))
            .when(pl.col("induction_dose_per_kg") < 1.1).then(pl.lit("0.9-1.09"))
            .when(pl.col("induction_dose_per_kg") < 1.3).then(pl.lit("1.1-1.29"))
            .otherwise(pl.lit(">=1.3"))
        )
        .otherwise(pl.lit(None))
        .alias("induction_categorical_mg_kg")
    )

    # NMB categorical dose (mg/kg)
    base = base.with_columns(
        pl.when(pl.col("med_category_par") == "rocuronium")
        .then(
            pl.when(pl.col("nmb_dose_mg_kg") < 0.4).then(pl.lit("<0.4"))
            .when(pl.col("nmb_dose_mg_kg") < 0.6).then(pl.lit("0.4-0.59"))
            .when(pl.col("nmb_dose_mg_kg") < 0.8).then(pl.lit("0.6-0.79"))
            .when(pl.col("nmb_dose_mg_kg") < 1.0).then(pl.lit("0.8-0.99"))
            .when(pl.col("nmb_dose_mg_kg") < 1.2).then(pl.lit("1.0-1.19"))
            .when(pl.col("nmb_dose_mg_kg") < 1.4).then(pl.lit("1.2-1.39"))
            .when(pl.col("nmb_dose_mg_kg") < 1.6).then(pl.lit("1.4-1.59"))
            .when(pl.col("nmb_dose_mg_kg") < 1.9).then(pl.lit("1.6-1.89"))
            .otherwise(pl.lit(">=1.9"))
        )
        .when(pl.col("med_category_par") == "succinylcholine")
        .then(
            pl.when(pl.col("nmb_dose_mg_kg") < 0.25).then(pl.lit("<0.25"))
            .when(pl.col("nmb_dose_mg_kg") < 0.50).then(pl.lit("0.25-0.49"))
            .when(pl.col("nmb_dose_mg_kg") < 0.75).then(pl.lit("0.50-0.74"))
            .when(pl.col("nmb_dose_mg_kg") < 1.00).then(pl.lit("0.75-0.99"))
            .when(pl.col("nmb_dose_mg_kg") < 1.25).then(pl.lit("1.00-1.24"))
            .when(pl.col("nmb_dose_mg_kg") < 1.50).then(pl.lit("1.25-1.49"))
            .otherwise(pl.lit(">=1.50"))
        )
        .otherwise(pl.lit(None))
        .alias("nmb_categorical_mg_kg")
    )

    print(f"Base dataset: {base.height:,} rows, {len(base.columns)} columns")
    base.head()
    return (base,)


@app.cell
def _(DATA_DIR, FILETYPE, MedicationAdminContinuous, TIMEZONE, cohort, pl):
    # Push-dose pressors: any vasoactive continuous med within 2 min before paralytic admin
    PUSH_DOSE_MEDS = ["phenylephrine", "epinephrine", "norepinephrine"]

    _cont_table = MedicationAdminContinuous.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "hospitalization_id": cohort["hospitalization_id"].to_list(),
            "med_category": PUSH_DOSE_MEDS,
        },
    )
    _cont_pd = _cont_table.df.copy()
    _cont_pd["admin_dttm"] = _cont_pd["admin_dttm"].dt.tz_localize(None)
    cont_vasoactive = pl.from_pandas(_cont_pd)
    del _cont_pd

    cont_vasoactive = cont_vasoactive.with_columns(
        pl.col("med_category").str.to_lowercase().alias("med_category")
    )

    # Join with cohort to get admin_dttm_par per hospitalization
    _push_check = (
        cohort.select(["hospitalization_id", "admin_dttm_par"])
        .join(
            cont_vasoactive.select(["hospitalization_id", "admin_dttm", "med_category"]),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            ((pl.col("admin_dttm_par") - pl.col("admin_dttm")).dt.total_seconds() / 60)
            .alias("min_before_par")
        )
        .filter((pl.col("min_before_par") >= 0) & (pl.col("min_before_par") <= 2))
    )

    push_dose = cohort.select("hospitalization_id")
    for med in PUSH_DOSE_MEDS:
        _ids = _push_check.filter(pl.col("med_category") == med)["hospitalization_id"].unique()
        push_dose = push_dose.with_columns(
            pl.col("hospitalization_id").is_in(_ids).cast(pl.Int8).alias(f"push_dose_{med}")
        )

    print(f"Push-dose pressors computed for {push_dose.height:,} hospitalizations")
    for med in PUSH_DOSE_MEDS:
        n = push_dose[f"push_dose_{med}"].sum()
        print(f"  {med}: {n}")
    return (push_dose,)


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    HospitalDiagnosis,
    TIMEZONE,
    calculate_cci,
    cohort,
    pl,
):
    # Charlson Comorbidity Index (CCI)
    diag_table = HospitalDiagnosis.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort["hospitalization_id"].to_list()},
    )

    # calculate_cci returns pandas despite type hint
    cci_pd = calculate_cci(diag_table)
    cci_df = pl.from_pandas(cci_pd).select([
        "hospitalization_id",
        pl.col("cci_score").alias("cci"),
    ])

    print(f"CCI computed for {cci_df.height:,} hospitalizations")
    print(f"  Median CCI: {cci_df['cci'].median()}")
    return (cci_df,)


@app.cell
def _(DATA_DIR, FILETYPE, TIMEZONE, cohort, compute_sofa_polars, pl):
    # SOFA score (24hr prior to index_dttm)
    sofa_cohort = cohort.select([
        "hospitalization_id",
        (pl.col("index_dttm") - pl.duration(hours=24)).alias("start_dttm"),
        pl.col("index_dttm").alias("end_dttm"),
    ])

    sofa_result = compute_sofa_polars(
        data_directory=DATA_DIR,
        cohort_df=sofa_cohort,
        filetype=FILETYPE,
        timezone=TIMEZONE,
    )

    sofa_df = sofa_result.select([
        "hospitalization_id",
        pl.col("sofa_total").alias("sofa_24hr_prior_intubation"),
    ])

    # SOFA 6hr window
    sofa_cohort_6hr = cohort.select([
        "hospitalization_id",
        (pl.col("index_dttm") - pl.duration(hours=6)).alias("start_dttm"),
        pl.col("index_dttm").alias("end_dttm"),
    ])

    sofa_result_6hr = compute_sofa_polars(
        data_directory=DATA_DIR,
        cohort_df=sofa_cohort_6hr,
        filetype=FILETYPE,
        timezone=TIMEZONE,
    )

    sofa_6hr_df = sofa_result_6hr.select([
        "hospitalization_id",
        pl.col("sofa_total").alias("sofa_6hr_prior_intubation"),
    ])

    sofa_df = sofa_df.join(sofa_6hr_df, on="hospitalization_id", how="outer_coalesce")

    print(f"SOFA computed for {sofa_df.height:,} hospitalizations")
    print(f"  Median SOFA 24hr: {sofa_df['sofa_24hr_prior_intubation'].median()}")
    print(f"  Median SOFA 6hr: {sofa_df['sofa_6hr_prior_intubation'].median()}")
    return (sofa_df,)


@app.cell
def _(CrrtTherapy, DATA_DIR, FILETYPE, TIMEZONE, cohort, pl):
    # CRRT within 24hr prior to index_dttm
    crrt_table = CrrtTherapy.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort["hospitalization_id"].to_list()},
    )
    _crrt_pd = crrt_table.df.copy()
    _crrt_pd["recorded_dttm"] = _crrt_pd["recorded_dttm"].dt.tz_localize(None)
    crrt_pl = pl.from_pandas(_crrt_pd)
    del _crrt_pd

    _crrt_check = (
        cohort.select(["hospitalization_id", "index_dttm"])
        .join(
            crrt_pl.select(["hospitalization_id", "recorded_dttm"]),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            ((pl.col("index_dttm") - pl.col("recorded_dttm")).dt.total_seconds() / 3600)
            .alias("hours_before")
        )
        .filter((pl.col("hours_before") >= 0) & (pl.col("hours_before") <= 24))
    )

    _crrt_ids = _crrt_check["hospitalization_id"].unique()

    _crrt_ids_6hr = _crrt_check.filter(pl.col("hours_before") <= 6)["hospitalization_id"].unique()

    crrt_df = cohort.select("hospitalization_id").with_columns(
        pl.col("hospitalization_id").is_in(_crrt_ids).cast(pl.Int8).alias("crrt_24hr_prior_intubation"),
        pl.col("hospitalization_id").is_in(_crrt_ids_6hr).cast(pl.Int8).alias("crrt_6hr_prior_intubation"),
    )

    print(f"CRRT 24hr prior: {crrt_df['crrt_24hr_prior_intubation'].sum()} hospitalizations")
    print(f"CRRT 6hr prior: {crrt_df['crrt_6hr_prior_intubation'].sum()} hospitalizations")
    return (crrt_df,)


@app.cell
def _(DATA_DIR, FILETYPE, TIMEZONE, cohort, compute_ase, pl):
    # Adult Sepsis Event (ASE) within 7 days prior to index_dttm
    ase_pd = compute_ase(
        hospitalization_ids=cohort["hospitalization_id"].to_list(),
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
    )

    ase_pl = pl.from_pandas(ase_pd)
    _tz_cols = [name for name, dtype in ase_pl.schema.items() if getattr(dtype, "time_zone", None) is not None]
    if _tz_cols:
        ase_pl = ase_pl.with_columns(pl.col(c).dt.replace_time_zone(None) for c in _tz_cols)

    # Filter to onset within 7 days before index_dttm
    _ase_check = (
        cohort.select(["hospitalization_id", "index_dttm"])
        .join(
            ase_pl.filter(pl.col("sepsis_wo_lactate") == 1)
            .select(["hospitalization_id", "ase_onset_wo_lactate_dttm"]),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            ((pl.col("index_dttm") - pl.col("ase_onset_wo_lactate_dttm")).dt.total_seconds() / 86400)
            .alias("days_before")
        )
        .filter((pl.col("days_before") >= 0) & (pl.col("days_before") <= 7))
    )

    _ase_ids = _ase_check["hospitalization_id"].unique()

    # Presumed infection within 7 days prior
    _infection_check = (
        cohort.select(["hospitalization_id", "index_dttm"])
        .join(
            ase_pl.filter(pl.col("presumed_infection") == 1)
            .select(["hospitalization_id", "presumed_infection_onset_dttm"]),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            ((pl.col("index_dttm") - pl.col("presumed_infection_onset_dttm")).dt.total_seconds() / 86400)
            .alias("days_before")
        )
        .filter((pl.col("days_before") >= 0) & (pl.col("days_before") <= 7))
    )
    _infection_ids = _infection_check["hospitalization_id"].unique()

    ase_df = cohort.select("hospitalization_id").with_columns(
        pl.col("hospitalization_id").is_in(_ase_ids).cast(pl.Int8).alias("ASE_7d_prior"),
        pl.col("hospitalization_id").is_in(_infection_ids).cast(pl.Int8).alias("presumed_infection_7d_prior"),
    )

    print(f"ASE 7d prior: {ase_df['ASE_7d_prior'].sum()} hospitalizations")
    return (ase_df,)


@app.cell
def _(DATA_DIR, FILETYPE, TIMEZONE, Vitals, cohort, pl):
    # Vitals: 24hr and 1hr prior to index_dttm
    VITAL_CATS = ["heart_rate", "sbp", "dbp", "map", "spo2"]

    vitals_table = Vitals.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "hospitalization_id": cohort["hospitalization_id"].to_list(),
            "vital_category": VITAL_CATS,
        },
    )
    _vit_pd = vitals_table.df.copy()
    _vit_pd["recorded_dttm"] = _vit_pd["recorded_dttm"].dt.tz_localize(None)
    vitals_pl = pl.from_pandas(_vit_pd)
    del _vit_pd

    vitals_pl = vitals_pl.with_columns(
        pl.col("vital_category").str.to_lowercase().alias("vital_category")
    )

    # Join with cohort to get index_dttm
    _vit_joined = (
        cohort.select(["hospitalization_id", "index_dttm"])
        .join(
            vitals_pl.select(["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"]),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            ((pl.col("index_dttm") - pl.col("recorded_dttm")).dt.total_seconds() / 3600)
            .alias("hours_before")
        )
        .filter(pl.col("hours_before") >= 0)
    )

    def _compute_vitals_window(df, max_hours, suffix):
        w = df.filter(pl.col("hours_before") <= max_hours)
        results = []
        # Highest heart rate
        hr = w.filter(pl.col("vital_category") == "heart_rate").group_by("hospitalization_id").agg(
            pl.col("vital_value").max().alias(f"highest_hr_{suffix}")
        )
        results.append(hr)
        # Lowest sbp, dbp, map, spo2
        for cat in ["sbp", "dbp", "map", "spo2"]:
            low = w.filter(pl.col("vital_category") == cat).group_by("hospitalization_id").agg(
                pl.col("vital_value").min().alias(f"lowest_{cat}_{suffix}")
            )
            results.append(low)
        # Join all
        out = results[0]
        for r in results[1:]:
            out = out.join(r, on="hospitalization_id", how="outer_coalesce")
        return out

    vitals_24hr = _compute_vitals_window(_vit_joined, 24, "24hrs_prior")
    vitals_6hr = _compute_vitals_window(_vit_joined, 6, "6hrs_prior")
    vitals_1hr = _compute_vitals_window(_vit_joined, 1, "1hr_prior")
    vitals_df = (
        vitals_24hr
        .join(vitals_6hr, on="hospitalization_id", how="outer_coalesce")
        .join(vitals_1hr, on="hospitalization_id", how="outer_coalesce")
    )

    print(f"Vitals computed for {vitals_df.height:,} hospitalizations")
    print(f"  Columns: {vitals_df.columns}")
    return (vitals_df,)


@app.cell
def _(DATA_DIR, FILETYPE, RespiratorySupport, TIMEZONE, cohort, pl):
    # Respiratory support: 24hr and 1hr prior to index_dttm
    resp_table = RespiratorySupport.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort["hospitalization_id"].to_list()},
    )
    _resp_pd = resp_table.df.copy()
    _resp_pd["recorded_dttm"] = _resp_pd["recorded_dttm"].dt.tz_localize(None)
    resp_pl = pl.from_pandas(_resp_pd)
    del _resp_pd

    resp_pl = resp_pl.with_columns(
        pl.col("device_category").str.to_lowercase().alias("device_category")
    )

    # Ordinal map for respiratory support level
    RESP_ORDINAL = {
        "room air": 1,
        "nasal cannula": 2,
        "face mask": 2,
        "high flow nc": 3,
        "nippv": 4,
        "cpap": 4,
    }

    resp_pl = resp_pl.with_columns(
        pl.col("device_category").replace_strict(RESP_ORDINAL, default=None).alias("resp_ordinal")
    )

    _resp_joined = (
        cohort.select(["hospitalization_id", "index_dttm"])
        .join(
            resp_pl.select(["hospitalization_id", "recorded_dttm", "device_category", "resp_ordinal"]),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            ((pl.col("index_dttm") - pl.col("recorded_dttm")).dt.total_seconds() / 3600)
            .alias("hours_before")
        )
        .filter(pl.col("hours_before") >= 0)
    )

    def _compute_resp_window(df, max_hours, suffix):
        w = df.filter(pl.col("hours_before") <= max_hours)
        # Highest respiratory support ordinal
        highest = w.group_by("hospitalization_id").agg(
            pl.col("resp_ordinal").max().alias(f"highest_resp_support_{suffix}")
        )
        # Binary flags for device types
        flags = cohort.select("hospitalization_id")
        for device, col_name in [
            ("nasal cannula", f"any_nc_{suffix}"),
            ("face mask", f"any_facemask_{suffix}"),
            ("high flow nc", f"any_hfno_{suffix}"),
            ("cpap", f"any_cpap_{suffix}"),
            ("nippv", f"any_nippv_{suffix}"),
        ]:
            _ids = w.filter(pl.col("device_category") == device)["hospitalization_id"].unique()
            flags = flags.with_columns(
                pl.col("hospitalization_id").is_in(_ids).cast(pl.Int8).alias(col_name)
            )
        return highest.join(flags, on="hospitalization_id", how="outer_coalesce")

    resp_24hr = _compute_resp_window(_resp_joined, 24, "24hrs_prior")
    resp_6hr = _compute_resp_window(_resp_joined, 6, "6hrs_prior")
    resp_1hr = _compute_resp_window(_resp_joined, 1, "1hr_prior")
    resp_df = (
        resp_24hr
        .join(resp_6hr, on="hospitalization_id", how="outer_coalesce")
        .join(resp_1hr, on="hospitalization_id", how="outer_coalesce")
    )

    print(f"Respiratory support computed for {resp_df.height:,} hospitalizations")
    print(f"  Columns: {resp_df.columns}")
    return (resp_df,)


@app.cell
def _(DATA_DIR, FILETYPE, MedicationAdminContinuous, TIMEZONE, cohort, pl):
    # Vasopressors continuous: 24hr and 1hr prior to index_dttm
    VASOPRESSORS = ["norepinephrine", "vasopressin", "epinephrine", "phenylephrine", "dopamine"]

    vaso_table = MedicationAdminContinuous.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "hospitalization_id": cohort["hospitalization_id"].to_list(),
            "med_category": VASOPRESSORS,
        },
    )
    _vaso_pd = vaso_table.df.copy()
    _vaso_pd["admin_dttm"] = _vaso_pd["admin_dttm"].dt.tz_localize(None)
    vaso_pl = pl.from_pandas(_vaso_pd)
    del _vaso_pd

    vaso_pl = vaso_pl.with_columns(
        pl.col("med_category").str.to_lowercase().alias("med_category")
    )

    _vaso_joined = (
        cohort.select(["hospitalization_id", "index_dttm"])
        .join(
            vaso_pl.select(["hospitalization_id", "admin_dttm", "med_category", "med_dose"]),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            ((pl.col("index_dttm") - pl.col("admin_dttm")).dt.total_seconds() / 3600)
            .alias("hours_before")
        )
        .filter(pl.col("hours_before") >= 0)
    )

    def _compute_vaso_window(df, max_hours, suffix):
        w = df.filter(pl.col("hours_before") <= max_hours)
        out = cohort.select("hospitalization_id")
        for med in VASOPRESSORS:
            _med_w = w.filter(pl.col("med_category") == med)
            _ids = _med_w["hospitalization_id"].unique()
            out = out.with_columns(
                pl.col("hospitalization_id").is_in(_ids).cast(pl.Int8).alias(f"any_vasopressor_{med}_{suffix}")
            )
            _max_dose = _med_w.group_by("hospitalization_id").agg(
                pl.col("med_dose").max().alias(f"{med}_dose_{suffix}")
            )
            out = out.join(_max_dose, on="hospitalization_id", how="left")
        out = out.with_columns(
            pl.max_horizontal([f"any_vasopressor_{med}_{suffix}" for med in VASOPRESSORS])
              .alias(f"any_vasopressor_{suffix}")
        )
        return out

    vaso_24hr = _compute_vaso_window(_vaso_joined, 24, "24hrs_prior")
    vaso_6hr = _compute_vaso_window(_vaso_joined, 6, "6hrs_prior")
    vaso_1hr = _compute_vaso_window(_vaso_joined, 1, "1hr_prior")
    vaso_df = (
        vaso_24hr
        .join(vaso_6hr, on="hospitalization_id", how="outer_coalesce")
        .join(vaso_1hr, on="hospitalization_id", how="outer_coalesce")
    )

    print(f"Vasopressors computed for {vaso_df.height:,} hospitalizations")
    print(f"  Columns: {vaso_df.columns}")
    return (vaso_df,)


@app.cell
def _(Adt, DATA_DIR, FILETYPE, PatientProcedures, TIMEZONE, cohort, pl):
    # Location, provider, hospital info
    adt_table = Adt.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort["hospitalization_id"].to_list()},
    )
    _adt_pd = adt_table.df.copy()
    _adt_pd["in_dttm"] = _adt_pd["in_dttm"].dt.tz_localize(None)
    _adt_pd["out_dttm"] = _adt_pd["out_dttm"].dt.tz_localize(None)
    adt_pl = pl.from_pandas(_adt_pd)
    del _adt_pd

    adt_pl = adt_pl.with_columns(
        pl.col("location_category").str.to_lowercase().alias("location_category"),
        pl.col("location_type").str.to_lowercase().alias("location_type"),
        pl.col("hospital_type").str.to_lowercase().alias("hospital_type"),
    )

    # Location at intubation (index_dttm)
    _loc_at_intub = (
        cohort.select(["hospitalization_id", "index_dttm"])
        .join(adt_pl, on="hospitalization_id", how="inner")
        .filter(
            (pl.col("index_dttm") >= pl.col("in_dttm"))
            & (pl.col("index_dttm") < pl.col("out_dttm"))
        )
        .select([
            "hospitalization_id",
            "hospital_id",
            pl.col("location_category").alias("location_at_intubation"),
            pl.when(pl.col("location_category") == "icu")
              .then(pl.col("location_type"))
              .otherwise(pl.lit(None))
              .alias("icu_type"),
            "hospital_type",
        ])
        .group_by("hospitalization_id").first()
    )

    # ICU LOS (sum of ICU time in days)
    _icu_stays = (
        adt_pl.filter(pl.col("location_category") == "icu")
        .with_columns(
            ((pl.col("out_dttm") - pl.col("in_dttm")).dt.total_seconds() / 86400)
            .alias("icu_days")
        )
        .group_by("hospitalization_id")
        .agg(pl.col("icu_days").sum().alias("los_icu"))
    )

    # Provider ID from CPT 31500
    procs_table = PatientProcedures.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": cohort["hospitalization_id"].to_list()},
    )
    _procs_pd = procs_table.df.copy()
    _procs_pd["procedure_billed_dttm"] = _procs_pd["procedure_billed_dttm"].dt.tz_localize(None)
    procs_pl = pl.from_pandas(_procs_pd)
    del _procs_pd

    _provider = (
        procs_pl.filter(
            (pl.col("procedure_code") == "31500")
            & (pl.col("procedure_code_format").str.to_uppercase() == "CPT")
        )
        .select(["hospitalization_id", pl.col("billing_provider_id").alias("provider_id")])
        .group_by("hospitalization_id").first()
    )

    location_df = (
        _loc_at_intub
        .join(_icu_stays, on="hospitalization_id", how="outer_coalesce")
        .join(_provider, on="hospitalization_id", how="outer_coalesce")
    )

    print(f"Location/provider info for {location_df.height:,} hospitalizations")
    return adt_pl, location_df


@app.cell
def _(adt_pl, cohort, pl):
    # Outcomes
    # LOS hospital (days)
    outcomes = cohort.select([
        "hospitalization_id",
        ((pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_seconds() / 86400)
        .alias("los_hospital"),
        pl.col("discharge_category").str.to_lowercase().alias("_dc_cat"),
    ])

    # Died in hospital
    outcomes = outcomes.with_columns(
        pl.col("_dc_cat").str.contains("(?i)expired|death").cast(pl.Int8).alias("died_hospital")
    )

    # Died in ICU: died AND last location was ICU
    _last_loc = (
        adt_pl.sort("out_dttm")
        .group_by("hospitalization_id")
        .last()
        .select([
            "hospitalization_id",
            pl.col("location_category").alias("last_location"),
        ])
    )

    outcomes = (
        outcomes.join(_last_loc, on="hospitalization_id", how="left")
        .with_columns(
            pl.when((pl.col("died_hospital") == 1) & (pl.col("last_location") == "icu"))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("died_icu")
        )
        .drop(["_dc_cat", "last_location"])
    )

    print(f"Outcomes computed for {outcomes.height:,} hospitalizations")
    print(f"  Died in hospital: {outcomes['died_hospital'].sum()}")
    print(f"  Died in ICU: {outcomes['died_icu'].sum()}")
    return (outcomes,)


@app.cell
def _(
    OUTPUT_DIR,
    ase_df,
    base,
    cci_df,
    crrt_df,
    location_df,
    mo,
    outcomes,
    pl,
    push_dose,
    resp_df,
    sofa_df,
    vaso_df,
    vitals_df,
):
    # Final join & save
    dataset = (
        base
        .join(push_dose, on="hospitalization_id", how="left")
        .join(cci_df, on="hospitalization_id", how="left")
        .join(sofa_df, on="hospitalization_id", how="left")
        .join(crrt_df, on="hospitalization_id", how="left")
        .join(ase_df, on="hospitalization_id", how="left")
        .join(vitals_df, on="hospitalization_id", how="left")
        .join(resp_df, on="hospitalization_id", how="left")
        .join(vaso_df, on="hospitalization_id", how="left")
        .join(location_df, on="hospitalization_id", how="left")
        .join(outcomes, on="hospitalization_id", how="left")
        .with_columns(pl.lit(1).cast(pl.Int8).alias("cpt_31500"))
    )

    dataset.write_parquet(OUTPUT_DIR / "rsi_analytical_dataset.parquet")

    print(f"Final analytical dataset: {dataset.height:,} rows, {len(dataset.columns)} columns")
    print(f"Saved to: {OUTPUT_DIR / 'rsi_analytical_dataset.parquet'}")
    print(f"\nColumns ({len(dataset.columns)}):")
    for col in dataset.columns:
        print(f"  {col}: {dataset[col].dtype}")

    mo.md(f"""
    ## Analytical Dataset Summary

    - **Rows:** {dataset.height:,}
    - **Columns:** {len(dataset.columns)}
    - **Saved to:** `{OUTPUT_DIR / 'rsi_analytical_dataset.parquet'}`
    """)
    return (dataset,)


@app.cell
def _(OUTPUT_DIR, SITE, dataset, mo, pl):
    # Analysis Dataset 2: focused variable set for modeling
    analysis_2 = dataset.select([
        "hospitalization_id",
        pl.col("med_dose_ind").alias("induction_dose_mg"),
        pl.col("induction_dose_per_kg").alias("dose_mg_kg"),
        pl.col("med_category_ind").alias("drug_received"),
        pl.col("age_at_admission").alias("age_years"),
        "sex_category",
        "race_category",
        "ethnicity_category",
        "cci",
        "ASE_7d_prior",
        "presumed_infection_7d_prior",
        pl.col("lowest_sbp_24hrs_prior").alias("worst_sbp_24hr_pre"),
        pl.col("highest_hr_24hrs_prior").alias("worst_hr_24hr_pre"),
        pl.col("lowest_spo2_24hrs_prior").alias("worst_spo2_24hr_pre"),
        "any_vasopressor_24hrs_prior",
        pl.col("lowest_sbp_6hrs_prior").alias("worst_sbp_6hr_pre"),
        pl.col("highest_hr_6hrs_prior").alias("worst_hr_6hr_pre"),
        pl.col("lowest_spo2_6hrs_prior").alias("worst_spo2_6hr_pre"),
        "any_vasopressor_6hrs_prior",
        "any_vasopressor_1hr_prior",
        "location_at_intubation",
        "icu_type",
        "hospital_type",
        "hospital_id",
        pl.col("index_dttm").dt.year().cast(pl.Int16).alias("calendar_year"),
        pl.col("index_dttm").dt.quarter().cast(pl.Int8).alias("calendar_quarter"),
        pl.lit(SITE).alias("site_id"),
        "provider_id",
        "weight_kg",
        "weight_source",
        "weight_recorded_dttm",
        "weight_to_rsi_hours",
        "post_rsi_weight_kg",
        "post_rsi_weight_recorded_dttm",
        "post_rsi_weight_to_rsi_hours",
    ])

    analysis_2.write_parquet(OUTPUT_DIR / "rsi_analysis_dataset_2.parquet")

    print(f"Analysis Dataset 2: {analysis_2.height:,} rows, {len(analysis_2.columns)} columns")
    print(f"Saved to: {OUTPUT_DIR / 'rsi_analysis_dataset_2.parquet'}")

    mo.md(f"""
    ## Analysis Dataset 2

    - **Rows:** {analysis_2.height:,}
    - **Columns:** {len(analysis_2.columns)}
    - **Saved to:** `{OUTPUT_DIR / 'rsi_analysis_dataset_2.parquet'}`
    """)
    return


if __name__ == "__main__":
    app.run()
