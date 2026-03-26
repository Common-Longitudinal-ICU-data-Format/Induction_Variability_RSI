import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import polars as pl
    import json
    from pathlib import Path
    from clifpy.tables import (
        Patient,
        Hospitalization,
        Adt,
        PatientProcedures,
        MedicationAdminIntermittent,
        RespiratorySupport,
        Vitals,
    )

    return (
        Adt,
        Hospitalization,
        MedicationAdminIntermittent,
        Path,
        Patient,
        PatientProcedures,
        RespiratorySupport,
        Vitals,
        json,
        mo,
        pl,
    )


@app.cell
def _(mo):
    mo.md("""
    # 01 Cohort Identification: RSI Induction Variability

    Identify patients who received Rapid Sequence Intubation (RSI) with
    etomidate or ketamine paired with rocuronium or succinylcholine.
    """)
    return


@app.cell
def _(Path, json):
    config_path = Path(__file__).parent.parent / "clif_config.json"
    with open(config_path, "r") as _f:
        config = json.load(_f)

    SITE = config["site_name"]
    DATA_DIR = config["data_directory"]
    FILETYPE = config["filetype"]
    TIMEZONE = config["timezone"]
    OUTPUT_DIR = Path(config["output_directory"])
    OUTPUT_TO_SHARE_DIR = Path(config["output_to_share_directory"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TO_SHARE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Site: {SITE}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output (PHI): {OUTPUT_DIR}")
    print(f"Output (to share): {OUTPUT_TO_SHARE_DIR}")
    return DATA_DIR, FILETYPE, OUTPUT_DIR, OUTPUT_TO_SHARE_DIR, SITE, TIMEZONE


@app.cell
def _(DATA_DIR, FILETYPE, Hospitalization, SITE, TIMEZONE, pl):
    # Load hospitalization table
    hosp_table = Hospitalization.from_file(
        data_directory=DATA_DIR, filetype=FILETYPE, timezone=TIMEZONE
    )
    _hosp_pd = hosp_table.df.copy()
    _hosp_pd["admission_dttm"] = _hosp_pd["admission_dttm"].dt.tz_localize(None)
    _hosp_pd["discharge_dttm"] = _hosp_pd["discharge_dttm"].dt.tz_localize(None)
    hosp_pl = pl.from_pandas(_hosp_pd)
    del _hosp_pd

    n_total = hosp_pl.height
    print(f"Total hospitalizations: {n_total:,}")

    # Inclusion 1: Age >= 18
    hosp_pl = hosp_pl.filter(pl.col("age_at_admission") >= 18)
    n_after_age = hosp_pl.height
    n_excl_age = n_total - n_after_age
    print(f"After age >= 18: {n_after_age:,} (excluded {n_excl_age:,})")

    # Inclusion 2: Admission and discharge date 2018-01-01 through 2025-12-31
    if SITE == "mimic":
        # MIMIC has different date range, skip date filter
        n_after_date = hosp_pl.height
        n_excl_date = 0
        print(f"Date filter SKIPPED for MIMIC site: {n_after_date:,}")
    else:
        hosp_pl = hosp_pl.filter(
            (pl.col("admission_dttm") >= pl.datetime(2018, 1, 1))
            & (pl.col("admission_dttm") < pl.datetime(2026, 1, 1))
            & (pl.col("discharge_dttm") >= pl.datetime(2018, 1, 1))
            & (pl.col("discharge_dttm") < pl.datetime(2026, 1, 1))
        )
        n_after_date = hosp_pl.height
        n_excl_date = n_after_age - n_after_date
        print(f"After date filter (2018-2025): {n_after_date:,} (excluded {n_excl_date:,})")

    hosp_pl.head()
    return hosp_pl, n_after_date, n_excl_age, n_excl_date, n_total


@app.cell
def _(DATA_DIR, FILETYPE, PatientProcedures, SITE, TIMEZONE, hosp_pl, pl):
    # Inclusion 3: CPT 31500 with non-missing billing_provider_id
    procs_table = PatientProcedures.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": hosp_pl["hospitalization_id"].to_list()},
    )
    _procs_pd = procs_table.df.copy()
    _procs_pd["procedure_billed_dttm"] = _procs_pd["procedure_billed_dttm"].dt.tz_localize(None)
    procs_pl = pl.from_pandas(_procs_pd)
    del _procs_pd

    _cpt_base = procs_pl.filter(
        (pl.col("procedure_code") == "31500")
        & (pl.col("procedure_code_format").str.to_uppercase() == "CPT")
    )
    if SITE == "mimic":
        # MIMIC lacks billing_provider_id; impute with -9999
        cpt_31500 = _cpt_base.with_columns(
            pl.col("billing_provider_id").fill_null("-9999").alias("billing_provider_id")
        )
    else:
        cpt_31500 = _cpt_base.filter(
            pl.col("billing_provider_id").is_not_null()
            & (pl.col("billing_provider_id") != "")
        )

    # Count hospitalizations with ANY CPT 31500 (regardless of provider)
    _hosp_with_any_cpt = hosp_pl.filter(
        pl.col("hospitalization_id").is_in(_cpt_base["hospitalization_id"].unique())
    )
    n_no_cpt = hosp_pl.height - _hosp_with_any_cpt.height
    n_cpt_no_provider = _hosp_with_any_cpt.height - hosp_pl.filter(
        pl.col("hospitalization_id").is_in(cpt_31500["hospitalization_id"].unique())
    ).height

    hosp_with_cpt = hosp_pl.filter(
        pl.col("hospitalization_id").is_in(cpt_31500["hospitalization_id"].unique())
    )
    n_after_cpt = hosp_with_cpt.height
    n_excl_cpt = hosp_pl.height - n_after_cpt
    print(f"After CPT 31500 filter: {n_after_cpt:,} (excluded {n_excl_cpt:,})")
    print(f"  - No CPT 31500: {n_no_cpt:,}")
    print(f"  - CPT 31500 but no billing provider: {n_cpt_no_provider:,}")
    return (
        hosp_with_cpt,
        n_after_cpt,
        n_cpt_no_provider,
        n_excl_cpt,
        n_no_cpt,
        procs_pl,
    )


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    MedicationAdminIntermittent,
    TIMEZONE,
    hosp_with_cpt,
    pl,
):
    # Inclusion 4: RSI medication pairing within 5 minutes
    INDUCTION_AGENTS = ["etomidate", "ketamine"]
    PARALYTICS = ["rocuronium", "succinylcholine"]
    RSI_MEDS = INDUCTION_AGENTS + PARALYTICS

    meds_table = MedicationAdminIntermittent.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": hosp_with_cpt["hospitalization_id"].to_list()},
    )
    _meds_pd = meds_table.df.copy()
    _meds_pd["admin_dttm"] = _meds_pd["admin_dttm"].dt.tz_localize(None)
    meds_pl = pl.from_pandas(_meds_pd)
    del _meds_pd

    # Lowercase med_category and mar_action_category for matching
    meds_pl = meds_pl.with_columns([
        pl.col("med_category").str.to_lowercase().alias("med_category"),
        pl.col("mar_action_category").str.to_lowercase().alias("mar_action_category"),
    ])

    # Keep only administered RSI medications
    rsi_meds = meds_pl.filter(
        pl.col("med_category").is_in(RSI_MEDS)
        & (pl.col("mar_action_category") == "given")
    )

    induction = rsi_meds.filter(pl.col("med_category").is_in(INDUCTION_AGENTS))
    paralytic = rsi_meds.filter(pl.col("med_category").is_in(PARALYTICS))

    # Join induction + paralytic within each hospitalization
    pairs = induction.select([
        "hospitalization_id",
        pl.col("admin_dttm").alias("admin_dttm_ind"),
        pl.col("med_category").alias("med_category_ind"),
        pl.col("med_dose").alias("med_dose_ind"),
        pl.col("med_dose_unit").alias("med_dose_unit_ind"),
    ]).join(
        paralytic.select([
            "hospitalization_id",
            pl.col("admin_dttm").alias("admin_dttm_par"),
            pl.col("med_category").alias("med_category_par"),
            pl.col("med_dose").alias("med_dose_par"),
            pl.col("med_dose_unit").alias("med_dose_unit_par"),
        ]),
        on="hospitalization_id",
        how="inner",
    )

    # Compute time difference and keep all pairs within 30 minutes (for sub-analysis)
    pairs = pairs.with_columns(
        ((pl.col("admin_dttm_par") - pl.col("admin_dttm_ind")).dt.total_seconds().abs() / 60)
        .alias("time_diff_min")
    )
    pairs_30min = pairs.filter(pl.col("time_diff_min") <= 30)

    # Dedup to first pair per hospitalization for sub-analysis
    pairs_30min = pairs_30min.with_columns(
        pl.min_horizontal("admin_dttm_ind", "admin_dttm_par").alias("index_dttm")
    ).sort("index_dttm").group_by("hospitalization_id").first()

    # Filter to pairs within 5 minutes for cohort definition
    pairs_5min = pairs_30min.filter(pl.col("time_diff_min") <= 5)

    # Index time = earliest of the pair
    pairs_5min = pairs_5min.with_columns(
        pl.min_horizontal("admin_dttm_ind", "admin_dttm_par").alias("index_dttm")
    )

    # Keep earliest RSI event per hospitalization
    rsi_events = (
        pairs_5min.sort("index_dttm")
        .group_by("hospitalization_id")
        .first()
    )

    n_after_rsi = rsi_events.height
    n_excl_rsi = hosp_with_cpt.height - n_after_rsi
    print(f"After RSI pairing: {n_after_rsi:,} (excluded {n_excl_rsi:,})")
    return meds_pl, n_after_rsi, n_excl_rsi, pairs_30min, rsi_events


@app.cell
def _(OUTPUT_DIR, OUTPUT_TO_SHARE_DIR, mo, pairs_30min, pl):
    # Sub-analysis: RSI induction-paralytic timing distribution (all pairs within 30 min)
    _td = pairs_30min["time_diff_min"]

    _overall = pl.DataFrame({
        "group": ["Overall"],
        "count": [pairs_30min.height],
        "median": [_td.median()],
        "mean": [_td.mean()],
        "p25": [_td.quantile(0.25)],
        "p75": [_td.quantile(0.75)],
        "min": [_td.min()],
        "max": [_td.max()],
    })

    _by_combo = (
        pairs_30min.group_by(["med_category_ind", "med_category_par"])
        .agg([
            pl.col("time_diff_min").count().alias("count"),
            pl.col("time_diff_min").median().alias("median"),
            pl.col("time_diff_min").mean().alias("mean"),
            pl.col("time_diff_min").quantile(0.25).alias("p25"),
            pl.col("time_diff_min").quantile(0.75).alias("p75"),
            pl.col("time_diff_min").min().alias("min"),
            pl.col("time_diff_min").max().alias("max"),
        ])
        .with_columns(
            (pl.col("med_category_ind") + " + " + pl.col("med_category_par")).alias("group")
        )
        .select(["group", "count", "median", "mean", "p25", "p75", "min", "max"])
        .sort("count", descending=True)
    )

    timing_stats = pl.concat([_overall, _by_combo], how="diagonal_relaxed")

    # Save pair-level data (PHI) as parquet
    pairs_30min.write_parquet(OUTPUT_DIR / "subanalysis_rsi_timing_pairs.parquet")

    # Save aggregate stats (no PHI) as CSV
    timing_stats.write_csv(OUTPUT_TO_SHARE_DIR / "subanalysis_rsi_timing_stats.csv")

    print(f"Timing sub-analysis: {pairs_30min.height:,} pairs within 30 min")
    print(f"  Saved pair-level data: {OUTPUT_DIR / 'subanalysis_rsi_timing_pairs.parquet'}")
    print(f"  Saved aggregate stats: {OUTPUT_TO_SHARE_DIR / 'subanalysis_rsi_timing_stats.csv'}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        pairs_30min["time_diff_min"].to_list(),
        bins=60,
        range=(0, 30),
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(x=5, color="red", linestyle="--", linewidth=1.5, label="Cohort threshold (5 min)")
    ax.set_xlabel("Time difference (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("RSI Induction-Paralytic Timing Distribution (first pair per hospitalization, ≤30 min)")
    ax.legend()
    plt.tight_layout()

    mo.as_html(fig)
    return


@app.cell
def _(Adt, DATA_DIR, FILETYPE, TIMEZONE, pl, rsi_events):
    # Inclusion 5: Location = ED or ICU at time of medication administration
    adt_table = Adt.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": rsi_events["hospitalization_id"].to_list()},
    )
    _adt_pd = adt_table.df.copy()
    _adt_pd["in_dttm"] = _adt_pd["in_dttm"].dt.tz_localize(None)
    _adt_pd["out_dttm"] = _adt_pd["out_dttm"].dt.tz_localize(None)
    adt_pl = pl.from_pandas(_adt_pd)
    del _adt_pd

    adt_pl = adt_pl.with_columns(
        pl.col("location_category").str.to_lowercase().alias("location_category")
    )

    # Find location at index_dttm
    rsi_loc = (
        rsi_events.select(["hospitalization_id", "index_dttm"])
        .join(adt_pl, on="hospitalization_id", how="inner")
        .filter(
            (pl.col("index_dttm") >= pl.col("in_dttm"))
            & (pl.col("index_dttm") < pl.col("out_dttm"))
        )
        .filter(pl.col("location_category").is_in(["ed", "icu"]))
    )

    valid_loc_ids = rsi_loc["hospitalization_id"].unique()
    rsi_events_loc = rsi_events.filter(
        pl.col("hospitalization_id").is_in(valid_loc_ids)
    )

    n_after_loc = rsi_events_loc.height
    n_excl_loc = rsi_events.height - n_after_loc
    print(f"After ED/ICU location: {n_after_loc:,} (excluded {n_excl_loc:,})")
    return n_after_loc, n_excl_loc, rsi_events_loc, rsi_loc


@app.cell
def _(DATA_DIR, FILETYPE, RespiratorySupport, TIMEZONE, pl, rsi_events_loc):
    # Inclusion 6: IMV initiated within 6 hours of index induction medication
    resp_table = RespiratorySupport.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={"hospitalization_id": rsi_events_loc["hospitalization_id"].to_list()},
    )
    _resp_pd = resp_table.df.copy()
    _resp_pd["recorded_dttm"] = _resp_pd["recorded_dttm"].dt.tz_localize(None)
    resp_pl = pl.from_pandas(_resp_pd)
    del _resp_pd

    resp_pl = resp_pl.with_columns(
        pl.col("device_category").str.to_lowercase().alias("device_category")
    )

    imv = resp_pl.filter(pl.col("device_category") == "imv")

    rsi_imv = (
        rsi_events_loc.select(["hospitalization_id", "index_dttm"])
        .join(
            imv.select(["hospitalization_id", "recorded_dttm"]),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            ((pl.col("recorded_dttm") - pl.col("index_dttm")).dt.total_seconds() / 3600)
            .alias("hours_to_imv")
        )
        .filter((pl.col("hours_to_imv") >= 0) & (pl.col("hours_to_imv") <= 6))
    )

    valid_imv_ids = rsi_imv["hospitalization_id"].unique()

    # Get first IMV time per hospitalization
    first_imv = (
        rsi_imv.sort("recorded_dttm")
        .group_by("hospitalization_id")
        .first()
        .select(["hospitalization_id", pl.col("recorded_dttm").alias("imv_dttm")])
    )

    cohort_incl = (
        rsi_events_loc
        .filter(pl.col("hospitalization_id").is_in(valid_imv_ids))
        .join(first_imv, on="hospitalization_id", how="left")
    )

    n_after_imv = cohort_incl.height
    n_excl_imv = rsi_events_loc.height - n_after_imv
    print(f"After IMV within 6h: {n_after_imv:,} (excluded {n_excl_imv:,})")
    print(f"\nAll inclusion criteria applied: {n_after_imv:,} hospitalizations")
    return cohort_incl, n_after_imv, n_excl_imv, resp_pl


@app.cell
def _(mo):
    mo.md("""
    ## Exclusion Criteria
    """)
    return


@app.cell
def _(cohort_incl, hosp_pl, pl, procs_pl, resp_pl):
    # Exclusion 2: Tracheostomy within 24 hours of admission
    _hosp_adm = hosp_pl.select(["hospitalization_id", "admission_dttm"])
    _cohort_ids = cohort_incl["hospitalization_id"]

    # Check respiratory_support tracheostomy flag
    _trach_resp = (
        resp_pl.filter(
            (pl.col("tracheostomy") == 1)
            & pl.col("hospitalization_id").is_in(_cohort_ids)
        )
        .join(_hosp_adm, on="hospitalization_id", how="inner")
        .with_columns(
            ((pl.col("recorded_dttm") - pl.col("admission_dttm")).dt.total_seconds() / 3600)
            .alias("hours_from_adm")
        )
        .filter((pl.col("hours_from_adm") >= 0) & (pl.col("hours_from_adm") <= 24))
    )
    _excl_trach_resp = _trach_resp["hospitalization_id"].unique()

    # Check patient_procedures for tracheostomy CPT codes
    TRACH_CPTS = ["31600", "31601", "31603", "31605", "31610"]
    _trach_proc = (
        procs_pl.filter(
            pl.col("procedure_code").is_in(TRACH_CPTS)
            & (pl.col("procedure_code_format").str.to_uppercase() == "CPT")
            & pl.col("hospitalization_id").is_in(_cohort_ids)
        )
        .join(_hosp_adm, on="hospitalization_id", how="inner")
        .with_columns(
            ((pl.col("procedure_billed_dttm") - pl.col("admission_dttm")).dt.total_seconds() / 3600)
            .alias("hours_from_adm")
        )
        .filter((pl.col("hours_from_adm") >= 0) & (pl.col("hours_from_adm") <= 24))
    )
    _excl_trach_proc = _trach_proc["hospitalization_id"].unique()

    # Combine
    _all_trach_ids = pl.concat([_excl_trach_resp, _excl_trach_proc]).unique()
    cohort_e2 = cohort_incl.filter(~pl.col("hospitalization_id").is_in(_all_trach_ids))
    n_excl_e2 = cohort_incl.height - cohort_e2.height
    n_after_e2 = cohort_e2.height
    print(f"Excl 2 - Tracheostomy within 24h: excluded {n_excl_e2:,} (resp flag: {_excl_trach_resp.len()}, proc codes: {_excl_trach_proc.len()}), remaining {n_after_e2:,}")
    return cohort_e2, n_after_e2, n_excl_e2


@app.cell
def _(cohort_e2, meds_pl, pl):
    # Exclusion 3: Received both etomidate AND ketamine during hospitalization
    _ind_meds = meds_pl.filter(
        pl.col("med_category").is_in(["etomidate", "ketamine"])
        & (pl.col("mar_action_category") == "given")
        & pl.col("hospitalization_id").is_in(cohort_e2["hospitalization_id"])
    )
    _both_agents = (
        _ind_meds.group_by("hospitalization_id")
        .agg(pl.col("med_category").n_unique().alias("n_agents"))
        .filter(pl.col("n_agents") > 1)
    )
    excl_both_ids = _both_agents["hospitalization_id"]

    cohort_e3 = cohort_e2.filter(~pl.col("hospitalization_id").is_in(excl_both_ids))
    n_excl_e3 = cohort_e2.height - cohort_e3.height
    n_after_e3 = cohort_e3.height
    print(f"Excl 3 - Both etomidate AND ketamine: excluded {n_excl_e3:,}, remaining {n_after_e3:,}")
    return cohort_e3, n_after_e3, n_excl_e3


@app.cell
def _(cohort_e3, meds_pl, pl):
    # Exclusion 4: Benzodiazepine or propofol within 60 min prior to RSI meds
    SEDATIVES = ["lorazepam", "diazepam", "midazolam", "propofol"]

    _sedatives = meds_pl.filter(
        pl.col("med_category").is_in(SEDATIVES)
        & (pl.col("mar_action_category") == "given")
        & pl.col("hospitalization_id").is_in(cohort_e3["hospitalization_id"])
    )

    _check = (
        cohort_e3.select(["hospitalization_id", "index_dttm"])
        .join(
            _sedatives.select(["hospitalization_id", "admin_dttm"]),
            on="hospitalization_id",
            how="inner",
        )
        .with_columns(
            ((pl.col("index_dttm") - pl.col("admin_dttm")).dt.total_seconds() / 60)
            .alias("min_before")
        )
        .filter((pl.col("min_before") > 0) & (pl.col("min_before") <= 60))
    )
    excl_sed_ids = _check["hospitalization_id"].unique()

    cohort_e4 = cohort_e3.filter(~pl.col("hospitalization_id").is_in(excl_sed_ids))
    n_excl_e4 = cohort_e3.height - cohort_e4.height
    n_after_e4 = cohort_e4.height
    print(f"Excl 4 - Benzo/propofol within 60 min prior: excluded {n_excl_e4:,}, remaining {n_after_e4:,}")
    return cohort_e4, n_after_e4, n_excl_e4


@app.cell
def _(DATA_DIR, FILETYPE, TIMEZONE, Vitals, cohort_e4, hosp_pl, pl):
    # Weight lookup: current hospitalization first, then 28-day lookback across prior hospitalizations
    vitals_table = Vitals.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "hospitalization_id": cohort_e4["hospitalization_id"].to_list(),
            "vital_category": ["weight_kg"],
        },
    )
    _vit_pd = vitals_table.df.copy()
    _vit_pd["recorded_dttm"] = _vit_pd["recorded_dttm"].dt.tz_localize(None)
    weights_pl = pl.from_pandas(_vit_pd)
    del _vit_pd

    weights_pl = weights_pl.with_columns(
        pl.col("vital_category").str.to_lowercase().alias("vital_category")
    )

    # Step 1: Find closest weight prior to index_dttm in current hospitalization
    _wt_current = (
        cohort_e4.select(["hospitalization_id", "index_dttm"])
        .join(
            weights_pl.select(["hospitalization_id", "recorded_dttm", "vital_value"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("recorded_dttm") <= pl.col("index_dttm"))
        .with_columns(
            (pl.col("index_dttm") - pl.col("recorded_dttm")).dt.total_seconds().alias("secs_before")
        )
        .sort("secs_before")
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("vital_value").alias("weight_kg"),
            pl.col("recorded_dttm").alias("weight_recorded_dttm"),
            pl.lit("current_hospitalization").alias("weight_source"),
        ])
    )
    _has_current = _wt_current["hospitalization_id"]

    # Step 2: For patients without current-hospitalization weight, look back 28 days
    _no_current = cohort_e4.filter(~pl.col("hospitalization_id").is_in(_has_current))
    _no_current_with_patient = _no_current.select(["hospitalization_id", "index_dttm"]).join(
        hosp_pl.select(["hospitalization_id", "patient_id", "admission_dttm"]),
        on="hospitalization_id",
        how="left",
    )

    _all_hosp = hosp_pl.select(["hospitalization_id", "patient_id", "admission_dttm"]).rename(
        {"hospitalization_id": "prior_hosp_id", "admission_dttm": "prior_adm_dttm"}
    )
    _prior = (
        _no_current_with_patient.select(["hospitalization_id", "patient_id", "admission_dttm", "index_dttm"])
        .join(_all_hosp, on="patient_id", how="inner")
        .filter(pl.col("prior_adm_dttm") < pl.col("admission_dttm"))
    )
    _prior_hosp_ids = _prior["prior_hosp_id"].unique().to_list()

    if _prior_hosp_ids:
        _prior_vit = Vitals.from_file(
            data_directory=DATA_DIR,
            filetype=FILETYPE,
            timezone=TIMEZONE,
            filters={
                "hospitalization_id": _prior_hosp_ids,
                "vital_category": ["weight_kg"],
            },
        )
        _prior_vit_pd = _prior_vit.df.copy()
        _prior_vit_pd["recorded_dttm"] = _prior_vit_pd["recorded_dttm"].dt.tz_localize(None)
        _prior_weights = pl.from_pandas(_prior_vit_pd).select(
            ["hospitalization_id", "recorded_dttm", "vital_value"]
        )
        del _prior_vit_pd

        # Map prior hospitalization weights back to current hospitalization
        _prior_mapped = (
            _prior_weights.rename({"hospitalization_id": "prior_hosp_id"})
            .join(
                _prior.select(["hospitalization_id", "prior_hosp_id", "index_dttm"]).unique(),
                on="prior_hosp_id",
                how="inner",
            )
        )

        # Find latest weight from prior hospitalizations (before current RSI)
        _wt_prior = (
            _prior_mapped
            .filter(pl.col("recorded_dttm") <= pl.col("index_dttm"))
            .with_columns(
                (pl.col("index_dttm") - pl.col("recorded_dttm")).dt.total_seconds().alias("secs_before")
            )
            .sort("secs_before")
            .group_by("hospitalization_id")
            .first()
            .select([
                "hospitalization_id",
                pl.col("vital_value").alias("weight_kg"),
                pl.col("recorded_dttm").alias("weight_recorded_dttm"),
                pl.lit("previous_hospitalization").alias("weight_source"),
            ])
        )
    else:
        _wt_prior = pl.DataFrame(
            schema={
                "hospitalization_id": pl.Utf8,
                "weight_kg": pl.Float64,
                "weight_recorded_dttm": pl.Datetime,
                "weight_source": pl.Utf8,
            }
        )

    # Step 3: Combine all weight sources
    _wt_all = pl.concat([_wt_current, _wt_prior], how="diagonal_relaxed")
    _has_any_weight = _wt_all["hospitalization_id"]

    # Step 4: Patients with no weight at all
    _no_weight = (
        cohort_e4.filter(~pl.col("hospitalization_id").is_in(_has_any_weight))
        .select(["hospitalization_id"])
        .with_columns([
            pl.lit(None, dtype=pl.Float64).alias("weight_kg"),
            pl.lit(None, dtype=pl.Datetime).alias("weight_recorded_dttm"),
            pl.lit("no_weight").alias("weight_source"),
        ])
    )

    _wt_final = pl.concat([_wt_all, _no_weight], how="diagonal_relaxed")

    # Add weight_to_rsi_hours
    cohort_e5 = (
        cohort_e4.join(_wt_final, on="hospitalization_id", how="left")
        .with_columns(
            pl.when(pl.col("weight_recorded_dttm").is_not_null())
            .then((pl.col("index_dttm") - pl.col("weight_recorded_dttm")).dt.total_seconds() / 3600)
            .otherwise(None)
            .alias("weight_to_rsi_hours")
        )
    )

    # Post-RSI weight: first weight after index_dttm in current hospitalization
    _post_rsi = (
        cohort_e5.select(["hospitalization_id", "index_dttm"])
        .join(
            weights_pl.select(["hospitalization_id", "recorded_dttm", "vital_value"]),
            on="hospitalization_id",
            how="inner",
        )
        .filter(pl.col("recorded_dttm") > pl.col("index_dttm"))
        .with_columns(
            (pl.col("recorded_dttm") - pl.col("index_dttm")).dt.total_seconds().alias("secs_after")
        )
        .sort("secs_after")
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("vital_value").alias("post_rsi_weight_kg"),
            pl.col("recorded_dttm").alias("post_rsi_weight_recorded_dttm"),
            (pl.col("secs_after") / 3600).alias("post_rsi_weight_to_rsi_hours"),
        ])
    )

    cohort_e5 = cohort_e5.join(_post_rsi, on="hospitalization_id", how="left")

    _n_current = _wt_current.height
    _n_prior = _wt_prior.height
    _n_none = _no_weight.height
    _n_post = _post_rsi.height
    print(f"Weight sources: current_hospitalization={_n_current:,}, previous_hospitalization={_n_prior:,}, no_weight={_n_none:,}")
    print(f"Post-RSI weight available: {_n_post:,} / {cohort_e5.height:,}")
    return cohort_e5, weights_pl


@app.cell
def _(cohort_e5):
    # Exclusion 6: Prior intubation during same hospitalization (keep first only)
    cohort_e6 = (
        cohort_e5.sort("index_dttm")
        .group_by("hospitalization_id")
        .first()
    )
    n_excl_e6 = cohort_e5.height - cohort_e6.height
    n_after_e6 = cohort_e6.height
    print(f"Excl 6 - Keep first RSI per hospitalization: removed {n_excl_e6:,} duplicates, remaining {n_after_e6:,}")
    return cohort_e6, n_after_e6, n_excl_e6


@app.cell
def _(cohort_e6, pl):
    # Exclusion 7 & 8: Non-feasible medication doses
    _etomidate_bad = (
        (pl.col("med_category_ind") == "etomidate")
        & ((pl.col("med_dose_ind") < 2) | (pl.col("med_dose_ind") > 100))
    )
    _ketamine_bad = (
        (pl.col("med_category_ind") == "ketamine")
        & ((pl.col("med_dose_ind") < 15) | (pl.col("med_dose_ind") > 300))
    )

    n_bad_etom = cohort_e6.filter(_etomidate_bad).height
    n_bad_ket = cohort_e6.filter(_ketamine_bad).height

    cohort_e7 = cohort_e6.filter(~(_etomidate_bad | _ketamine_bad))
    n_excl_e7 = cohort_e6.height - cohort_e7.height
    n_after_e7 = cohort_e7.height
    print(f"Excl 7&8 - Non-feasible doses: excluded {n_excl_e7:,} (etomidate: {n_bad_etom}, ketamine: {n_bad_ket}), remaining {n_after_e7:,}")
    return cohort_e7, n_after_e7, n_excl_e7


@app.cell
def _(cohort_e7, pl):
    # Exclusion 9: Non-physiological weight (<20 kg or >300 kg) — null weights pass through
    cohort_final = cohort_e7.filter(
        pl.col("weight_kg").is_null()
        | ((pl.col("weight_kg") >= 20) & (pl.col("weight_kg") <= 300))
    )
    n_excl_e8 = cohort_e7.height - cohort_final.height
    n_after_e8 = cohort_final.height
    print(f"Excl 9 - Non-physiological weight: excluded {n_excl_e8:,}, remaining {n_after_e8:,}")
    print(f"\nFinal cohort: {n_after_e8:,} hospitalizations")
    return cohort_final, n_after_e8, n_excl_e8


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    Patient,
    TIMEZONE,
    cohort_final,
    hosp_pl,
    pl,
    rsi_loc,
):
    # Build final cohort dataset with demographics
    patient_table = Patient.from_file(
        data_directory=DATA_DIR,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        filters={
            "patient_id": hosp_pl.filter(
                pl.col("hospitalization_id").is_in(cohort_final["hospitalization_id"])
            )["patient_id"].unique().to_list()
        },
    )
    patient_pl = pl.from_pandas(patient_table.df)

    _demo = patient_pl.select([
        "patient_id",
        pl.col("sex_category").str.to_lowercase(),
        pl.col("race_category").str.to_lowercase(),
        pl.col("ethnicity_category").str.to_lowercase(),
    ])

    _hosp_cols = hosp_pl.select([
        "patient_id",
        "hospitalization_id",
        "admission_dttm",
        "discharge_dttm",
        "age_at_admission",
        "admission_type_category",
        "discharge_category",
    ])

    _cohort_cols = cohort_final.select([
        "hospitalization_id",
        "index_dttm",
        "imv_dttm",
        "med_category_ind",
        "med_dose_ind",
        "med_dose_unit_ind",
        "med_category_par",
        "admin_dttm_par",
        "med_dose_par",
        "med_dose_unit_par",
        "weight_kg",
        "weight_source",
        "weight_recorded_dttm",
        "weight_to_rsi_hours",
        "post_rsi_weight_kg",
        "post_rsi_weight_recorded_dttm",
        "post_rsi_weight_to_rsi_hours",
    ])

    _loc_lookup = (
        rsi_loc
        .select(["hospitalization_id", "location_category"])
        .group_by("hospitalization_id")
        .first()
    )

    cohort_out = (
        _cohort_cols
        .join(_hosp_cols, on="hospitalization_id", how="left")
        .join(_demo, on="patient_id", how="left")
        .join(_loc_lookup, on="hospitalization_id", how="left")
        .with_columns(
            (pl.col("med_dose_ind") / pl.col("weight_kg")).alias("induction_dose_per_kg")
        )
    )

    print(f"Final cohort: {cohort_out.height:,} RSI events, {cohort_out['patient_id'].n_unique():,} unique patients")
    print(f"Columns: {cohort_out.columns}")
    return (cohort_out,)


@app.cell
def _(
    OUTPUT_DIR,
    OUTPUT_TO_SHARE_DIR,
    SITE,
    cohort_out,
    json,
    n_after_cpt,
    n_after_date,
    n_after_e2,
    n_after_e3,
    n_after_e4,
    n_after_e6,
    n_after_e7,
    n_after_e8,
    n_after_imv,
    n_after_loc,
    n_after_rsi,
    n_cpt_no_provider,
    n_excl_age,
    n_excl_cpt,
    n_excl_date,
    n_excl_e2,
    n_excl_e3,
    n_excl_e4,
    n_excl_e6,
    n_excl_e7,
    n_excl_e8,
    n_excl_imv,
    n_excl_loc,
    n_excl_rsi,
    n_no_cpt,
    n_total,
    pl,
):
    # Build CONSORT-style flow and save outputs
    consort_flow = {
        "site": SITE,
        "steps": [
            {"step": 0, "description": "Total hospitalizations", "n_remaining": n_total, "n_excluded": 0, "exclusion_reason": None},
            {"step": 1, "description": "Age >= 18", "n_remaining": n_total - n_excl_age, "n_excluded": n_excl_age, "exclusion_reason": "Age < 18"},
            {"step": 2, "description": "Admission & discharge 2018-01-01 to 2025-12-31", "n_remaining": n_after_date, "n_excluded": n_excl_date, "exclusion_reason": "Admission or discharge outside study date range"},
            {"step": 3, "description": "CPT 31500 with billing provider", "n_remaining": n_after_cpt, "n_excluded": n_excl_cpt, "exclusion_reason": "No CPT 31500 or missing billing_provider_id", "n_no_cpt": n_no_cpt, "n_cpt_no_provider": n_cpt_no_provider},
            {"step": 4, "description": "RSI pairing (induction + paralytic within 5 min)", "n_remaining": n_after_rsi, "n_excluded": n_excl_rsi, "exclusion_reason": "No valid RSI medication pair"},
            {"step": 5, "description": "ED or ICU location at RSI time", "n_remaining": n_after_loc, "n_excluded": n_excl_loc, "exclusion_reason": "Not in ED or ICU at RSI time"},
            {"step": 6, "description": "IMV within 6 hours of induction", "n_remaining": n_after_imv, "n_excluded": n_excl_imv, "exclusion_reason": "No IMV within 6 hours"},
            {"step": 7, "description": "No tracheostomy within 24h of admission", "n_remaining": n_after_e2, "n_excluded": n_excl_e2, "exclusion_reason": "Tracheostomy within 24h of admission"},
            {"step": 8, "description": "Not both etomidate and ketamine", "n_remaining": n_after_e3, "n_excluded": n_excl_e3, "exclusion_reason": "Received both etomidate and ketamine"},
            {"step": 9, "description": "No benzo/propofol within 60 min prior", "n_remaining": n_after_e4, "n_excluded": n_excl_e4, "exclusion_reason": "Benzo or propofol within 60 min prior to RSI"},
            {"step": 10, "description": "First RSI per hospitalization", "n_remaining": n_after_e6, "n_excluded": n_excl_e6, "exclusion_reason": "Prior intubation in same hospitalization"},
            {"step": 11, "description": "Feasible induction dose", "n_remaining": n_after_e7, "n_excluded": n_excl_e7, "exclusion_reason": "Non-feasible etomidate or ketamine dose"},
            {"step": 12, "description": "Physiological weight (20-300 kg)", "n_remaining": n_after_e8, "n_excluded": n_excl_e8, "exclusion_reason": "Non-physiological weight"},
        ],
        "final_cohort": {
            "n_hospitalizations": cohort_out.height,
            "n_patients": cohort_out["patient_id"].n_unique(),
        },
    }

    # Save cohort parquet (PHI) to output_directory
    cohort_out.write_parquet(OUTPUT_DIR / "rsi_cohort.parquet")
    print(f"Cohort saved to: {OUTPUT_DIR / 'rsi_cohort.parquet'}")

    # Save CONSORT flow JSON (shareable) to output_to_share_directory
    consort_json_path = OUTPUT_TO_SHARE_DIR / "consort_cohort.json"
    with open(consort_json_path, "w") as _f:
        json.dump(consort_flow, _f, indent=2)
    print(f"CONSORT JSON saved to: {consort_json_path}")

    # Save CONSORT flow CSV (shareable)
    consort_csv = pl.DataFrame(consort_flow["steps"])
    consort_csv.write_csv(OUTPUT_TO_SHARE_DIR / "consort_cohort.csv")
    print(f"CONSORT CSV saved to: {OUTPUT_TO_SHARE_DIR / 'consort_cohort.csv'}")
    return (consort_flow,)


@app.cell
def _(cohort_out, consort_flow, mo, pl):
    consort_df = pl.DataFrame(consort_flow["steps"]).select([
        "step", "description", "n_remaining", "n_excluded", "exclusion_reason",
    ])

    _ind_counts = cohort_out.group_by("med_category_ind").len().sort("len", descending=True)
    _par_counts = cohort_out.group_by("med_category_par").len().sort("len", descending=True)

    mo.vstack([
        mo.md("## Cohort Flow Summary"),
        mo.ui.table(consort_df),
        mo.md(f"**Final cohort: {cohort_out.height:,} hospitalizations, {cohort_out['patient_id'].n_unique():,} patients**"),
        mo.md("### Induction agent distribution"),
        mo.ui.table(_ind_counts),
        mo.md("### Paralytic distribution"),
        mo.ui.table(_par_counts),
        mo.md(f"""### Summary
    | Metric | Value |
    |--------|-------|
    | Total RSI events | {cohort_out.height:,} |
    | Unique patients | {cohort_out['patient_id'].n_unique():,} |
    | Median age | {cohort_out['age_at_admission'].median():.0f} |
    | Median weight (kg) | {cohort_out['weight_kg'].median():.1f} |
    | Median induction dose/kg | {cohort_out['induction_dose_per_kg'].median():.2f} |"""),
    ])
    return


@app.cell
def _(OUTPUT_TO_SHARE_DIR, cohort_out, mo, pl):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _plot_data = (
        cohort_out
        .with_columns(
            (pl.col("med_category_ind") + " + " + pl.col("med_category_par")).alias("med_pair")
        )
        .group_by(["med_pair", "location_category"])
        .len()
        .sort("med_pair")
    )

    _pivot = _plot_data.pivot(
        on="location_category", index="med_pair", values="len",
    ).fill_null(0).sort("med_pair")

    _pairs = _pivot["med_pair"].to_list()
    _ed = _pivot["ed"].to_list() if "ed" in _pivot.columns else [0] * len(_pairs)
    _icu = _pivot["icu"].to_list() if "icu" in _pivot.columns else [0] * len(_pairs)

    _x = _np.arange(len(_pairs))
    _width = 0.35

    _fig, _ax = _plt.subplots(figsize=(10, 5))
    _bars_ed = _ax.bar(_x - _width / 2, _ed, _width, label="ED", color="#1f77b4")
    _bars_icu = _ax.bar(_x + _width / 2, _icu, _width, label="ICU", color="#ff7f0e")

    for _bars in (_bars_ed, _bars_icu):
        for _bar in _bars:
            _h = _bar.get_height()
            if _h > 0:
                _ax.text(_bar.get_x() + _bar.get_width() / 2., _h,
                         f'{int(_h)}', ha='center', va='bottom', fontsize=9)

    _ax.set_xlabel("Induction-Paralytic Pair")
    _ax.set_ylabel("Count")
    _ax.set_title("RSI Location at Index Time by Induction-Paralytic Pair")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_pairs)
    _ax.legend()
    _plt.tight_layout()

    _plots_dir = OUTPUT_TO_SHARE_DIR / "plots"
    _plots_dir.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_plots_dir / "rsi_location_by_med_pair.png", dpi=150, bbox_inches="tight")
    mo.as_html(_fig)
    return


@app.cell
def _(OUTPUT_DIR, OUTPUT_TO_SHARE_DIR, cohort_out, mo, pl, weights_pl):
    import matplotlib.pyplot as _plt

    # --- Weight Timing & Change Sub-Analysis ---
    _final_ids = cohort_out["hospitalization_id"].unique()
    _wt = weights_pl.filter(pl.col("hospitalization_id").is_in(_final_ids))

    _idx = cohort_out.select(["hospitalization_id", "index_dttm", "med_category_ind", "med_category_par", "location_category"])

    _wt_idx = _wt.join(_idx, on="hospitalization_id", how="inner")

    # Last weight BEFORE RSI
    _pre = (
        _wt_idx
        .filter(
            (pl.col("recorded_dttm") <= pl.col("index_dttm"))
            & (((pl.col("index_dttm") - pl.col("recorded_dttm")).dt.total_seconds() / 3600) <= 100)
        )
        .with_columns(
            ((pl.col("index_dttm") - pl.col("recorded_dttm")).dt.total_seconds() / 3600).alias("pre_weight_hours_before")
        )
        .sort("pre_weight_hours_before")
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("recorded_dttm").alias("pre_weight_dttm"),
            pl.col("vital_value").alias("pre_weight_kg"),
            "pre_weight_hours_before",
        ])
    )

    # First weight AFTER RSI
    _post = (
        _wt_idx
        .filter(
            (pl.col("recorded_dttm") > pl.col("index_dttm"))
            & (((pl.col("recorded_dttm") - pl.col("index_dttm")).dt.total_seconds() / 3600) <= 100)
        )
        .with_columns(
            ((pl.col("recorded_dttm") - pl.col("index_dttm")).dt.total_seconds() / 3600).alias("post_weight_hours_after")
        )
        .sort("post_weight_hours_after")
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("recorded_dttm").alias("post_weight_dttm"),
            pl.col("vital_value").alias("post_weight_kg"),
            "post_weight_hours_after",
        ])
    )

    # Combine
    weight_timing = (
        _idx
        .join(_pre, on="hospitalization_id", how="left")
        .join(_post, on="hospitalization_id", how="left")
        .with_columns([
            ((pl.col("post_weight_dttm") - pl.col("pre_weight_dttm")).dt.total_seconds() / 3600).alias("pre_to_post_hours"),
            (pl.col("post_weight_kg") - pl.col("pre_weight_kg")).alias("weight_change_kg"),
            (pl.col("med_category_ind") + " + " + pl.col("med_category_par")).alias("med_pair"),
        ])
    )

    # --- Aggregate stats ---
    _metrics = ["pre_weight_hours_before", "post_weight_hours_after", "pre_to_post_hours", "weight_change_kg"]

    def _agg_stats(df, group_col=None):
        rows = []
        if group_col is not None:
            groups = df.select(group_col).unique().sort(group_col).to_series().to_list()
        else:
            groups = [None]
        for g in groups:
            sub = df if g is None else df.filter(pl.col(group_col) == g)
            for m in _metrics:
                vals = sub[m].drop_nulls()
                if vals.len() == 0:
                    continue
                rows.append({
                    "group_by": group_col or "overall",
                    "group_value": g or "all",
                    "metric": m,
                    "n": vals.len(),
                    "median": vals.median(),
                    "mean": vals.mean(),
                    "p25": vals.quantile(0.25),
                    "p75": vals.quantile(0.75),
                    "min": vals.min(),
                    "max": vals.max(),
                })
        return rows

    _all_rows = _agg_stats(weight_timing)
    for _gcol in ["med_category_ind", "med_category_par", "location_category", "med_pair"]:
        _all_rows.extend(_agg_stats(weight_timing, _gcol))

    stats_df = pl.DataFrame(_all_rows)

    # --- Save outputs ---
    weight_timing.write_parquet(OUTPUT_DIR / "subanalysis_weight_timing.parquet")
    stats_df.write_csv(OUTPUT_TO_SHARE_DIR / "subanalysis_weight_timing_stats.csv")
    print(f"Weight timing parquet saved: {OUTPUT_DIR / 'subanalysis_weight_timing.parquet'}")
    print(f"Weight timing stats CSV saved: {OUTPUT_TO_SHARE_DIR / 'subanalysis_weight_timing_stats.csv'}")
    print(f"Weight timing records: {weight_timing.height:,} | With pre-weight: {weight_timing['pre_weight_kg'].drop_nulls().len():,} | With post-weight: {weight_timing['post_weight_kg'].drop_nulls().len():,}")

    # --- Plots (2x2) ---
    _fig, _axes = _plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Hours before RSI of last pre-RSI weight
    _pre_hrs = weight_timing["pre_weight_hours_before"].drop_nulls().to_list()
    _axes[0, 0].hist(_pre_hrs, bins=50, edgecolor="black", alpha=0.7)
    _axes[0, 0].set_xlabel("Hours Before RSI")
    _axes[0, 0].set_ylabel("Count")
    _axes[0, 0].set_title("Last Pre-RSI Weight: Hours Before Intubation")

    # Top-right: Hours after RSI of first post-RSI weight
    _post_hrs = weight_timing["post_weight_hours_after"].drop_nulls().to_list()
    _axes[0, 1].hist(_post_hrs, bins=50, edgecolor="black", alpha=0.7, color="orange")
    _axes[0, 1].set_xlabel("Hours After RSI")
    _axes[0, 1].set_ylabel("Count")
    _axes[0, 1].set_title("First Post-RSI Weight: Hours After Intubation")

    # Bottom-left: Weight change (kg)
    _wt_change = weight_timing["weight_change_kg"].drop_nulls().to_list()
    if _wt_change:
        _axes[1, 0].hist(_wt_change, bins=range(int(min(_wt_change)), int(max(_wt_change)) + 2), edgecolor="black", alpha=0.7, color="green")
        _axes[1, 0].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    _axes[1, 0].set_xlabel("Weight Change (kg)")
    _axes[1, 0].set_ylabel("Count")
    _axes[1, 0].set_title("Weight Change: Pre to Post RSI")

    # Bottom-right: Box plot of weight change by med pair
    _pairs = weight_timing.select(["med_pair", "weight_change_kg"]).drop_nulls()
    _pair_labels = sorted(_pairs["med_pair"].unique().to_list())
    _box_data = [_pairs.filter(pl.col("med_pair") == p)["weight_change_kg"].to_list() for p in _pair_labels]
    if any(len(d) > 0 for d in _box_data):
        _axes[1, 1].boxplot(_box_data, labels=_pair_labels, patch_artist=True)
        _axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=1)
    _axes[1, 1].set_xlabel("Induction-Paralytic Pair")
    _axes[1, 1].set_ylabel("Weight Change (kg)")
    _axes[1, 1].set_title("Weight Change by Medication Pair")
    _axes[1, 1].tick_params(axis="x", rotation=15)

    _plt.tight_layout()
    _plots_dir = OUTPUT_TO_SHARE_DIR / "plots"
    _plots_dir.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_plots_dir / "weight_timing.png", dpi=150, bbox_inches="tight")
    mo.as_html(_fig)
    return


@app.cell
def _(
    DATA_DIR,
    FILETYPE,
    OUTPUT_TO_SHARE_DIR,
    TIMEZONE,
    Vitals,
    cohort_e4,
    cohort_e5,
    hosp_pl,
    mo,
    pl,
    weights_pl,
):
    import matplotlib.pyplot as _plt

    # --- Sub-analysis: weight gap for patients excluded by no pre-RSI weight ---
    _excl_ids = cohort_e4.filter(
        ~pl.col("hospitalization_id").is_in(cohort_e5["hospitalization_id"])
    )
    _excl_with_patient = _excl_ids.select(["hospitalization_id", "index_dttm"]).join(
        hosp_pl.select(["hospitalization_id", "patient_id", "admission_dttm"]),
        on="hospitalization_id",
        how="left",
    )

    # Find all prior hospitalizations within 1 year for these patients
    _all_hosp = hosp_pl.select(["hospitalization_id", "patient_id", "admission_dttm"]).rename(
        {"hospitalization_id": "prior_hosp_id", "admission_dttm": "prior_adm_dttm"}
    )
    _prior = (
        _excl_with_patient.select(["hospitalization_id", "patient_id", "admission_dttm"])
        .join(_all_hosp, on="patient_id", how="inner")
        .filter(
            (pl.col("prior_adm_dttm") < pl.col("admission_dttm"))
            & (
                (pl.col("admission_dttm") - pl.col("prior_adm_dttm")).dt.total_seconds()
                <= 365.25 * 24 * 3600
            )
        )
    )
    _prior_hosp_ids = _prior["prior_hosp_id"].unique().to_list()

    # Load weight data for prior hospitalizations
    if _prior_hosp_ids:
        _prior_vit = Vitals.from_file(
            data_directory=DATA_DIR,
            filetype=FILETYPE,
            timezone=TIMEZONE,
            filters={
                "hospitalization_id": _prior_hosp_ids,
                "vital_category": ["weight_kg"],
            },
        )
        _prior_vit_pd = _prior_vit.df.copy()
        _prior_vit_pd["recorded_dttm"] = _prior_vit_pd["recorded_dttm"].dt.tz_localize(None)
        _prior_weights = pl.from_pandas(_prior_vit_pd).select(
            ["hospitalization_id", "recorded_dttm", "vital_value"]
        )
        del _prior_vit_pd

        # Map prior hospitalization weights back to current hospitalization via patient_id
        _prior_mapped = (
            _prior_weights.rename({"hospitalization_id": "prior_hosp_id"})
            .join(
                _prior.select(["hospitalization_id", "prior_hosp_id"]).unique(),
                on="prior_hosp_id",
                how="inner",
            )
            .select(["hospitalization_id", "recorded_dttm", "vital_value"])
        )
    else:
        _prior_mapped = pl.DataFrame(
            schema={"hospitalization_id": pl.Utf8, "recorded_dttm": pl.Datetime, "vital_value": pl.Float64}
        )

    # Current hospitalization weights for excluded patients
    _curr_weights = weights_pl.filter(
        pl.col("hospitalization_id").is_in(_excl_ids["hospitalization_id"])
    ).select(["hospitalization_id", "recorded_dttm", "vital_value"])

    # Combine all available weights
    _all_weights = pl.concat([_curr_weights, _prior_mapped], how="diagonal_relaxed")

    _idx = _excl_with_patient.select(["hospitalization_id", "index_dttm"])
    _wt_idx = _all_weights.join(_idx, on="hospitalization_id", how="inner")

    # Last weight BEFORE RSI (from any hospitalization within 1 year)
    _pre = (
        _wt_idx.filter(pl.col("recorded_dttm") < pl.col("index_dttm"))
        .with_columns(
            ((pl.col("index_dttm") - pl.col("recorded_dttm")).dt.total_seconds() / 3600)
            .alias("hours_before")
        )
        .sort("hours_before")
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("recorded_dttm").alias("pre_weight_dttm"),
            pl.col("vital_value").alias("pre_weight_kg"),
        ])
    )

    # First weight AFTER RSI (current hospitalization only)
    _post = (
        _curr_weights.join(_idx, on="hospitalization_id", how="inner")
        .filter(pl.col("recorded_dttm") > pl.col("index_dttm"))
        .with_columns(
            ((pl.col("recorded_dttm") - pl.col("index_dttm")).dt.total_seconds() / 3600)
            .alias("hours_after")
        )
        .sort("hours_after")
        .group_by("hospitalization_id")
        .first()
        .select([
            "hospitalization_id",
            pl.col("recorded_dttm").alias("post_weight_dttm"),
            pl.col("vital_value").alias("post_weight_kg"),
        ])
    )

    # Combine and compute metrics
    _merged = (
        _idx.join(_pre, on="hospitalization_id", how="inner")
        .join(_post, on="hospitalization_id", how="inner")
        .with_columns([
            ((pl.col("post_weight_dttm") - pl.col("pre_weight_dttm")).dt.total_seconds() / (24 * 3600))
            .alias("time_diff_days"),
            (pl.col("post_weight_kg") - pl.col("pre_weight_kg")).alias("weight_diff_kg"),
        ])
    )

    _n_excl = _excl_ids.height
    _n_with_both = _merged.height
    print(f"Weight gap sub-analysis: {_n_excl} excluded patients, {_n_with_both} with both pre (1yr lookback) and post-RSI weights")

    # Scatter plot
    _fig, _ax = _plt.subplots(figsize=(10, 7))
    if _merged.height > 0:
        _x = _merged["time_diff_days"].to_list()
        _y = _merged["weight_diff_kg"].to_list()
        _ax.scatter(_x, _y, alpha=0.5, edgecolors="black", linewidths=0.3, s=30,
                    label=f"With both weights (n={_n_with_both})")
        _ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    _ax.legend(title=f"Total excluded: {_n_excl}")
    _ax.set_xlabel("Time Between Last Pre-RSI Weight and First Post-RSI Weight (days)")
    _ax.set_ylabel("Weight Difference: Post - Pre (kg)")
    _ax.set_title("Weight Gap for Excluded Patients (No Pre-RSI Weight During Hospitalization)")
    _plt.tight_layout()
    _plots_dir = OUTPUT_TO_SHARE_DIR / "plots"
    _plots_dir.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_plots_dir / "weight_gap_excluded.png", dpi=150, bbox_inches="tight")
    mo.as_html(_fig)
    return


if __name__ == "__main__":
    app.run()
