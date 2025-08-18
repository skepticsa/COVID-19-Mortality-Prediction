#!/usr/bin/env python3
import argparse
import os
import json
from typing import List
import boto3
import numpy as np
import pandas as pd

YES_CODE = 1             # dataset uses 1=yes, 2=no, 97/99 unknown
NO_LIKE = {2, 97, 99}    # treat unknown/not_appl as no for booleans

def parse_s3_uri(s3_uri: str):
    assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
    b, k = s3_uri[5:].split("/", 1)
    return b, k

def is_yes(sr: pd.Series) -> pd.Series:
    s = pd.to_numeric(sr, errors="coerce").fillna(2).astype(int)
    return (s == YES_CODE).astype(int)

def count_yes(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    return sum(is_yes(df[c]) for c in cols)

def main():
    p = argparse.ArgumentParser("Feature engineering for COVID dataset (creates binary mortality).")
    p.add_argument("--input-s3-uri", required=True, help="s3://bucket/key to raw CSV with header")
    p.add_argument("--label-name", default="mortality", help="Output label column name")
    p.add_argument("--elderly-age", type=int, default=65)
    p.add_argument("--delay-threshold-days", type=int, default=2)
    p.add_argument("--output-filename", default="engineered.csv")
    args = p.parse_args()

    # I/O layout in Processing container
    in_dir = "/opt/ml/processing/input"
    out_dir = "/opt/ml/processing/engineered"
    meta_dir = "/opt/ml/processing/meta"
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    # Download raw CSV from S3
    bucket, key = parse_s3_uri(args.input_s3_uri)
    local_csv = os.path.join(in_dir, "data.csv")
    boto3.client("s3").download_file(bucket, key, local_csv)

    # Load
    df = pd.read_csv(local_csv)

    # --- Parse dates and build label ---
    # Dates are formatted like '04-05-2020' and '9999-99-99' as sentinel
    for dc in ["entry_date", "date_symptoms", "date_died"]:
        if dc in df.columns:
            df[dc] = df[dc].replace("9999-99-99", pd.NA)
            df[dc] = pd.to_datetime(df[dc], errors="coerce", dayfirst=True)

    # mortality: 1 if date_died is a valid date; 0 otherwise
    df[args.label_name] = df.get("date_died", pd.Series([pd.NaT]*len(df))).notna().astype(int)

    # --- Coerce expected numeric code columns ---
    # (keeps original 1/2 codes; unknowns default to 2=no)
    code_cols = [
        "sex","patient_type","pneumonia","pregnancy","diabetes","copd","asthma","inmsupr",
        "hypertension","other_disease","cardiovascular","obesity","renal_chronic","tobacco",
        "contact_other_covid","icu","intubed"
    ]
    for c in code_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(2).astype(int)

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    else:
        df["age"] = 0

    # --- Feature blocks ---
    comorb_cols = [
        "diabetes","copd","asthma","inmsupr","hypertension","other_disease",
        "cardiovascular","obesity","renal_chronic"
    ]
    df["num_comorbidities"] = count_yes(df, comorb_cols)

    df["critical_care"] = ((df.get("icu", 2) == YES_CODE) | (df.get("intubed", 2) == YES_CODE)).astype(int)
    df["is_elderly"] = (df["age"] >= args.elderly_age).astype(int)

    df["has_respiratory"] = (
        (df.get("pneumonia", 2) == YES_CODE) |
        (df.get("copd", 2) == YES_CODE) |
        (df.get("asthma", 2) == YES_CODE)
    ).astype(int)

    df["has_cardiovascular"] = (
        (df.get("cardiovascular", 2) == YES_CODE) |
        (df.get("hypertension", 2) == YES_CODE)
    ).astype(int)

    # Days from symptoms to entry & delayed admission
    if "entry_date" in df.columns and "date_symptoms" in df.columns:
        d = (df["entry_date"] - df["date_symptoms"]).dt.days
        df["days_symptoms_to_entry"] = d.clip(lower=0).fillna(0).astype(int)
    else:
        df["days_symptoms_to_entry"] = 0
    df["delayed_admission"] = (df["days_symptoms_to_entry"] > args.delay_threshold_days).astype(int)

    # Interactions and composites mirroring your notebook
    df["elderly_diabetic"] = (df["is_elderly"] & (df.get("diabetes", 2) == YES_CODE)).astype(int)
    df["elderly_respiratory"] = (df["is_elderly"] & (df["has_respiratory"] == 1)).astype(int)
    df["elderly_cardiovascular"] = (df["is_elderly"] & (df["has_cardiovascular"] == 1)).astype(int)
    df["elderly_multiple_risks"] = (df["is_elderly"] & (df["num_comorbidities"] >= 2)).astype(int)

    df["severe_case"] = ((df["critical_care"] == 1) | (df.get("pneumonia", 2) == YES_CODE)).astype(int)
    df["high_risk_profile"] = ((df["is_elderly"] & (df["severe_case"] == 1)) | (df["num_comorbidities"] >= 3)).astype(int)

    df["age_squared"] = (df["age"] ** 2).astype(int)
    df["age_log"] = np.log1p(df["age"])

    df["diabetes_hypertension"] = ((df.get("diabetes", 2) == YES_CODE) & (df.get("hypertension", 2) == YES_CODE)).astype(int)
    df["obesity_diabetes"] = ((df.get("obesity", 2) == YES_CODE) & (df.get("diabetes", 2) == YES_CODE)).astype(int)
    df["respiratory_cardiovascular"] = (df["has_respiratory"] & df["has_cardiovascular"]).astype(int)

    # --- Choose columns to keep (your engineered set + label) ---
    # Order below matches your example (adjust freely)
    keep = [
        "age","sex","patient_type","pneumonia","pregnancy","diabetes","copd","asthma","inmsupr","hypertension",
        "other_disease","cardiovascular","obesity","renal_chronic","tobacco","contact_other_covid","icu","intubed",
        "num_comorbidities","critical_care","is_elderly","has_respiratory","has_cardiovascular","days_symptoms_to_entry",
        "delayed_admission","elderly_diabetic","elderly_respiratory","elderly_cardiovascular","elderly_multiple_risks",
        "severe_case","high_risk_profile","age_squared","age_log","diabetes_hypertension","obesity_diabetes",
        "respiratory_cardiovascular", args.label_name,
    ]
    keep = [c for c in keep if c in df.columns]  # tolerate missing
    out = df[keep].copy()

    # Write engineered CSV (headered, label is *last* for readability here;
    # the downstream preprocess step will reorder to label-first & remove header)
    out_path = os.path.join(out_dir, args.output_filename)
    out.to_csv(out_path, index=False)

    # Emit meta for traceability
    with open(os.path.join(meta_dir, "feature_meta.json"), "w") as f:
        json.dump({
            "input_s3": args.input_s3_uri,
            "rows": int(len(out)),
            "cols": list(out.columns),
            "label": args.label_name,
            "elderly_age": args.elderly_age,
            "delay_threshold_days": args.delay_threshold_days,
            "output_csv": out_path,
        }, f, indent=2)

    print("✅ Feature engineering complete:", out_path)

if __name__ == "__main__":
    main()


