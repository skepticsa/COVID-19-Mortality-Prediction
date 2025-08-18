# preprocess.py
# Prepares data for JumpStart AutoGluon Tabular training + evaluation inside a Processing step.
# - Input:  /opt/ml/processing/input/engineered.csv  (headered; includes the label column)
# - Outputs:
#     /opt/ml/processing/train/train.csv            (NO header, label first)
#     /opt/ml/processing/val/validation.csv        (NO header, label first)
#     /opt/ml/processing/eval/val_with_header.csv  (WITH header, label + features in train order)
#     /opt/ml/processing/meta/*.json               (summary + schema)
#
# Notes:
#   * Headerless + label-first format is what your TrainingStep expects.
#   * The Evaluate step reads the headered `val_with_header.csv` and
#     renames columns to the model’s expected `feature_*` names before prediction.

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_DIR   = Path("/opt/ml/processing/input")
TRAIN_DIR   = Path("/opt/ml/processing/train")
VAL_DIR     = Path("/opt/ml/processing/val")
EVAL_DIR    = Path("/opt/ml/processing/eval")
META_DIR    = Path("/opt/ml/processing/meta")

DEFAULT_INPUT_FILE = "engineered.csv"
RANDOM_STATE = 2024  # reproducible split


def _find_input_file(preferred: Optional[str]) -> Path:
    """
    Locate the input file inside INPUT_DIR. If `preferred` is provided (e.g., '.../engineered.csv'),
    use its basename; otherwise pick 'engineered.csv' if present, else the largest CSV in the folder.
    """
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if preferred:
        cand = INPUT_DIR / Path(preferred).name
        if cand.exists():
            return cand

    # Prefer engineered.csv
    engineered = INPUT_DIR / DEFAULT_INPUT_FILE
    if engineered.exists():
        return engineered

    # Fallback: any CSV (pick the largest to avoid picking tiny side files)
    csvs = list(INPUT_DIR.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {INPUT_DIR}. "
                                f"Pass --input-s3-uri pointing at a CSV (e.g. engineered.csv).")
    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return csvs[0]


def _ensure_label(df: pd.DataFrame, label_col: str) -> None:
    if label_col not in df.columns:
        raise AssertionError(f"Label column '{label_col}' not found. "
                             f"Available columns: {list(df.columns)}")
    if df[label_col].isna().any():
        # Drop rows with missing labels
        before = len(df)
        df.dropna(subset=[label_col], inplace=True)
        print(f"Dropped {before - len(df)} rows with NA labels.")


def _split(df: pd.DataFrame, label_col: str, val_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_fraction = float(val_fraction)
    if not (0.0 < val_fraction < 0.9):
        raise ValueError(f"--val-fraction must be in (0, 0.9), got {val_fraction}")

    y = df[label_col]
    stratify = None
    # Use stratified split if classification-like: <= 50 unique labels and all integer-like or object categories
    unique_labels = y.nunique(dropna=True)
    if unique_labels >= 2 and unique_labels <= 50:
        stratify = y

    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify
    )
    print(f"Split: train={len(train_df)} rows, val={len(val_df)} rows (val_fraction={val_fraction}).")
    return train_df, val_df


def _reorder_label_first(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != label_col]
    return df[[label_col] + feature_cols], feature_cols


def _write_headerless_label_first(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # header=False, label first already enforced by caller
    df.to_csv(out_path, index=False, header=False)
    print(f"Wrote {out_path}  (headerless, label-first)  rows={len(df)}, cols={df.shape[1]}")


def _write_eval_with_header(val_df: pd.DataFrame, label_col: str, feature_cols: List[str]) -> Path:
    """
    Save an evaluation CSV WITH header. Keep the same column order used in training:
      [label] + feature_cols
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ordered = val_df[[label_col] + feature_cols]
    out_path = EVAL_DIR / "val_with_header.csv"
    ordered.to_csv(out_path, index=False, header=True)
    print(f"Wrote {out_path}  (with header)  rows={len(ordered)}, cols={ordered.shape[1]}")
    return out_path


def _write_meta(df_all: pd.DataFrame, label_col: str, feature_cols: List[str]) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows_total": int(len(df_all)),
        "label_column": label_col,
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
    }
    with open(META_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # Also write a tiny preview (up to 50 rows)
    preview = df_all[[label_col] + feature_cols].head(50)
    preview.to_csv(META_DIR / "preview.csv", index=False)
    print(f"Wrote meta: {META_DIR/'summary.json'}, {META_DIR/'preview.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-s3-uri", type=str, required=True,
                        help="S3 URI or local filename for the engineered CSV (pipeline passes the file name).")
    parser.add_argument("--label-column", type=str, required=True,
                        help="Name of the label column (e.g., 'mortality').")
    parser.add_argument("--problem-type", type=str, default="classification",
                        help="Only 'classification' is used by the pipeline.")
    parser.add_argument("--val-fraction", type=str, default="0.2",
                        help="Validation fraction in (0, 0.9).")
    args = parser.parse_args()

    print("==== Preprocess: configuration ====")
    print(f"input-s3-uri     : {args.input_s3_uri}")
    print(f"label-column     : {args.label_column}")
    print(f"problem-type     : {args.problem_type}")
    print(f"val-fraction     : {args.val_fraction}")

    # 1) Locate and load the engineered CSV
    in_file = _find_input_file(args.input_s3_uri)
    print(f"Resolved input file: {in_file}")
    df = pd.read_csv(in_file)
    print(f"Loaded shape: {df.shape}")

    # 2) Validate label & basic cleanup
    _ensure_label(df, args.label_column)
    # Optional: ensure label is int/binary if it looks like that (non-strict).
    # If you know your label is 0/1, uncomment the next two lines:
    # df[args.label_column] = pd.to_numeric(df[args.label_column], errors="coerce")
    # df.dropna(subset=[args.label_column], inplace=True)

    # 3) Train/val split (stratified when feasible)
    train_df, val_df = _split(df, args.label_column, float(args.val_fraction))

    # 4) Reorder columns so LABEL is FIRST (for JumpStart training format)
    train_ordered, feature_cols = _reorder_label_first(train_df, args.label_column)
    val_ordered, _            = _reorder_label_first(val_df,   args.label_column)

    # 5) Write headerless CSVs for training channels
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    _write_headerless_label_first(train_ordered, TRAIN_DIR / "train.csv")
    _write_headerless_label_first(val_ordered,   VAL_DIR   / "validation.csv")

    # 6) Write evaluation CSV WITH header (label + features in SAME order)
    _write_eval_with_header(val_df, args.label_column, feature_cols)

    # 7) Write metadata for traceability
    _write_meta(df_all=df, label_col=args.label_column, feature_cols=feature_cols)

    print("Preprocess completed.")


if __name__ == "__main__":
    main()

