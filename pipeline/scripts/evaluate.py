# evaluate.py
import argparse
import json
import os
import tarfile
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

MODEL_INPUT_DIR = Path("/opt/ml/processing/model")
EVAL_INPUT_DIR = Path("/opt/ml/processing/input")
EVAL_OUTPUT_DIR = Path("/opt/ml/processing/evaluation")


def find_predictor_dir(root: Path) -> Path:
    # Find a directory containing predictor/learner artifacts
    for p, dnames, fnames in os.walk(root):
        if "predictor.pkl" in fnames or "learner.pkl" in fnames:
            return Path(p)
    return root


def load_predictor() -> TabularPredictor:
    # Model is provided as model.tar.gz via ProcessingInput
    tar_path = MODEL_INPUT_DIR / "model.tar.gz"
    extract_dir = MODEL_INPUT_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)
    pred_dir = find_predictor_dir(extract_dir)
    return TabularPredictor.load(pred_dir)


def align_eval_columns(X: pd.DataFrame, predictor: TabularPredictor, label_col: str) -> pd.DataFrame:
    # Try to get the exact feature names used during training
    expected_cols = None
    try:
        expected_cols = list(predictor._learner.feature_generator.features_in)
    except Exception:
        expected_cols = None

    if expected_cols is not None:
        if len(expected_cols) != X.shape[1]:
            raise ValueError(
                f"Eval features ({X.shape[1]}) != training features ({len(expected_cols)}). "
                f"Did you drop the label column '{label_col}'? Columns in eval: {list(X.columns)[:5]}..."
            )
        X = X.copy()
        X.columns = expected_cols
    else:
        # Fallback: common pattern for headerless training is feature_0..feature_{n-1}
        X = X.copy()
        X.columns = [f"feature_{i}" for i in range(X.shape[1])]
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifacts-s3", type=str, required=False)  # not used directly; artifacts are mounted
    parser.add_argument("--eval-data-s3", type=str, required=False)        # not used directly; data are mounted
    parser.add_argument("--label-column", type=str, required=True)
    parser.add_argument("--problem-type", type=str, default="classification")
    args = parser.parse_args()

    predictor = load_predictor()

    # Locate the eval file (your pipeline copies it as 'val_with_header.csv')
    eval_csv = EVAL_INPUT_DIR / "val_with_header.csv"
    if not eval_csv.exists():
        # fallback: look for any CSV in the input dir
        candidates = list(EVAL_INPUT_DIR.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No evaluation CSV found in {EVAL_INPUT_DIR}")
        eval_csv = candidates[0]

    df = pd.read_csv(eval_csv)
    if args.label_column not in df.columns:
        raise AssertionError(f"Label column '{args.label_column}' not found in {eval_csv.name}. "
                             f"Columns={list(df.columns)}")

    y = df[args.label_column]
    X = df.drop(columns=[args.label_column])

    # Align names to what the model expects (fix for headerless training)
    X = align_eval_columns(X, predictor, args.label_column)

    # Predict and score
    y_pred = predictor.predict(X)
    metrics = {}

    if args.problem_type == "classification":
        acc = accuracy_score(y, y_pred)
        metrics["accuracy"] = float(acc)
        # Optional extras (try/catch if multiclass/imbalanced)
        try:
            y_proba = predictor.predict_proba(X)
            # If binary, get proba for positive class
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y, y_proba.iloc[:, 1] if hasattr(y_proba, "iloc") else y_proba[:, 1])
                metrics["roc_auc"] = float(auc)
        except Exception:
            pass
        try:
            f1 = f1_score(y, y_pred, average="binary")
            metrics["f1"] = float(f1)
        except Exception:
            pass
        primary_name = "accuracy"
        primary_value = metrics["accuracy"]
    else:
        # If you ever add regression, plug in rmse/mae here
        raise NotImplementedError("This evaluation script currently supports classification only.")

    # Write evaluation.json for the ConditionStep (expects 'primary_metric.value')
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "primary_metric": {"name": primary_name, "value": float(primary_value)},
        "metrics": metrics,
    }
    with open(EVAL_OUTPUT_DIR / "evaluation.json", "w") as f:
        json.dump(report, f)
    print("Wrote", EVAL_OUTPUT_DIR / "evaluation.json", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

