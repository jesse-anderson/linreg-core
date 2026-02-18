# ============================================================================
# Prediction Intervals Reference Implementation (Python with statsmodels)
# ============================================================================
# This script generates reference values for OLS prediction intervals using
# statsmodels' get_prediction() method which provides obs_ci_lower/upper
# (prediction intervals).
#
# Source: statsmodels.regression.linear_model.OLS + get_prediction()
#
# Usage:
#   python test_prediction_intervals.py --csv [csv_path] --output-dir [output_dir]
#   Args:
#     --csv        - Path to CSV file (first col = response, rest = predictors)
#                    Default: ../../datasets/csv/mtcars.csv
#     --output-dir - Path to output directory
#                    Default: ../../results/python
# ============================================================================

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
import sys
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Helper for JSON serialization of numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def convert_categorical_to_numeric(data, dataset_name):
    """Convert categorical columns to numeric representations."""
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(non_numeric_cols) > 0:
        print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: "
              f"{', '.join(non_numeric_cols)}")
        print("Converting categorical variables to numeric factor codes...")

        for col in non_numeric_cols:
            numeric_vals = pd.to_numeric(data[col], errors='coerce')
            if not numeric_vals.isna().any():
                data[col] = numeric_vals
            else:
                codes, _ = pd.factorize(data[col])
                data[col] = codes.astype(float)
                print(f"  Column '{col}' encoded as 0-based factor codes")

    return data


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate OLS prediction interval reference values"
    )
    parser.add_argument(
        "--csv",
        default="../../datasets/csv/mtcars.csv",
        help="Path to CSV file (first col = response, rest = predictors)"
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/python",
        help="Path to output directory"
    )

    args = parser.parse_args()

    # Resolve paths - use absolute paths directly, resolve relative paths from cwd
    csv_path = Path(args.csv) if Path(args.csv).is_absolute() else Path.cwd() / args.csv
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else Path.cwd() / args.output_dir

    csv_path = csv_path.resolve()
    output_dir = output_dir.resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset_name = csv_path.stem
    print(f"Processing dataset: {dataset_name}")
    print(f"CSV path: {csv_path}")

    data = pd.read_csv(csv_path)
    data = convert_categorical_to_numeric(data, dataset_name)

    y_name = data.columns[0]
    x_names = data.columns[1:].tolist()

    print(f"Response: {y_name}")
    print(f"Predictors: {', '.join(x_names)}")
    print(f"n = {len(data)}, p = {len(x_names)}")

    y = data[y_name].values
    X = data[x_names].values
    X_with_const = sm.add_constant(X)

    # Fit OLS model
    model = sm.OLS(y, X_with_const).fit()
    print(model.summary())

    # ========================================================================
    # Training data prediction intervals
    # ========================================================================
    pred_train = model.get_prediction(X_with_const)
    frame_train = pred_train.summary_frame(alpha=0.05)

    predicted_train = frame_train['mean'].values
    lower_train = frame_train['obs_ci_lower'].values
    upper_train = frame_train['obs_ci_upper'].values

    # Compute SE_pred = sqrt(MSE * (1 + h))
    mse = model.mse_resid
    leverage_train = model.get_influence().hat_matrix_diag
    se_pred_train = np.sqrt(mse * (1 + leverage_train))

    # ========================================================================
    # Extrapolation points
    # ========================================================================
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min

    # 3 extrapolation points
    new_x = np.zeros((3, len(x_names)))
    new_x[0] = x_max + 0.1 * x_range   # slightly beyond
    new_x[1] = x_max + 0.5 * x_range   # moderately beyond
    new_x[2] = x_max + 1.0 * x_range   # far beyond

    new_x_with_const = sm.add_constant(new_x)

    pred_new = model.get_prediction(new_x_with_const)
    frame_new = pred_new.summary_frame(alpha=0.05)

    predicted_new = frame_new['mean'].values
    lower_new = frame_new['obs_ci_lower'].values
    upper_new = frame_new['obs_ci_upper'].values

    # Compute leverage for new points: h = diag(X_new (X'X)^{-1} X_new')
    xtx_inv = np.linalg.inv(X_with_const.T @ X_with_const)
    hat_new = np.diag(new_x_with_const @ xtx_inv @ new_x_with_const.T)
    se_pred_new = np.sqrt(mse * (1 + hat_new))

    print(f"\n=== Training data prediction intervals (first 5) ===")
    print(frame_train.head())

    print(f"\n=== Extrapolation prediction intervals ===")
    print(frame_new)
    print(f"Leverage (new): {hat_new}")
    print(f"SE_pred (new): {se_pred_new}")

    # ========================================================================
    # Build output JSON
    # ========================================================================
    new_x_dict = {}
    for j, nm in enumerate(x_names):
        new_x_dict[nm] = new_x[:, j].tolist()

    result = {
        "dataset": dataset_name,
        "alpha": 0.05,
        "n": int(len(data)),
        "p": int(len(x_names)),
        "df_residuals": int(model.df_resid),
        "mse": float(mse),

        "train": {
            "predicted": predicted_train.tolist(),
            "lower": lower_train.tolist(),
            "upper": upper_train.tolist(),
            "se_pred": se_pred_train.tolist(),
            "leverage": leverage_train.tolist(),
        },

        "extrapolation": {
            "new_x": new_x_dict,
            "predicted": predicted_new.tolist(),
            "lower": lower_new.tolist(),
            "upper": upper_new.tolist(),
            "se_pred": se_pred_new.tolist(),
            "leverage": hat_new.tolist(),
        },
    }

    # Write JSON output
    output_file = output_dir / f"{dataset_name}_prediction_intervals.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)

    print(f"\nResults written to: {output_file}")
    print("Done.")


if __name__ == "__main__":
    main()
