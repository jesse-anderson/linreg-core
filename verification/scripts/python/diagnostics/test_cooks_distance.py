# ============================================================================
# Cook's Distance Reference Implementation (Python)
# ============================================================================
# This script generates reference values for Cook's distance using
# statsmodels.stats.outliers_influence.OLSInfluence.cooks_distance.
# Cook's distance measures how much each observation influences the
# regression model.
#
# Source: statsmodels package, OLSInfluence.cooks_distance
# Reference: Cook, R. D. (1977), "Detection of Influential Observations in
#            Linear Regression", Technometrics, 19(1), 15-18
#
# Usage:
#   python test_cooks_distance.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR]
#
#   Args:
#     --csv       Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     --output-dir Path to output directory
#                 Default: ../../results/python
# ============================================================================

import argparse
import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence


def convert_categorical_to_numeric(data, dataset_name):
    """Convert categorical variables to numeric representations."""
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    if not non_numeric_cols:
        return []

    print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: {non_numeric_cols}")
    print(f"Converting categorical variables to numeric representations...")

    # Process each non-numeric column
    for col in non_numeric_cols:
        if data[col].dtype == 'object':
            # For string/categorical data, use factorize (integer encoding)
            # This is similar to R's factor() function
            encoded, categories = pd.factorize(data[col])
            data[col] = encoded
            print(f"  {col}: {len(categories)} unique categories -> encoded as 0, 1, 2, ...")
            print(f"    Categories: {list(categories)}")
        else:
            # Already numeric or other type
            pass

    return non_numeric_cols


def main():
    parser = argparse.ArgumentParser(
        description="Cook's Distance - Identify influential observations using statsmodels"
    )
    parser.add_argument(
        "--csv",
        default="../../datasets/csv/mtcars.csv",
        help="Path to CSV file (first column = response, rest = predictors)"
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/python",
        help="Path to output directory"
    )

    args = parser.parse_args()

    # Validate CSV path
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    # Extract dataset name from filename
    dataset_name = os.path.splitext(os.path.basename(args.csv))[0]

    # Read CSV data
    data = pd.read_csv(args.csv)

    # If there are non-numeric columns, convert them to numeric
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        convert_categorical_to_numeric(data, dataset_name)

    # After conversion, re-check if we still have non-numeric columns
    remaining_non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if remaining_non_numeric:
        # These could be things like empty strings or other non-numeric types that couldn't be converted
        raise ValueError(f"Could not convert the following non-numeric columns to numeric: {remaining_non_numeric}")

    # Assume first column is response variable, rest are predictors
    response_col = data.columns[0]
    predictor_cols = data.columns[1:]

    # Prepare data for statsmodels (add constant for intercept)
    X = data[predictor_cols]
    X = sm.add_constant(X)
    y = data[response_col]

    # Build formula string for output
    formula_str = f"{response_col} ~ {' + '.join(predictor_cols)}"

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Compute Cook's distance using OLSInfluence
    influence = OLSInfluence(model)
    cooks_d = influence.cooks_distance[0]  # First element is distances, second is p-values

    # Compute model info
    n = len(y)
    p = len(model.params)  # number of parameters including intercept
    df_residual = model.df_resid
    mse = model.mse_resid

    # Compute thresholds
    threshold_4_over_n = 4.0 / n
    threshold_4_over_df = 4.0 / df_residual
    threshold_1 = 1.0

    # Identify influential observations (1-based indexing)
    influential_4_over_n = np.where(cooks_d > threshold_4_over_n)[0] + 1
    influential_4_over_df = np.where(cooks_d > threshold_4_over_df)[0] + 1
    influential_1 = np.where(cooks_d > threshold_1)[0] + 1

    # Find max Cook's distance
    max_idx = np.argmax(cooks_d) + 1
    max_d = cooks_d[max_idx - 1]

    # Print results
    print("Cook's Distance (Python - statsmodels)")
    print("=" * 42)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"n: {n}")
    print(f"p: {p}")
    print(f"MSE: {mse}")
    print(f"Max Cook's D: {max_d} (observation {max_idx})")
    print(f"Threshold 4/n: {threshold_4_over_n}")
    print(f"Threshold 4/(n-p): {threshold_4_over_df}")
    print(f"Threshold 1: {threshold_1}")
    print(f"Influential (4/n): {len(influential_4_over_n)} observations")
    print(f"Influential (4/(n-p)): {len(influential_4_over_df)} observations")
    print(f"Influential (>1): {len(influential_1)} observations")
    if len(influential_1) > 0:
        print(f"Highly influential indices: {influential_1.tolist()}")
    print()

    # Prepare output - convert numpy types to native Python types
    output = {
        "test_name": "Cook's Distance (Python - statsmodels)",
        "dataset": dataset_name,
        "formula": formula_str,
        "distances": [float(d) for d in cooks_d],
        "p": p,
        "mse": float(mse),
        "threshold_4_over_n": float(threshold_4_over_n),
        "threshold_4_over_df": float(threshold_4_over_df),
        "threshold_1": float(threshold_1),
        "influential_4_over_n": [int(i) for i in influential_4_over_n],
        "influential_4_over_df": [int(i) for i in influential_4_over_df],
        "influential_1": [int(i) for i in influential_1],
        "max_distance": float(max_d),
        "max_index": int(max_idx),
        "description": "Measures influence of each observation on regression coefficients. Uses statsmodels.stats.outliers_influence.OLSInfluence.cooks_distance."
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_cooks_distance.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_cooks_distance.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
