# ============================================================================
# DFFITS Reference Implementation (Python)
# ============================================================================
# This script generates reference values for DFFITS using
# statsmodels.stats.outliers_influence.OLSInfluence.dffits.
# DFFITS measures the influence of each observation on its own
# fitted value.
#
# Source: statsmodels package, OLSInfluence.dffits
# Reference: Belsley, D. A., Kuh, E., & Welsch, R. E. (1980),
#            "Regression Diagnostics", Wiley
#
# Usage:
#   python test_dffits.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR]
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
        description="DFFITS - Identify influential observations on fitted values using statsmodels"
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

    # Compute DFFITS using OLSInfluence
    influence = OLSInfluence(model)
    dff = influence.dffits[0]

    # Compute model info
    n = len(y)
    p = len(model.params)  # number of parameters including intercept

    # Compute threshold: 2*sqrt(p/n)
    threshold = 2.0 * np.sqrt(p / n)

    # Identify influential observations (|DFFITS| > threshold)
    # Returns 1-based indices of observations that exceed threshold
    influential_indices = np.where(np.abs(dff) > threshold)[0] + 1

    # Find max absolute DFFITS value and its index
    max_idx = np.argmax(np.abs(dff)) + 1  # 1-based indexing
    max_abs_val = float(np.abs(dff[max_idx - 1]))

    # Print results
    print("DFFITS (Python - statsmodels)")
    print("=" * 40)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"n: {n}")
    print(f"p: {p}")
    print(f"Threshold (2*sqrt(p/n)): {threshold}")
    print(f"Max |DFFITS|: {max_abs_val}")
    print(f"Max index: observation {max_idx}")
    print(f"Influential observations: {len(influential_indices)}")
    if len(influential_indices) > 0:
        print(f"Influential indices: {influential_indices.tolist()}")
    else:
        print("Influential indices: none")
    print()

    # Prepare output - convert numpy types to native Python types
    output = {
        "test_name": "DFFITS (Python - statsmodels)",
        "dataset": dataset_name,
        "formula": formula_str,
        "dffits": [float(d) for d in dff],
        "n": n,
        "p": p,
        "threshold": float(threshold),
        "influential_observations": [int(i) for i in influential_indices],
        "description": "Measures influence of each observation on its fitted value."
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_dffits.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_dffits.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
