# ============================================================================
# Jarque-Bera Test Reference Implementation (Python)
# ============================================================================
# This script generates reference values for the Jarque-Bera test using
# statsmodels.stats.stattools.jarque_bera. The Jarque-Bera test checks whether
# residuals are normally distributed by examining skewness and kurtosis.
#
# Source: statsmodels package, jarque_bera function
# Reference: Jarque & Bera (1987), "A test for normality of observations and
#            regression residuals"
#
# Usage:
#   python test_jarque_bera.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR]
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
from statsmodels.stats.stattools import jarque_bera


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
            # For string/categorical data, use integer encoding
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Replace NaN (missing values from coercing) with mode
            if data[col].isnull().any():
                mode_val = data[col].mode()[0]
                data[col].fillna(mode_val, inplace=True)
                print(f"  {col}: {len(data[col].unique())} unique values -> integer encoding (missing filled with mode: {mode_val})")
            else:
                print(f"  {col}: {len(data[col].unique())} unique values -> integer encoding")
        else:
            # Already numeric or other type
            pass

    return non_numeric_cols


def main():
    parser = argparse.ArgumentParser(
        description="Jarque-Bera Test - Check normality of residuals using statsmodels"
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

    # Run Jarque-Bera test on residuals
    # jarque_bera returns (JB_statistic, p_value, skewness, kurtosis)
    jb_result = jarque_bera(model.resid)

    # Print results
    print("Jarque-Bera Test (Python - statsmodels)")
    print("=" * 40)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"JB-statistic: {jb_result[0]}")
    print(f"p-value: {jb_result[1]}")
    print(f"Skewness: {jb_result[2]}")
    print(f"Kurtosis: {jb_result[3]}")
    print(f"Passed: {jb_result[1] > 0.05}")
    print()

    # Prepare output
    output = {
        "test_name": "Jarque-Bera Test (Python - statsmodels)",
        "dataset": dataset_name,
        "formula": formula_str,
        "statistic": float(jb_result[0]),
        "p_value": float(jb_result[1]),
        "passed": bool(jb_result[1] > 0.05),
        "skewness": float(jb_result[2]),
        "kurtosis": float(jb_result[3]),
        "description": "Tests for normality of residuals by examining skewness and kurtosis. Uses statsmodels.stats.stattools.jarque_bera."
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_jarque_bera.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_jarque_bera.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
