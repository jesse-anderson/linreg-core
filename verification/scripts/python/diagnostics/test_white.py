# ============================================================================
# White Test Reference Implementation (Python)
# ============================================================================
# This script generates reference values for the White test using
# statsmodels.stats.diagnostic.het_white. The White test is a more general
# test for heteroscedasticity that uses squares and cross-products of
# predictors.
#
# Source: statsmodels package, het_white function
# Reference: White (1980), "A Heteroskedasticity-Consistent Covariance
#            Matrix Estimator and a Direct Test for Heteroskedasticity"
#
# Usage:
#   python test_white.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR]
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
from statsmodels.stats.diagnostic import het_white


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
        description="White Test - Check heteroscedasticity using statsmodels"
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

    # Run White test
    # het_white returns (LM_statistic, p_value, f_statistic, f_p_value)
    white_result = het_white(model.resid, model.model.exog)

    # Print results
    print("White Test (Python - statsmodels)")
    print("=" * 35)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"LM-statistic: {white_result[0]}")
    print(f"p-value: {white_result[1]}")
    print(f"Passed: {white_result[1] > 0.05}")
    print()

    # Prepare output
    output = {
        "test_name": "White Test (Python - statsmodels)",
        "dataset": dataset_name,
        "formula": formula_str,
        "statistic": float(white_result[0]),
        "p_value": float(white_result[1]),
        "passed": bool(white_result[1] > 0.05),
        "f_statistic": float(white_result[2]),
        "f_p_value": float(white_result[3]),
        "description": "Tests for heteroscedasticity using squares and cross-products of predictors. Uses statsmodels.stats.diagnostic.het_white."
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_white.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_white.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
