# ============================================================================
# Durbin-Watson Test Reference Implementation (Python)
# ============================================================================
# This script generates reference values for the Durbin-Watson test using
# statsmodels.stats.stattools.durbin_watson. The Durbin-Watson test checks
# for autocorrelation in the residuals.
#
# Source: statsmodels package, durbin_watson function
# Reference: Durbin & Watson (1950), "Testing for Serial Correlation in
#            Least Squares Regression: I", Biometrika, Vol. 37, pp. 409-428
#            (1951), "Testing for Serial Correlation in Least Squares
#            Regression: II", Biometrika, Vol. 38, pp. 159-178
#
# Usage:
#   python test_durbin_watson.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR]
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
from statsmodels.stats.stattools import durbin_watson


def convert_categorical_to_numeric(data, dataset_name):
    """Convert categorical variables to numeric representations."""
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    if not non_numeric_cols:
        return []

    print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: {non_numeric_cols}")
    print(f"Converting categorical variables to numeric representations...")

    # Process each non-numeric column
    for col in non_numeric_cols:
        # Use factorize for reliable categorical encoding
        data[col], uniques = pd.factorize(data[col])
        print(f"  {col}: {len(uniques)} unique values -> integer level encoding")

    return non_numeric_cols


def main():
    parser = argparse.ArgumentParser(
        description="Durbin-Watson Test - Check autocorrelation using statsmodels"
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

    # Run Durbin-Watson test
    # Returns DW statistic (no p-value from statsmodels)
    dw_stat = durbin_watson(model.resid)

    # Interpret result (values near 2 = no autocorrelation)
    if dw_stat > 2:
        interpretation = "No positive autocorrelation"
    elif dw_stat < 2:
        interpretation = "Possible positive autocorrelation"
    else:
        interpretation = "Inconclusive"

    # Print results
    print("Durbin-Watson Test (Python - statsmodels)")
    print("=" * 42)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"DW-statistic: {dw_stat}")
    print(f"Interpretation: {interpretation}")
    print(f"Passed (1.5 < d < 2.5): {1.5 < dw_stat < 2.5}")
    print()

    # Prepare output
    output = {
        "test_name": "Durbin-Watson Test (Python - statsmodels)",
        "dataset": dataset_name,
        "formula": formula_str,
        "statistic": float(dw_stat),
        "p_value": None,  # statsmodels doesn't provide p-value for DW test
        "passed": bool(1.5 < dw_stat < 2.5),
        "interpretation": interpretation,
        "description": "Tests for autocorrelation in residuals. Values near 2 indicate no autocorrelation, values near 0 suggest positive autocorrelation, and values near 4 suggest negative autocorrelation. Uses statsmodels.stats.stattools.durbin_watson."
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_durbin_watson.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_durbin_watson.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
