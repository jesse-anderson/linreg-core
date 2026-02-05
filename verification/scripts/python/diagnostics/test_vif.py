# ============================================================================
# VIF (Variance Inflation Factor) Reference Implementation (Python)
# ============================================================================
# This script generates reference values for VIF using statsmodels
#
# The VIF measures how much the variance of a regression coefficient is
# inflated due to multicollinearity among predictor variables.
#
# Source: statsmodels package, variance_inflation_factor function
# Reference: statsmodels.stats.outliers_influence.variance_inflation_factor
#
# Usage:
#   python test_vif.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR]
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
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
        description="VIF Test - Calculate Variance Inflation Factor using statsmodels"
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

    # Calculate VIF using statsmodels
    # variance_inflation_factor expects the exog matrix, not the fitted model
    vif_result = [
        variance_inflation_factor(X.values, i)
        for i in range(1, len(predictor_cols) + 1)  # Skip intercept (column 0)
    ]

    # Print results
    print("VIF Test (Python - statsmodels)")
    print("=" * 40)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"Number of predictors: {len(vif_result)}")
    print()

    # Print each variable's VIF
    print("VIF Results:")
    for i, name in enumerate(predictor_cols):
        print(f"  {name}: VIF = {vif_result[i]:.6f}, R² = {1 - 1/vif_result[i]:.6f}")

    print(f"\nMax VIF: {max(vif_result):.6f}\n")

    # Interpretation
    max_vif = max(vif_result)
    if max_vif > 10:
        interpretation = "Severe multicollinearity detected (VIF > 10)"
    elif max_vif > 5:
        interpretation = "Moderate multicollinearity detected (VIF > 5)"
    else:
        interpretation = "Low multicollinearity (VIF <= 5)"
    print(f"Interpretation: {interpretation}\n")

    # Prepare output
    output = {
        "test_name": "VIF Test (Python - statsmodels)",
        "dataset": dataset_name,
        "formula": formula_str,
        "vif_values": vif_result,
        "variable_names": list(predictor_cols),
        "max_vif": float(max_vif),
        "interpretation": interpretation,
        "description": "Variance Inflation Factor measures multicollinearity among predictors. Uses statsmodels.stats.outliers_influence.variance_inflation_factor."
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_vif.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_vif.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
