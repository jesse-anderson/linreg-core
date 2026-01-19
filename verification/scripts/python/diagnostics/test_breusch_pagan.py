# ============================================================================
# Breusch-Pagan Test Reference Implementation (Python)
# ============================================================================
# This script generates reference values for the Breusch-Pagan test using
# statsmodels.stats.diagnostic.het_breuschpagan. The Breusch-Pagan test checks
# for heteroscedasticity by regressing squared residuals on the original
# predictors.
#
# Source: statsmodels package, het_breuschpagan function
# Reference: Breusch & Pagan (1979), "A Simple Test for Heteroscedasticity
#            and Random Coefficient Variation"
#
# Usage:
#   python test_breusch_pagan.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR]
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
from statsmodels.stats.diagnostic import het_breuschpagan


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
        description="Breusch-Pagan Test - Check heteroscedasticity using statsmodels"
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

    # Run Breusch-Pagan test
    # het_breuschpagan returns (LM_statistic, p_value, f_statistic, f_p_value)
    bp_result = het_breuschpagan(model.resid, model.model.exog)

    # Print results
    print("Breusch-Pagan Test (Python - statsmodels)")
    print("=" * 42)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"LM-statistic: {bp_result[0]}")
    print(f"p-value: {bp_result[1]}")
    print(f"Passed: {bp_result[1] > 0.05}")
    print()

    # Prepare output
    output = {
        "test_name": "Breusch-Pagan Test (Python - statsmodels)",
        "dataset": dataset_name,
        "formula": formula_str,
        "statistic": float(bp_result[0]),
        "p_value": float(bp_result[1]),
        "passed": bool(bp_result[1] > 0.05),
        "f_statistic": float(bp_result[2]),
        "f_p_value": float(bp_result[3]),
        "description": "Tests for heteroscedasticity by regressing squared residuals on predictors. Uses statsmodels.stats.diagnostic.het_breuschpagan."
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_breusch_pagan.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_breusch_pagan.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
