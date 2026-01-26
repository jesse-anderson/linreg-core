# ============================================================================
# Breusch-Godfrey Test Reference Implementation (Python)
# ============================================================================
# This script generates reference values for the Breusch-Godfrey test using
# statsmodels.stats.diagnostic.acorr_breusch_godfrey. The Breusch-Godfrey test
# checks for higher-order serial correlation in residuals.
#
# Source: statsmodels package, acorr_breusch_godfrey function
# Reference: Breusch, T.S. (1978). Testing for Autocorrelation in Dynamic Linear
#            Models, Australian Economic Papers, 17, 334-355.
#            Godfrey, L.G. (1978). Testing Against General Autoregressive and
#            Moving Average Error Models when the Regressors Include Lagged
#            Dependent Variables, Econometrica, 46, 1293-1301.
#
# Usage:
#   python test_breusch_godfrey.py [--csv CSV_PATH] [--output-dir OUTPUT_DIR] [--order ORDER]
#
#   Args:
#     --csv       Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/synthetic_autocorrelated.csv
#     --output-dir Path to output directory
#                 Default: ../../results/python
#     --order     Order of serial correlation to test (default: 1)
# ============================================================================

import argparse
import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey


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
                mode_vals = data[col].mode()
                if len(mode_vals) > 0:
                    mode_val = mode_vals.iloc[0]
                else:
                    mode_val = 0
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
        description="Breusch-Godfrey Test - Check higher-order serial correlation using statsmodels"
    )
    parser.add_argument(
        "--csv",
        default="../../datasets/csv/synthetic_autocorrelated.csv",
        help="Path to CSV file (first column = response, rest = predictors)"
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/python",
        help="Path to output directory"
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Order of serial correlation to test (default: 1)"
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

    # Run Breusch-Godfrey test
    # Returns (lm_stat, lm_pvalue, f_stat, f_pvalue)
    # acorr_breusch_godfrey(resid, x=None, nlags=1)
    # resid can be a result object (with .resid attribute) or array of residuals
    lm_stat, lm_pval, f_stat, f_pval = acorr_breusch_godfrey(model, nlags=args.order)

    # Degrees of freedom
    n = len(y)
    k = len(predictor_cols) + 1  # including intercept
    df_lm = args.order  # Chi-squared df = order
    df_f_num = args.order  # F df1 = order
    df_f_den = n - k - args.order  # F df2 = n - k - order

    # Determine interpretation based on p-value
    alpha = 0.05
    if lm_pval > alpha:
        interpretation = f"No significant serial correlation detected up to order {args.order}"
    else:
        interpretation = f"Significant serial correlation detected at order <= {args.order}"

    # Print results
    print("Breusch-Godfrey Test (Python - statsmodels)")
    print("=" * 45)
    print(f"Dataset: {dataset_name}")
    print(f"Formula: {formula_str}")
    print(f"Order: {args.order}")
    print()
    print("Chi-squared (LM) statistic:", lm_stat)
    print("  p-value:", lm_pval)
    print("  df:", df_lm)
    print()
    print("F statistic:", f_stat)
    print("  p-value:", f_pval)
    print("  df:", f"{df_f_num}, {df_f_den}")
    print()
    print(f"Interpretation: {interpretation}")
    print(f"Passed (p > 0.05): {lm_pval > alpha}")
    print()

    # Prepare output - we output the Chi-squared version as primary
    # but include F statistic values for reference
    output = {
        "test_name": "Breusch-Godfrey Test (Python - statsmodels)",
        "dataset": dataset_name,
        "formula": formula_str,
        "order": args.order,
        "statistic": float(lm_stat),
        "p_value": float(lm_pval),
        "test_type": "Chisq",
        "df": [float(df_lm)],
        "f_statistic": float(f_stat),
        "f_p_value": float(f_pval),
        "f_df": [float(df_f_num), float(df_f_den)],
        "passed": bool(lm_pval > alpha),
        "interpretation": interpretation,
        "description": f"Tests for serial correlation up to order {args.order}. The LM (Chi-squared) statistic is asymptotically distributed as chi-squared with {args.order} degrees of freedom. The F statistic provides a finite-sample correction. Uses statsmodels.stats.diagnostic.acorr_breusch_godfrey."
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to JSON with naming convention: {dataset}_breusch_godfrey.json
    output_file = os.path.join(args.output_dir, f"{dataset_name}_breusch_godfrey.json")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
