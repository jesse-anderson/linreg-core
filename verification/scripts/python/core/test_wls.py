# ============================================================================
# WLS Regression Test Reference Implementation (Python with statsmodels)
# ============================================================================
# This script generates reference values for Weighted Least Squares regression
# using Python's statsmodels library. The test validates that the Rust
# implementation matches statsmodels' behavior for coefficient estimates and
# statistics.
#
# Source: statsmodels.regression.linear_model.WLS
# Reference: "Applied Linear Statistical Models" (Kutner et al.)
#
# Usage:
#   python test_wls.py --csv [csv_path] --output-dir [output_dir]
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
    """Convert categorical columns to numeric representations (0-based factor codes)."""
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    if len(non_numeric_cols) > 0:
        print(f"INFO: Dataset '{dataset_name}' contains non-numeric columns: "
              f"{', '.join(non_numeric_cols)}")
        print("Converting categorical variables to numeric factor codes...")

        for col in non_numeric_cols:
            # First try to convert to numeric directly (handles numeric strings)
            numeric_vals = pd.to_numeric(data[col], errors='coerce')
            if not numeric_vals.isna().any():
                # All values converted successfully - use numeric values
                data[col] = numeric_vals
            else:
                # Contains non-numeric strings - convert to factor codes (0-based)
                codes, _ = pd.factorize(data[col])
                data[col] = codes.astype(float)
                print(f"  Column '{col}' encoded as 0-based factor codes")

    return data


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate WLS regression reference values"
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

    # Validate CSV path
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Extract dataset name from filename
    dataset_name = csv_path.stem

    print(f"Running WLS regression test on dataset: {dataset_name}")

    # Load data
    data = pd.read_csv(csv_path)
    data = convert_categorical_to_numeric(data, dataset_name)

    # Extract response (first column) and predictors (remaining columns)
    response_col = data.columns[0]
    predictor_cols = data.columns[1:]

    # Prepare data
    y = data[response_col].values
    X = data[predictor_cols].values

    # Add intercept
    X = sm.add_constant(X)

    # Create weights (equal weights for validation - WLS with w=1 is equivalent to OLS)
    weights = np.ones(len(y))

    # Fit WLS model
    model = sm.WLS(y, X, weights=weights)
    results = model.fit()

    # Extract coefficients
    n = len(y)
    k = len(predictor_cols)
    df_residual = int(results.df_resid)
    df_model = k

    # Variable names (including intercept)
    variable_names = ['Intercept'] + list(predictor_cols)

    # Coefficients and statistics
    coefficients = results.params.tolist()
    std_errors = results.bse.tolist()
    t_stats = results.tvalues.tolist()
    p_values = results.pvalues.tolist()

    # Model fit statistics
    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj
    f_statistic = results.fvalue
    f_p_value = results.f_pvalue

    # Residuals and error metrics
    residuals_arr = results.resid
    residuals = residuals_arr.tolist()
    fitted_values = results.fittedvalues.tolist()
    mse = results.mse_resid
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals_arr))
    residual_std_error = np.sqrt(np.sum(weights * residuals_arr**2) / df_residual)

    # Model selection criteria
    log_likelihood = results.llf
    aic_val = results.aic
    bic_val = results.bic

    # Confidence intervals (95%)
    ci = results.conf_int(alpha=0.05)
    conf_int_lower = ci[:, 0].tolist()
    conf_int_upper = ci[:, 1].tolist()

    # Build formula string
    formula_str = f"{response_col} ~ {' + '.join(predictor_cols)}"

    # Build result object
    result = {
        "test": "wls",
        "method": "statsmodels",
        "dataset": dataset_name,
        "formula": formula_str,
        "n": n,
        "k": k,
        "df_residual": df_residual,
        "df_model": df_model,
        "variable_names": variable_names,
        "coefficients": coefficients,
        "std_errors": std_errors,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "residual_std_error": residual_std_error,
        "log_likelihood": log_likelihood,
        "aic": aic_val,
        "bic": bic_val,
        "conf_int_lower": conf_int_lower,
        "conf_int_upper": conf_int_upper,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "weights": weights.tolist()
    }

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    output_file = output_dir / f"{dataset_name}_wls.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)

    print(f"Wrote: {output_file.absolute()}")


if __name__ == "__main__":
    main()
