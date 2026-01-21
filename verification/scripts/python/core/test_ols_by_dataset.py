# ============================================================================
# OLS Regression Test Reference Implementation (Python with statsmodels)
# ============================================================================
# This script generates reference values for OLS regression using Python's
# statsmodels library. The test validates that the Rust implementation matches
# statsmodels' behavior for coefficient estimates and statistics.
#
# Source: statsmodels.regression.linear_model.OLS
# Reference: "Applied Linear Statistical Models" (Kutner et al.)
#
# Usage:
#   python test_ols_by_dataset.py [csv_path] [output_dir]
#   Args:
#     csv_path    - Path to CSV file (first col = response, rest = predictors)
#                   Default: ../../datasets/csv/mtcars.csv
#     output_dir  - Path to output directory
#                   Default: ../../results/python
# ============================================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
        print("Converting categorical variables to numeric representations...")

        for col in non_numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if data[col].isna().any():
                mode_val = data[col].mode()
                if len(mode_val) > 0:
                    data[col].fillna(mode_val[0], inplace=True)

    return data


def main():
    # Parse command line arguments
    args = sys.argv[1:]

    # Set defaults
    default_csv = "../../datasets/csv/mtcars.csv"
    default_output = "../../results/python"

    csv_path_input = args[0] if len(args) >= 1 else default_csv
    output_dir_input = args[1] if len(args) >= 2 else default_output

    # Resolve paths - use absolute paths directly, resolve relative paths from cwd
    csv_path = Path(csv_path_input) if Path(csv_path_input).is_absolute() else Path.cwd() / csv_path_input
    output_dir = Path(output_dir_input) if Path(output_dir_input).is_absolute() else Path.cwd() / output_dir_input

    # Validate CSV path
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    # Extract dataset name from filename
    dataset_name = csv_path.stem

    print(f"Running OLS regression test on dataset: {dataset_name}")

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

    # Fit OLS model
    model = sm.OLS(y, X)
    results = model.fit()

    # Extract coefficients
    n = len(y)
    k = len(predictor_cols)
    df_residual = n - k - 1

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

    # Residuals and MSE
    residuals = results.resid.tolist()
    mse = results.mse_resid
    std_error = np.sqrt(mse)

    # Confidence intervals (95%)
    ci = results.conf_int(alpha=0.05)
    conf_int_lower = ci[:, 0].tolist()
    conf_int_upper = ci[:, 1].tolist()

    # VIF calculation (exclude intercept)
    vif = []
    if k > 1:
        try:
            for i, var_name in enumerate(predictor_cols):
                vif_val = variance_inflation_factor(X, i + 1)  # +1 to skip intercept
                rsq = 1.0 - 1.0/vif_val if vif_val > 0 else 0.0
                vif.append({
                    "variable": var_name,
                    "vif": vif_val,
                    "rsquared": rsq
                })
        except Exception as e:
            # VIF calculation failed (e.g., singular matrix)
            vif = []

    # Build formula string
    formula_str = f"{response_col} ~ {' + '.join(predictor_cols)}"

    # Build result object
    result = {
        "test": "ols",
        "method": "statsmodels",
        "dataset": dataset_name,
        "formula": formula_str,
        "n": n,
        "k": k,
        "df_residual": int(df_residual),
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
        "std_error": std_error,
        "conf_int_lower": conf_int_lower,
        "conf_int_upper": conf_int_upper,
        "residuals": residuals,
        "vif": vif
    }

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    output_file = output_dir / f"{dataset_name}_ols.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, cls=NumpyEncoder, indent=2)

    print(f"Wrote: {output_file.absolute()}")


if __name__ == "__main__":
    main()
