"""
Extended Datasets Validation - Python Reference Implementation
===========================================================

This script validates the Rust OLS implementation against Python (statsmodels)
using synthetic and real-world datasets designed to test edge cases and stress conditions.

Output: verification/datasets/results/python/*.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Set up output directory (relative to script location)
# Output: verification/results/python/
output_dir = os.path.join(script_dir, "..", "..", "..", "results", "python")
os.makedirs(output_dir, exist_ok=True)

# Set path to datasets (relative to script location)
datasets_dir = os.path.join(script_dir, "..", "..", "..", "datasets", "csv")

def run_regression(dataset_name, df, y_var, x_vars=None):
    """Run OLS regression and save results to JSON."""
    print(f"\n=== {dataset_name} ===")

    # Prepare data
    y = df[y_var]
    if x_vars is None:
        X = df.drop(columns=[y_var])
    else:
        X = df[x_vars]

    # Add intercept
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X).fit()

    # Extract results
    n = len(y)
    k = len(x_vars) if x_vars else len(X.columns) - 1
    df_residual = model.df_resid

    coefficients = model.params.tolist()
    std_errors = model.bse.tolist()
    t_stats = model.tvalues.tolist()
    p_values = model.pvalues.tolist()

    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_statistic = model.fvalue
    f_p_value = model.f_pvalue

    # Confidence intervals
    ci = model.conf_int(alpha=0.05)
    ci_lower = ci[:, 0].tolist()
    ci_upper = ci[:, 1].tolist()

    # VIF for multiple regression
    vif_results = None
    if k >= 2:
        vif_values = [variance_inflation_factor(X.values, i)
                      for i in range(1, len(X.columns))]
        vif_results = {
            "variables": list(X.columns[1:]),
            "vif": vif_values
        }

    # Output JSON
    json_output = json.dumps({
        "dataset": dataset_name,
        "n": n,
        "k": k,
        "df_residual": df_residual,
        "coefficients": coefficients,
        "std_errors": std_errors,
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "vif": vif_results
    }, indent=2)

    # Write to file
    safe_name = dataset_name.lower().replace(" ", "_")
    output_file = os.path.join(output_dir, f"{safe_name}.json")
    with open(output_file, "w") as f:
        f.write(json_output)
    print(f"  Wrote: {os.path.basename(output_file)}")

    # Print summary
    print(f"    n = {n}, k = {k}")
    print(f"    R² = {r_squared:.6f}, Adj R² = {adj_r_squared:.6f}")
    print(f"    F({k}, {df_residual}) = {f_statistic:.4f}, p = {f_p_value:.6f}")
    if vif_results:
        vif_str = ", ".join([f"{v:.2f}" for v in vif_results["vif"]])
        print(f"    VIF: {vif_str}")

    return {
        "dataset": dataset_name,
        "n": n,
        "k": k,
        "r_squared": r_squared,
        "vif": vif_results["vif"] if vif_results else None
    }


def main():
    """Run validation on all datasets."""
    print("=" * 60)
    print("Extended Datasets Validation - Python Reference")
    print("=" * 60)

    results = []

    # --- Synthetic Datasets ---

    # 1. Simple Linear
    synthetic_simple = pd.read_csv(os.path.join(datasets_dir, "synthetic_simple_linear.csv"))
    results.append(run_regression("Synthetic Simple Linear", synthetic_simple, "y", ["x"]))

    # 2. Multiple Regression
    synthetic_multiple = pd.read_csv(os.path.join(datasets_dir, "synthetic_multiple.csv"))
    results.append(run_regression("Synthetic Multiple", synthetic_multiple, "y", ["x1", "x2", "x3"]))

    # 3. Collinear
    synthetic_collinear = pd.read_csv(os.path.join(datasets_dir, "synthetic_collinear.csv"))
    print("\n--- Testing Collinear Dataset (expecting high VIF or singular matrix) ---")
    try:
        results.append(run_regression("Synthetic Collinear", synthetic_collinear, "y", ["x1", "x2", "x3"]))
    except np.linalg.LinAlgError as e:
        print(f"  Expected error: {e}")

    # 4. Heteroscedastic
    synthetic_hetero = pd.read_csv(os.path.join(datasets_dir, "synthetic_heteroscedastic.csv"))
    results.append(run_regression("Synthetic Heteroscedastic", synthetic_hetero, "y", ["x"]))

    # 5. Nonlinear
    synthetic_nonlinear = pd.read_csv(os.path.join(datasets_dir, "synthetic_nonlinear.csv"))
    results.append(run_regression("Synthetic Nonlinear", synthetic_nonlinear, "y", ["x"]))

    # 6. Nonnormal
    synthetic_nonnormal = pd.read_csv(os.path.join(datasets_dir, "synthetic_nonnormal.csv"))
    results.append(run_regression("Synthetic Nonnormal", synthetic_nonnormal, "y", ["x"]))

    # 7. Autocorrelated
    synthetic_auto = pd.read_csv(os.path.join(datasets_dir, "synthetic_autocorrelated.csv"))
    results.append(run_regression("Synthetic Autocorrelated", synthetic_auto, "y", ["x"]))

    # 8. High VIF
    synthetic_high_vif = pd.read_csv(os.path.join(datasets_dir, "synthetic_high_vif.csv"))
    results.append(run_regression("Synthetic High VIF", synthetic_high_vif, "y", None))

    # 9. Outliers
    synthetic_outliers = pd.read_csv(os.path.join(datasets_dir, "synthetic_outliers.csv"))
    results.append(run_regression("Synthetic Outliers", synthetic_outliers, "y", None))

    # 10. Small Sample
    synthetic_small = pd.read_csv(os.path.join(datasets_dir, "synthetic_small.csv"))
    results.append(run_regression("Synthetic Small", synthetic_small, "y", None))

    # --- Real-World Datasets ---

    # Longley
    longley = pd.read_csv(os.path.join(datasets_dir, "longley.csv"))
    longley_y = "Employed"
    longley_x = [col for col in longley.columns if col not in [longley_y, "Year", "Unnamed: 0"]]
    results.append(run_regression("Longley", longley, longley_y, longley_x))

    # Mtcars
    mtcars = pd.read_csv(os.path.join(datasets_dir, "mtcars.csv"))
    results.append(run_regression("Mtcars", mtcars, "mpg", ["cyl", "disp", "hp", "wt", "qsec"]))

    # Bodyfat
    bodyfat = pd.read_csv(os.path.join(datasets_dir, "bodyfat.csv"))
    bodyfat_y = bodyfat.columns[0]
    results.append(run_regression("Bodyfat", bodyfat, bodyfat_y, None))

    # Prostate
    prostate = pd.read_csv(os.path.join(datasets_dir, "prostate.csv"))
    prostate_y = prostate.columns[-1]
    results.append(run_regression("Prostate", prostate, prostate_y, None))

    print("\n" + "=" * 60)
    print("Extended validation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
