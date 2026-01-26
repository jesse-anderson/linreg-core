#!/usr/bin/env python3
"""
 ============================================================================
 Master Reference Generation Script (Python)
 ============================================================================

 This script generates reference outputs for all test datasets using Python's
 native statistical libraries. The outputs are saved as JSON files for
 cross-validation with the Rust WASM implementation.

 Usage: python generate_all_references.py

 Output: JSON files in verification/datasets/references/expanded/
 ============================================================================
"""

import json
import os
import sys
from pathlib import Path

# Check for required packages
try:
    import numpy as np
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import (
        linear_rainbow, linear_harvey_collier, het_breuschpagan,
        het_white, acorr_ljungbox, acorr_breusch_godfrey
    )
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install numpy scipy statsmodels")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("verification/datasets/references/expanded")
ALPHA = 0.05

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Dataset Definitions
# ============================================================================

datasets = {}

# 1. Housing dataset (synthetic)
datasets['housing'] = {
    'y': np.array([
        245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1,
        445.8, 167.9, 367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9,
        223.4, 312.5, 156.8, 423.7, 267.9
    ]),
    'x_vars': {
        'Square_Feet': np.array([
            1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0,
            2200.0, 900.0, 1950.0, 1500.0, 1050.0, 2600.0, 1300.0, 1850.0, 1000.0, 2100.0,
            1250.0, 1700.0, 850.0, 2350.0, 1400.0
        ]),
        'Bedrooms': np.array([
            3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0,
            4.0, 2.0, 4.0, 3.0, 2.0, 5.0, 3.0, 4.0, 2.0, 4.0,
            3.0, 3.0, 2.0, 4.0, 3.0
        ]),
        'Age': np.array([
            15.0, 10.0, 25.0, 5.0, 8.0, 12.0, 20.0, 2.0, 18.0, 7.0,
            3.0, 30.0, 6.0, 14.0, 22.0, 1.0, 16.0, 9.0, 28.0, 4.0,
            19.0, 11.0, 35.0, 3.0, 13.0
        ])
    },
    'variable_names': ['Intercept', 'Square_Feet', 'Bedrooms', 'Age']
}

# 2. Perfect fit dataset (synthetic)
datasets['perfect_fit'] = {
    'y': np.array([5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0, 37.0, 41.0]),
    'x_vars': {
        'x1': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        'x2': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    },
    'variable_names': ['Intercept', 'x1', 'x2']
}

# 3. Single predictor dataset
datasets['single_predictor'] = {
    'y': np.array([3.1, 5.0, 6.9, 9.0, 11.1, 12.8, 15.0, 17.1, 18.9, 21.0]),
    'x_vars': {
        'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    },
    'variable_names': ['Intercept', 'x']
}

# 4. High multicollinearity dataset
datasets['high_multicollinearity'] = {
    'y': np.array([5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0, 37.0, 41.0]),
    'x_vars': {
        'x1': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        'x2': np.array([2.02, 4.01, 5.99, 8.01, 9.98, 12.02, 13.99, 16.01, 17.98, 20.02]),
        'x3': np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    },
    'variable_names': ['Intercept', 'x1', 'x2', 'x3']
}

# 5. Small n dataset (boundary case)
datasets['small_n'] = {
    'y': np.array([3.1, 5.0, 6.9, 9.0, 11.1]),
    'x_vars': {
        'x1': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'x2': np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    },
    'variable_names': ['Intercept', 'x1', 'x2']
}

# ============================================================================
# Helper Functions
# ============================================================================

def jarque_bera_test(residuals):
    """
    Jarque-Bera test for normality using scipy
    """
    from scipy.stats import jarque_bera as scipy_jb

    stat, p_value = scipy_jb(residuals)
    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'passed': p_value > ALPHA
    }

def durbin_watson_statistic(residuals):
    """
    Durbin-Watson statistic
    """
    diff = np.diff(residuals)
    return float(np.sum(diff ** 2) / np.sum(residuals ** 2))

# ============================================================================
# Main Validation Function
# ============================================================================

def generate_reference(dataset_name, dataset):
    """Generate reference output for a single dataset"""
    print(f"\n=== Generating Python reference for: {dataset_name} ===")

    y = dataset['y']
    x_vars_dict = dataset['x_vars']
    variable_names = dataset['variable_names']

    # Create X matrix
    x_columns = [x_vars_dict[name] for name in list(x_vars_dict.keys())]
    X = np.column_stack(x_columns)
    X = add_constant(X)  # Add intercept

    n = len(y)
    k = len(x_vars_dict)
    df_residual = n - k - 1

    # Fit model
    model = sm.OLS(y, X).fit()

    # Basic results
    coefs = model.params.tolist()
    std_errors = model.bse.tolist()
    t_stats = model.tvalues.tolist()
    p_values = model.pvalues.tolist()

    # Confidence intervals
    conf_int = model.conf_int(alpha=ALPHA).tolist()
    conf_int_lower = [row[0] for row in conf_int]
    conf_int_upper = [row[1] for row in conf_int]

    # Model fit statistics
    r_squared = float(model.rsquared)
    adj_r_squared = float(model.rsquared_adj)
    f_statistic = float(model.fvalue)
    f_p_value = float(model.f_pvalue)

    # Residuals
    residuals_val = model.resid.tolist()
    mse = float(model.mse_resid)
    std_error = float(np.sqrt(mse))

    # Standardized residuals
    influence = model.get_influence()
    standardized_residuals = influence.resid_studentized_internal.tolist()
    leverage = influence.hat_matrix_diag.tolist()

    # Predictions
    predictions = model.fittedvalues.tolist()

    # VIF
    vif_results = []
    if k > 1:
        try:
            for i, name in enumerate(list(x_vars_dict.keys())):
                vif_value = variance_inflation_factor(X, i + 1)  # +1 for intercept
                vif_results.append({
                    'variable': name,
                    'vif': float(vif_value),
                    'rsquared': float(1 - 1 / vif_value) if vif_value > 1 else 0.0
                })
        except:
            pass  # VIF calculation may fail for singular matrices

    # Rainbow test (Python statsmodels method)
    try:
        rainbow_stat, rainbow_p = linear_rainbow(model, frac=0.5)
        rainbow_result = {
            'statistic': float(rainbow_stat),
            'p_value': float(rainbow_p),
            'passed': rainbow_p > ALPHA
        }
    except Exception as e:
        rainbow_result = {'statistic': None, 'p_value': None, 'passed': False}

    # Harvey-Collier test
    try:
        hc_stat, hc_p, _ = linear_harvey_collier(model)
        hc_result = {
            'statistic': float(hc_stat),
            'p_value': float(hc_p),
            'passed': hc_p > ALPHA
        }
    except Exception as e:
        hc_result = {'statistic': None, 'p_value': None, 'passed': False}

    # Breusch-Pagan test
    try:
        bp_stat, bp_p, _ = het_breuschpagan(model.resid, model.model.exog)
        bp_result = {
            'statistic': float(bp_stat),
            'p_value': float(bp_p),
            'passed': bp_p > ALPHA
        }
    except Exception as e:
        bp_result = {'statistic': None, 'p_value': None, 'passed': False}

    # White test (Python statsmodels method with interactions)
    try:
        white_stat, white_p, _ = het_white(model.resid, model.model.exog)
        white_result = {
            'statistic': float(white_stat),
            'p_value': float(white_p),
            'passed': white_p > ALPHA
        }
    except Exception as e:
        white_result = {'statistic': None, 'p_value': None, 'passed': False}

    # Jarque-Bera test
    try:
        jb_result = jarque_bera_test(model.resid)
    except Exception as e:
        jb_result = {'statistic': None, 'p_value': None, 'passed': False}

    # Durbin-Watson statistic
    try:
        dw_stat = durbin_watson_statistic(model.resid)
        dw_result = {
            'statistic': dw_stat,
            'p_value': None,  # DW doesn't have a simple p-value
            'passed': True  # Context-dependent
        }
    except Exception as e:
        dw_result = {'statistic': None, 'p_value': None, 'passed': False}

    # Breusch-Godfrey test (order = 1)
    try:
        bg_lm_stat, bg_lm_pval, bg_f_stat, bg_f_pval = acorr_breusch_godfrey(model, order=1)
        bg_result = {
            'statistic': float(bg_lm_stat),
            'p_value': float(bg_lm_pval),
            'passed': bg_lm_pval > ALPHA
        }
    except Exception as e:
        bg_result = {'statistic': None, 'p_value': None, 'passed': False}

    # Build output structure
    output = {
        'dataset_name': dataset_name,
        'coefficients': coefs,
        'std_errors': std_errors,
        't_stats': t_stats,
        'p_values': p_values,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'f_statistic': f_statistic,
        'f_p_value': f_p_value,
        'mse': mse,
        'std_error': std_error,
        'conf_int_lower': conf_int_lower,
        'conf_int_upper': conf_int_upper,
        'residuals': residuals_val,
        'standardized_residuals': standardized_residuals,
        'leverage': leverage,
        'vif': vif_results,
        'rainbow': rainbow_result,
        'harvey_collier': hc_result,
        'breusch_pagan': bp_result,
        'white': white_result,
        'jarque_bera': jb_result,
        'durbin_watson': dw_result,
        'breusch_godfrey': bg_result,
        'n': n,
        'k': k,
        'df': df_residual,
        'variable_names': variable_names
    }

    # Write to JSON
    output_file = OUTPUT_DIR / f"{dataset_name}_python.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  -> Wrote: {output_file}")
    print(f"  R² = {r_squared:.4f}, F = {f_statistic:.2f}")

    return output

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 70)
    print("  Python Reference Generation Script")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of datasets: {len(datasets)}")

    # Generate references for each dataset
    results = {}
    for name, dataset in datasets.items():
        try:
            result = generate_reference(name, dataset)
            results[name] = result
        except Exception as e:
            print(f"ERROR generating {name}: {e}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Successfully generated: {len(results)} / {len(datasets)} datasets")

    # Print table of results
    if results:
        print("\nDataset           R²       F-stat   Rainbow  HC       BP       White    JB       DW       BG")
        print("-" * 90)
        for name, r in results.items():
            def pass_str(res_dict):
                if res_dict.get('p_value') is None:
                    return "N/A"
                return "PASS" if res_dict['p_value'] > ALPHA else "FAIL"

            print(f"{name:<16s}  {r['r_squared']:.4f}   {r['f_statistic']:6.2f}   "
                  f"{pass_str(r['rainbow']):>3s}  {pass_str(r['harvey_collier']):>3s}  "
                  f"{pass_str(r['breusch_pagan']):>3s}  {pass_str(r['white']):>3s}  "
                  f"{pass_str(r['jarque_bera']):>3s}  "
                  f"{f\"{r['durbin_watson']['statistic']:.2f}\" if r['durbin_watson']['statistic'] else 'N/A':>3s}  "
                  f"{pass_str(r['breusch_godfrey']):>3s}")

    print()

if __name__ == "__main__":
    main()
