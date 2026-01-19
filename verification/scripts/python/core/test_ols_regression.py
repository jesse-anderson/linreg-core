# ============================================================================
# Linear Regression Tests - Python Reference Implementation
# ============================================================================
# This script runs the same test cases as the Rust WASM implementation
# to verify correctness. Output can be compared for validation.
#
# IMPORTANT: This script uses NATIVE Python library functions for all tests.
# Manual implementations are NOT used as they would not validate correctness.
#
# Required packages:
#   pip install numpy pandas scipy statsmodels
# ============================================================================ 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_rainbow, linear_harvey_collier
from statsmodels.stats.stattools import jarque_bera, durbin_watson
import json
import os

print("=" * 60)
print("Linear Regression Tests - Python Reference Implementation")
print("=" * 60)

# Helper for JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

results_json = {}

# ============================================================================ 
# TEST 2: Multiple Linear Regression (Housing Prices)
# ============================================================================ 
print("\n--- Test 2: Multiple Linear Regression (Housing Prices) ---\n")

df_housing = pd.DataFrame({
    'Price': [245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1,
               445.8, 167.9, 367.4, 289.6, 198.2, 478.5, 256.3, 334.7, 178.5, 398.9,
               223.4, 312.5, 156.8, 423.7, 267.9],
    'Square_Feet': [1200, 1800, 950, 2400, 1450, 2000, 1100, 2800, 1350, 1650,
                     2200, 900, 1950, 1500, 1050, 2600, 1300, 1850, 1000, 2100,
                     1250, 1700, 850, 2350, 1400],
    'Bedrooms': [3, 4, 2, 4, 3, 4, 2, 5, 3, 3,
                 4, 2, 4, 3, 2, 5, 3, 4, 2, 4,
                 3, 3, 2, 4, 3],
    'Age': [15, 10, 25, 5, 8, 12, 20, 2, 18, 7,
           3, 30, 6, 14, 22, 1, 16, 9, 28, 4,
           19, 11, 35, 3, 13]
})

X_housing = df_housing[['Square_Feet', 'Bedrooms', 'Age']]
X_housing = sm.add_constant(X_housing) # Adds 'const' column for intercept
y_housing = df_housing['Price']

model_housing = OLS(y_housing, X_housing).fit()

# Extract VIF
vif_data = []
for i in range(X_housing.shape[1]):
    vif_val = variance_inflation_factor(X_housing.values, i)
    if not np.isinf(vif_val) and vif_val > 0:
        rsq = 1.0 - 1.0/vif_val
    else:
        rsq = 0.0 # Constant or perfect correlation
        
    vif_data.append({
        "variable": X_housing.columns[i],
        "vif": vif_val,
        "rsquared": rsq
    })

# Rainbow Test (statsmodels.stats.diagnostic.linear_rainbow)
rainbow_result = linear_rainbow(model_housing)
rainbow_json = {
    "statistic": rainbow_result[0],
    "p_value": rainbow_result[1],
    "passed": bool(rainbow_result[1] > 0.05)
}

# Harvey-Collier Test (statsmodels.stats.diagnostic.linear_harvey_collier)
hc_result = linear_harvey_collier(model_housing)
hc_json = {
    "statistic": hc_result[0],
    "p_value": hc_result[1],
    "passed": bool(hc_result[1] > 0.05)
}

# Breusch-Pagan Test (statsmodels.stats.diagnostic.het_breuschpagan)
bp_result = het_breuschpagan(model_housing.resid, X_housing)
bp_json = {
    "statistic": bp_result[0],
    "p_value": bp_result[1],
    "passed": bool(bp_result[1] > 0.05)
}

# White Test (statsmodels.stats.diagnostic.het_white)
white_result = het_white(model_housing.resid, X_housing)
white_json = {
    "statistic": white_result[0],
    "p_value": white_result[1],
    "passed": bool(white_result[1] > 0.05)
}

# Jarque-Bera Test (statsmodels.stats.stattools.jarque_bera)
jb_result = jarque_bera(model_housing.resid)
jb_json = {
    "statistic": jb_result[0],
    "p_value": jb_result[1],
    "passed": bool(jb_result[1] > 0.05)
}

# Durbin-Watson Test (statsmodels.stats.stattools.durbin_watson)
dw_stat = durbin_watson(model_housing.resid)
dw_json = {
    "statistic": dw_stat,
    "p_value": 0.0,  # Statistic only - no p-value from statsmodels
    "passed": bool(1.5 < dw_stat < 2.5)
}

# Confidence Intervals (95%)
ci = model_housing.conf_int(alpha=0.05)
ci_lower = ci[0].tolist()
ci_upper = ci[1].tolist()

influence = model_housing.get_influence()
standardized_residuals = influence.resid_studentized_internal

# Construct JSON object matching Rust struct structure
housing_result = {
    "coefficients": model_housing.params.tolist(),
    "std_errors": model_housing.bse.tolist(),
    "t_stats": model_housing.tvalues.tolist(),
    "p_values": model_housing.pvalues.tolist(),
    "r_squared": model_housing.rsquared,
    "adj_r_squared": model_housing.rsquared_adj,
    "f_statistic": model_housing.fvalue,
    "f_p_value": model_housing.f_pvalue,
    "mse": model_housing.mse_resid,
    "std_error": np.sqrt(model_housing.mse_resid),
    "conf_int_lower": ci_lower,
    "conf_int_upper": ci_upper,
    "residuals": model_housing.resid.tolist(),
    "standardized_residuals": standardized_residuals.tolist(),
    "vif": [v for v in vif_data if v['variable'] != 'const'],
    "rainbow": rainbow_json,
    "harvey_collier": hc_json,
    "breusch_pagan": bp_json,
    "white": white_json,
    "jarque_bera": jb_json,
    "durbin_watson": dw_json,
    "n": int(model_housing.nobs),
    "k": int(model_housing.df_model),
    "df": int(model_housing.df_resid)
}

results_json["housing_regression"] = housing_result

print(f"R²: {model_housing.rsquared:.6f}")
print(f"MSE: {model_housing.mse_resid:.6f}")
print(f"Rainbow F-stat: {rainbow_result[0]:.4f}, p={rainbow_result[1]:.4f}")
print(f"Harvey-Collier t-stat: {hc_result[0]:.4f}, p={hc_result[1]:.4f}")
print(f"Breusch-Pagan LM: {bp_result[0]:.4f}, p={bp_result[1]:.4f}")
print(f"White LM: {white_result[0]:.4f}, p={white_result[1]:.4f}")
print(f"Jarque-Bera: {jb_result[0]:.4f}, p={jb_result[1]:.4f}")
print(f"Durbin-Watson: {dw_stat:.4f}")

print("\n" + "=" * 60)
print("Python Library Functions Used for Validation")
print("=" * 60)
print("Core Regression:")
print("  statsmodels.regression.linear_model.OLS")
print("\nDiagnostic Tests:")
print("  statsmodels.stats.diagnostic.linear_rainbow")
print("  statsmodels.stats.diagnostic.linear_harvey_collier")
print("  statsmodels.stats.diagnostic.het_breuschpagan")
print("  statsmodels.stats.diagnostic.het_white")
print("  statsmodels.stats.stattools.jarque_bera")
print("  statsmodels.stats.stattools.durbin_watson")
print("\nMulticollinearity:")
print("  statsmodels.stats.outliers_influence.variance_inflation_factor")
print("=" * 60)

# ============================================================================
# Write JSON - ALWAYS OVERWRITE to ensure fresh validation data
# Output to verification/results/python/
# ============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, '..', '..', '..', 'results', 'python', 'Python_results.json')
with open(output_path, 'w') as f:
    json.dump(results_json, f, cls=NumpyEncoder, indent=2)

print(f"\n[SUCCESS] Wrote fresh validation data to: {output_path}")