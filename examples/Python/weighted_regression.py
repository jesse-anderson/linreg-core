"""Weighted Least Squares (WLS) regression example.

Run with:
    pip install linreg-core
    python weighted_regression.py

Demonstrates:
- Why OLS fails on heteroscedastic data (variance grows with income)
- How WLS corrects for this using precision weights (1/variance)
- Side-by-side coefficient and SE comparison: OLS vs WLS
- Sanity check: equal weights reproduces OLS exactly
"""

import linreg_core


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║            WEIGHTED LEAST SQUARES (WLS) REGRESSION                  ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # Dataset: income ($k) vs. spending ($k)
    # Variance in spending grows with income — classic heteroscedasticity.
    income   = [20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 75.0, 90.0, 110.0]
    spending = [18.0, 21.0, 25.0, 27.0, 31.0, 38.0, 50.0, 55.0, 72.0, 95.0]

    # Precision weights = 1 / variance = 1 / sd²
    std_devs = [1.0, 1.2, 1.5, 2.0, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0]
    weights  = [1.0 / (s * s) for s in std_devs]

    print("Dataset: 10 households — income vs. spending")
    print("Heteroscedasticity: spending variance grows with income.")
    print()

    # ── 1. Data and Weights ───────────────────────────────────────────────────
    print("━━━ 1. Data and Precision Weights (w = 1/variance) ━━━━━━━━━━━━━━━━━")
    print(f"  {'Income':>8}  {'Spending':>10}  {'Std Dev':>8}  {'Weight':>10}")
    print(f"  {'─'*42}")
    for i in range(10):
        print(f"  {income[i]:>8.1f}  {spending[i]:>10.1f}  "
              f"{std_devs[i]:>8.1f}  {weights[i]:>10.4f}")
    print()
    print("  Low-income obs -> high weight (reliable).")
    print("  High-income obs -> low weight (noisy).")
    print()

    # ── 2. OLS (ignores heteroscedasticity) ───────────────────────────────────
    print("━━━ 2. OLS Regression (ignores heteroscedasticity) ━━━━━━━━━━━━━━━━━")
    names = ["Intercept", "Income"]
    ols = linreg_core.ols_regression(spending, [income], names)
    print(f"  Intercept:  {ols.coefficients[0]:>8.4f}  (SE: {ols.standard_errors[0]:.4f})")
    print(f"  Income:     {ols.coefficients[1]:>8.4f}  (SE: {ols.standard_errors[1]:.4f})")
    print(f"  R²:         {ols.r_squared:.4f}")
    print(f"  F-stat:     {ols.f_statistic:.4f}  (p = {ols.f_p_value:.6f})")
    print(f"  MSE:        {ols.mse:.4f}")
    print()
    print("  Problem: OLS gives equal weight to all observations.")
    print("  Noisy high-income points pull the line and inflate standard errors.")
    print()

    # ── 3. WLS (precision-weighted) ───────────────────────────────────────────
    print("━━━ 3. WLS Regression (precision-weighted, w = 1/variance) ━━━━━━━━━")
    wls = linreg_core.wls_regression(spending, [income], weights)
    print(f"  Intercept:   {wls.coefficients[0]:>8.4f}  (SE: {wls.standard_errors[0]:.4f})")
    print(f"  Income:      {wls.coefficients[1]:>8.4f}  (SE: {wls.standard_errors[1]:.4f})")
    print(f"  R²:          {wls.r_squared:.4f}")
    print(f"  F-stat:      {wls.f_statistic:.4f}  (p = {wls.f_p_value:.6f})")
    print(f"  Residual SE: {wls.residual_std_error:.4f}")
    print()
    print("  Fix: WLS down-weights noisy high-income observations.")
    print("  The fit is driven by reliable low-income points, tightening SEs.")
    print()

    # ── 4. Side-by-side Comparison ────────────────────────────────────────────
    print("━━━ 4. OLS vs WLS Comparison ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'':20}  {'OLS':>12}  {'WLS':>12}")
    print(f"  {'─'*48}")
    print(f"  {'Intercept':<20}  {ols.coefficients[0]:>12.4f}  {wls.coefficients[0]:>12.4f}")
    print(f"  {'Intercept SE':<20}  {ols.standard_errors[0]:>12.4f}  {wls.standard_errors[0]:>12.4f}")
    print(f"  {'Income slope':<20}  {ols.coefficients[1]:>12.4f}  {wls.coefficients[1]:>12.4f}")
    print(f"  {'Income slope SE':<20}  {ols.standard_errors[1]:>12.4f}  {wls.standard_errors[1]:>12.4f}")
    print(f"  {'R²':<20}  {ols.r_squared:>12.4f}  {wls.r_squared:>12.4f}")
    print(f"  {'MSE':<20}  {ols.mse:>12.4f}  {wls.residual_std_error**2:>12.4f}")
    print()

    # ── 5. Fitted Values Comparison ───────────────────────────────────────────
    print("━━━ 5. Fitted Values vs Actual ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Income':>8}  {'Actual':>10}  {'OLS fit':>10}  {'WLS fit':>10}  {'Weight':>10}")
    print(f"  {'─'*54}")
    for i in range(10):
        ols_fit = ols.coefficients[0] + ols.coefficients[1] * income[i]
        print(f"  {income[i]:>8.1f}  {spending[i]:>10.2f}  "
              f"{ols_fit:>10.2f}  {wls.fitted_values[i]:>10.2f}  {weights[i]:>10.4f}")
    print()

    # ── 6. Sanity Check: equal weights → OLS ─────────────────────────────────
    print("━━━ 6. Sanity Check: Equal Weights Reproduces OLS ━━━━━━━━━━━━━━━━━━")
    wls_eq = linreg_core.wls_regression(spending, [income], [1.0] * 10)
    max_diff = max(
        abs(ols.coefficients[0] - wls_eq.coefficients[0]),
        abs(ols.coefficients[1] - wls_eq.coefficients[1]),
    )
    print(f"  OLS intercept:         {ols.coefficients[0]:.6f}")
    print(f"  WLS (equal) intercept: {wls_eq.coefficients[0]:.6f}")
    print(f"  OLS slope:             {ols.coefficients[1]:.6f}")
    print(f"  WLS (equal) slope:     {wls_eq.coefficients[1]:.6f}")
    print(f"  Max coefficient diff:  {max_diff:.2e}")
    print(f"  Matches OLS: {max_diff < 1e-8}")


if __name__ == "__main__":
    main()
