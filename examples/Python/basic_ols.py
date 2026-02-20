"""Basic OLS regression example.

Run with:
    pip install linreg-core
    python basic_ols.py

Demonstrates:
- Simple and multiple OLS regression
- Reading coefficients, standard errors, t-stats, p-values
- Model fit statistics (R², F-stat, AIC/BIC)
- Making predictions from the fitted model
- VIF for multicollinearity detection
"""

import linreg_core


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              OLS REGRESSION — PYTHON BINDINGS                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # ── 1. Simple Linear Regression ───────────────────────────────────────────
    print("━━━ 1. Simple Linear Regression (advertising -> sales) ━━━━━━━━━━━━━━━━")
    y       = [2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 8.2, 9.1]
    advertising = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    result = linreg_core.ols_regression(y, [advertising], ["Intercept", "Advertising"])

    print(f"  Intercept:   {result.coefficients[0]:.4f}  "
          f"(SE: {result.standard_errors[0]:.4f}, "
          f"t: {result.t_statistics[0]:.4f}, "
          f"p: {result.p_values[0]:.4f})")
    print(f"  Advertising: {result.coefficients[1]:.4f}  "
          f"(SE: {result.standard_errors[1]:.4f}, "
          f"t: {result.t_statistics[1]:.4f}, "
          f"p: {result.p_values[1]:.4f})")
    print()
    print(f"  R²:              {result.r_squared:.4f}")
    print(f"  Adjusted R²:     {result.r_squared_adjusted:.4f}")
    print(f"  F-statistic:     {result.f_statistic:.4f}  (p = {result.f_p_value:.6f})")
    print(f"  MSE:             {result.mse:.4f}")
    print(f"  RMSE:            {result.rmse:.4f}")
    print(f"  Observations:    {result.n_observations}")
    print()

    # Predict for a new advertising spend
    spend = 10.0
    pred = result.coefficients[0] + result.coefficients[1] * spend
    print(f"  Prediction at advertising={spend}: {pred:.2f}")
    print()

    # ── 2. Multiple Regression ─────────────────────────────────────────────────
    print("━━━ 2. Multiple Regression (housing: sqft + bedrooms -> price) ━━━━━━━━━")
    price    = [245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1]
    sqft     = [1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0]
    bedrooms = [3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0]
    names    = ["Intercept", "SqFt", "Bedrooms"]

    r2 = linreg_core.ols_regression(price, [sqft, bedrooms], names)

    print(f"  {'Variable':<12}  {'Coef':>10}  {'SE':>10}  {'t':>10}  {'p':>10}  {'Sig'}")
    print(f"  {'─'*64}")
    for i, name in enumerate(names):
        p = r2.p_values[i]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:<12}  {r2.coefficients[i]:>10.4f}  "
              f"{r2.standard_errors[i]:>10.4f}  "
              f"{r2.t_statistics[i]:>10.4f}  "
              f"{p:>10.4f}  {stars}")
    print()
    print(f"  R²: {r2.r_squared:.4f}  Adj R²: {r2.r_squared_adjusted:.4f}  "
          f"F: {r2.f_statistic:.2f}  (p = {r2.f_p_value:.6f})")
    print()

    # ── 3. VIF — multicollinearity check ──────────────────────────────────────
    print("━━━ 3. VIF (Variance Inflation Factor) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  (VIF > 5: moderate concern  |  VIF > 10: severe multicollinearity)")
    print()
    # vif is a list of floats, one per predictor (excluding intercept)
    for i, vif_val in enumerate(r2.vif):
        flag = " !!!" if vif_val > 5 else ""
        print(f"  {names[i+1]:<12}  VIF = {vif_val:.4f}{flag}")
    print()

    # ── 4. Residuals and leverage ──────────────────────────────────────────────
    print("━━━ 4. Residuals and Leverage ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Obs':>4}  {'Actual':>8}  {'Fitted':>8}  {'Residual':>10}  "
          f"{'Std Resid':>10}  {'Leverage':>9}")
    print(f"  {'─'*56}")
    fitted = [r2.coefficients[0] + r2.coefficients[1]*sqft[i] + r2.coefficients[2]*bedrooms[i]
              for i in range(len(price))]
    for i in range(len(price)):
        flag = " *" if abs(r2.standardized_residuals[i]) > 2 else ""
        print(f"  {i+1:>4}  {price[i]:>8.1f}  {fitted[i]:>8.1f}  "
              f"{r2.residuals[i]:>10.2f}  "
              f"{r2.standardized_residuals[i]:>10.3f}  "
              f"{r2.leverage[i]:>9.4f}{flag}")
    print()
    print("  * |standardized residual| > 2 may indicate an outlier")
    print()

    # ── 5. Model selection criteria ───────────────────────────────────────────
    print("━━━ 5. Model Selection: Simple vs Multiple ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    r_simple = linreg_core.ols_regression(price, [sqft], ["Intercept", "SqFt"])
    print(f"  {'Model':<22}  {'R²':>8}  {'Adj R²':>8}  {'F':>10}")
    print(f"  {'─'*52}")
    print(f"  {'SqFt only':<22}  {r_simple.r_squared:>8.4f}  "
          f"{r_simple.r_squared_adjusted:>8.4f}  {r_simple.f_statistic:>10.2f}")
    print(f"  {'SqFt + Bedrooms':<22}  {r2.r_squared:>8.4f}  "
          f"{r2.r_squared_adjusted:>8.4f}  {r2.f_statistic:>10.2f}")
    print()
    print("  Higher Adj R² = better after penalising extra parameters.")


if __name__ == "__main__":
    main()
