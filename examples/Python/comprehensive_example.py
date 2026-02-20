"""Comprehensive OLS regression analysis example.

Run with:
    pip install linreg-core
    python comprehensive_example.py

Demonstrates a complete regression workflow:
1. Fit an OLS model (Longley econometrics dataset)
2. Examine model statistics and coefficient table
3. Check VIF for multicollinearity
4. Run diagnostic tests to validate assumptions
5. Generate predictions for new observations
"""

import linreg_core


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def print_test(name, stat, p):
    status = "FAIL" if p < 0.05 else "PASS"
    print(f"  {name:<26} stat={stat:9.3f}  p={p:.4f}  {status}")


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           COMPREHENSIVE OLS REGRESSION ANALYSIS                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # Longley dataset — classic econometrics dataset, predicting employment
    y = [
        60323.0, 61122.0, 60171.0, 61187.0, 63221.0, 63639.0, 64989.0,
        63761.0, 66019.0, 67857.0, 68169.0, 66513.0, 68655.0, 69564.0,
        69331.0, 70551.0,
    ]  # Total Employment

    gnp = [
        234289.0, 259426.0, 258054.0, 284599.0, 328975.0, 346999.0, 365385.0,
        363112.0, 397469.0, 419180.0, 442769.0, 444546.0, 482704.0, 502601.0,
        518173.0, 554894.0,
    ]

    armed = [
        1590.0, 1406.0, 1230.0, 1275.0, 1495.0, 1606.0, 1641.0, 1483.0,
        1541.0, 1679.0, 1704.0, 1744.0, 1869.0, 1883.0, 2089.0, 2294.0,
    ]

    pop = [
        107608.0, 108632.0, 109773.0, 110929.0, 112075.0, 113270.0, 115094.0,
        116219.0, 117389.0, 118734.0, 120445.0, 121950.0, 123366.0, 125368.0,
        127852.0, 130081.0,
    ]

    year = [
        1947.0, 1948.0, 1949.0, 1950.0, 1951.0, 1952.0, 1953.0, 1954.0,
        1955.0, 1956.0, 1957.0, 1958.0, 1959.0, 1960.0, 1961.0, 1962.0,
    ]

    x_vars  = [gnp, armed, pop, year]
    names   = ["Intercept", "GNP", "Armed Forces", "Population", "Year"]

    # ── 1. Fit the Model ───────────────────────────────────────────────────────
    print("━━━ 1. Fitting the Model ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    result = linreg_core.ols_regression(y, x_vars, names)
    print(f"  Observations:       {result.n_observations}")
    print(f"  R²:                 {result.r_squared:.4f}")
    print(f"  Adjusted R²:        {result.r_squared_adjusted:.4f}")
    print(f"  F-statistic:        {result.f_statistic:.4f}  (p = {result.f_p_value:.4e})")
    print(f"  MSE:                {result.mse:.2f}")
    print(f"  RMSE:               {result.rmse:.2f}")
    print()

    # ── 2. Coefficient Table ───────────────────────────────────────────────────
    print("━━━ 2. Coefficient Table ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Variable':<16} {'Estimate':>12} {'Std.Error':>10} {'t value':>10} {'Pr(>|t|)':>12} {'Sig'}")
    print(f"  {'─'*66}")
    for i, name in enumerate(names):
        coef = result.coefficients[i]
        se   = result.standard_errors[i]
        t    = result.t_statistics[i]
        p    = result.p_values[i]
        print(f"  {name:<16} {coef:>12.2f} {se:>10.2f} {t:>10.3f} {p:>12.4f} {sig_stars(p)}")
    print()
    print("  Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05")
    print()

    # ── 3. VIF Analysis ────────────────────────────────────────────────────────
    print("━━━ 3. VIF — Multicollinearity Check ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  (Longley is a canonical high-collinearity dataset — VIF will be large)")
    print()
    vif = linreg_core.vif_test(y, x_vars)
    for v in vif.vif_results:
        flag = " XXX" if v.vif > 10 else ""
        print(f"  {v.variable:<16} VIF = {v.vif:>10.2f}{flag}")
    print()

    # ── 4. Diagnostic Tests ────────────────────────────────────────────────────
    print("━━━ 4. Diagnostic Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    print("  Linearity:")
    rainbow = linreg_core.rainbow_test(y, x_vars, fraction=0.5, method="r")
    print_test("Rainbow (R)", rainbow.r_statistic, rainbow.r_p_value)

    print()
    print("  Heteroscedasticity:")
    bp = linreg_core.breusch_pagan_test(y, x_vars)
    print_test("Breusch-Pagan", bp.statistic, bp.p_value)

    print()
    print("  Normality:")
    jb = linreg_core.jarque_bera_test(y, x_vars)
    print_test("Jarque-Bera", jb.statistic, jb.p_value)
    sw = linreg_core.shapiro_wilk_test(y, x_vars)
    print_test("Shapiro-Wilk", sw.statistic, sw.p_value)

    print()
    print("  Autocorrelation:")
    dw = linreg_core.durbin_watson_test(y, x_vars)
    print(f"  {'Durbin-Watson':<26} stat={dw.statistic:9.3f}  (~2.0 = no autocorrelation)")
    print()

    # ── 5. Prediction ──────────────────────────────────────────────────────────
    print("━━━ 5. Prediction for 1963 (extrapolation) ━━━━━━━━━━━━━━━━━━━━━━━━━")
    new_gnp    = 580000.0
    new_armed  = 2400.0
    new_pop    = 132000.0
    new_year   = 1963.0

    coefs = result.coefficients
    pred = (coefs[0]
            + coefs[1] * new_gnp
            + coefs[2] * new_armed
            + coefs[3] * new_pop
            + coefs[4] * new_year)

    print(f"  GNP:                ${new_gnp:,.0f}")
    print(f"  Armed Forces:       {new_armed:,.0f}")
    print(f"  Population:         {new_pop:,.0f}")
    print(f"  Year:               {new_year:.0f}")
    print(f"  ──────────────────────────────────")
    print(f"  Predicted Employment: {pred:,.0f}")
    print()

    # ── 6. Fitted vs Actual ────────────────────────────────────────────────────
    print("━━━ 6. Fitted Values vs Actual (first 8 obs) ━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Year':>6}  {'Actual':>10}  {'Fitted':>10}  {'Residual':>10}")
    print(f"  {'─'*42}")
    fitted = [coefs[0] + coefs[1]*gnp[i] + coefs[2]*armed[i]
              + coefs[3]*pop[i] + coefs[4]*year[i] for i in range(16)]
    for i in range(8):
        resid = y[i] - fitted[i]
        print(f"  {int(year[i]):>6}  {y[i]:>10.0f}  {fitted[i]:>10.0f}  {resid:>10.1f}")


if __name__ == "__main__":
    main()
