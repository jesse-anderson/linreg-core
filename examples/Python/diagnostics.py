"""Regression diagnostic tests example.

Run with:
    pip install linreg-core
    python diagnostics.py

Demonstrates:
- Linearity tests (Rainbow, Harvey-Collier, RESET)
- Heteroscedasticity tests (Breusch-Pagan, White)
- Normality tests (Jarque-Bera, Shapiro-Wilk, Anderson-Darling)
- Autocorrelation tests (Durbin-Watson, Breusch-Godfrey)
- Influential observations (Cook's Distance, DFBETAS, DFFITS)
"""

import linreg_core


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           REGRESSION DIAGNOSTIC TESTS — PYTHON BINDINGS            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("Dataset: 25 cars — mpg ~ weight (1000 lbs) + horsepower")
    print()

    # Fuel efficiency dataset: mpg ~ weight + horsepower
    # weight and horsepower are moderately correlated but VIF stays low enough
    # for all diagnostic tests to run cleanly.
    y = [
        28.0, 24.5, 30.2, 18.3, 22.7, 26.1, 15.8, 32.4, 20.5, 17.9,
        25.3, 29.8, 14.6, 23.1, 31.5, 19.4, 27.6, 16.2, 33.1, 21.8,
        24.9, 18.7, 29.0, 22.3, 17.1,
    ]  # mpg

    x1 = [
        2.8, 3.2, 2.5, 3.8, 3.1, 2.9, 4.2, 2.2, 3.5, 4.0,
        3.0, 2.6, 4.5, 3.3, 2.3, 3.7, 2.7, 4.1, 2.1, 3.4,
        3.0, 3.9, 2.6, 3.2, 4.3,
    ]  # weight (1000 lbs)

    x2 = [
        95.0, 110.0, 88.0, 150.0, 105.0, 98.0, 175.0, 78.0, 130.0, 160.0,
        100.0, 92.0, 185.0, 115.0, 82.0, 140.0, 96.0, 165.0, 75.0, 120.0,
        102.0, 155.0, 90.0, 108.0, 170.0,
    ]  # horsepower

    x_vars = [x1, x2]

    def fmt(stat, p):
        status = "FAIL" if p < 0.05 else "PASS"
        return f"stat={stat:8.4f}  p={p:.4f}  {status}"

    # ── 1. Linearity Tests ────────────────────────────────────────────────────
    print("━━━ 1. Linearity Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  (p < 0.05 suggests non-linearity or misspecification)")
    print()

    rainbow = linreg_core.rainbow_test(y, x_vars, fraction=0.5, method="r")
    print(f"  Rainbow Test (R method):    {fmt(rainbow.r_statistic, rainbow.r_p_value)}")

    hc = linreg_core.harvey_collier_test(y, x_vars)
    print(f"  Harvey-Collier:             {fmt(hc.statistic, hc.p_value)}")

    reset = linreg_core.reset_test(y, x_vars, [2, 3], "fitted")
    print(f"  RESET (powers 2,3):         {fmt(reset.statistic, reset.p_value)}")
    print()

    # ── 2. Heteroscedasticity Tests ───────────────────────────────────────────
    print("━━━ 2. Heteroscedasticity Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  (p < 0.05 suggests non-constant variance)")
    print()

    bp = linreg_core.breusch_pagan_test(y, x_vars)
    print(f"  Breusch-Pagan:              {fmt(bp.statistic, bp.p_value)}")

    white_r = linreg_core.r_white_test(y, x_vars)
    print(f"  White Test (R method):      {fmt(white_r.statistic, white_r.p_value)}")

    white_py = linreg_core.python_white_test(y, x_vars)
    print(f"  White Test (Python method): {fmt(white_py.statistic, white_py.p_value)}")
    print()

    # ── 3. Normality Tests ────────────────────────────────────────────────────
    print("━━━ 3. Normality Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  (p < 0.05 suggests non-normal residuals)")
    print()

    jb = linreg_core.jarque_bera_test(y, x_vars)
    print(f"  Jarque-Bera:                {fmt(jb.statistic, jb.p_value)}")

    sw = linreg_core.shapiro_wilk_test(y, x_vars)
    print(f"  Shapiro-Wilk:               {fmt(sw.statistic, sw.p_value)}")

    ad = linreg_core.anderson_darling_test(y, x_vars)
    print(f"  Anderson-Darling:           {fmt(ad.statistic, ad.p_value)}")
    print()

    # ── 4. Autocorrelation Tests ──────────────────────────────────────────────
    print("━━━ 4. Autocorrelation Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    dw = linreg_core.durbin_watson_test(y, x_vars)
    print(f"  Durbin-Watson statistic:    {dw.statistic:.4f}")
    print(f"  Interpretation: ~2.0 = no autocorrelation, "
          f"<1.5 = positive, >2.5 = negative")

    bg = linreg_core.breusch_godfrey_test(y, x_vars, order=2, test_type="chisq")
    print(f"  Breusch-Godfrey (lag=2):    {fmt(bg.statistic, bg.p_value)}")
    print()

    # ── 5. Influential Observations ───────────────────────────────────────────
    print("━━━ 5. Influential Observations ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    cd = linreg_core.cooks_distance_test(y, x_vars)
    print(f"  Cook's Distance:")
    print(f"    Threshold (4/n):   {cd.threshold_4_over_n:.4f}")
    if cd.influential_4_over_n:
        # influential_4_over_n contains 1-based observation numbers
        obs  = list(cd.influential_4_over_n)
        vals = [f"{cd.distances[i - 1]:.4f}" for i in cd.influential_4_over_n]
        print(f"    Influential obs (1-based): {obs}")
        print(f"    Distances:                 {vals}")
    else:
        print(f"    No highly influential observations")
    print()

    dfb = linreg_core.dfbetas_test(y, x_vars)
    print(f"  DFBETAS:")
    print(f"    Threshold (2/√n): {dfb.threshold:.4f}")
    if dfb.influential_observations:
        for coef_idx, obs_list in sorted(dfb.influential_observations.items()):
            label = "Intercept" if coef_idx == 1 else f"X{coef_idx - 1}"
            print(f"    {label}: obs {obs_list}")
    else:
        print(f"    No influential observations detected")
    print()

    dff = linreg_core.dffits_test(y, x_vars)
    print(f"  DFFITS:")
    print(f"    Threshold (2√(p/n)): {dff.threshold:.4f}")
    if dff.influential_observations:
        print(f"    Influential obs (1-based): {dff.influential_observations}")
    else:
        print(f"    No influential observations detected")
    print()

    # ── 6. VIF (Multicollinearity) ─────────────────────────────────────────────
    print("━━━ 6. Variance Inflation Factor (VIF) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  (VIF > 5: moderate concern  |  VIF > 10: severe multicollinearity)")
    print()

    vif = linreg_core.vif_test(y, x_vars)
    for v in vif.vif_results:
        flag = " !!!" if v.vif > 10 else " !" if v.vif > 5 else ""
        print(f"  {v.variable:<12}  VIF = {v.vif:.4f}{flag}  ({v.interpretation})")
    print(f"  Max VIF: {vif.max_vif:.4f}")


if __name__ == "__main__":
    main()
