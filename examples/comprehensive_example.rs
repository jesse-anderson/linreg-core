//! Comprehensive regression analysis example.
//!
//! This example demonstrates a complete regression workflow:
//! 1. Fit an OLS regression model
//! 2. Examine model statistics and coefficients
//! 3. Check VIF for multicollinearity
//! 4. Run diagnostic tests to validate assumptions
//! 5. Generate predictions

use linreg_core::core::ols_regression;

fn main() {
    // Longley dataset (classic econometrics dataset with multicollinearity)
    // Predicting employment based on economic indicators
    let y = vec![
        60323.0, 61122.0, 60171.0, 61187.0, 63221.0, 63639.0, 64989.0, 63761.0, 66019.0, 67857.0,
        68169.0, 66513.0, 68655.0, 69564.0, 69331.0, 70551.0,
    ]; // Total Employment

    let gnp = vec![
        234289.0, 259426.0, 258054.0, 284599.0, 328975.0, 346999.0, 365385.0, 363112.0, 397469.0,
        419180.0, 442769.0, 444546.0, 482704.0, 502601.0, 518173.0, 554894.0,
    ]; // GNP

    let armed = vec![
        1590.0, 1406.0, 1230.0, 1275.0, 1495.0, 1606.0, 1641.0, 1483.0, 1541.0, 1679.0, 1704.0,
        1744.0, 1869.0, 1883.0, 2089.0, 2294.0,
    ]; // Armed Forces

    let pop = vec![
        107608.0, 108632.0, 109773.0, 110929.0, 112075.0, 113270.0, 115094.0, 116219.0, 117389.0,
        118734.0, 120445.0, 121950.0, 123366.0, 125368.0, 127852.0, 130081.0,
    ]; // Population

    let time = vec![
        1947.0, 1948.0, 1949.0, 1950.0, 1951.0, 1952.0, 1953.0, 1954.0, 1955.0, 1956.0, 1957.0,
        1958.0, 1959.0, 1960.0, 1961.0, 1962.0,
    ]; // Year

    let names = vec![
        "Intercept".to_string(),
        "GNP".to_string(),
        "Armed Forces".to_string(),
        "Population".to_string(),
        "Year".to_string(),
    ];

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           COMPREHENSIVE OLS REGRESSION ANALYSIS                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // -------------------------------------------------------------------------
    // 1. Fit the Model
    // -------------------------------------------------------------------------
    println!("1. FITTING THE MODEL");
    println!("{}\n", "─".repeat(70));

    let result = match ols_regression(
        &y,
        &[gnp.clone(), armed.clone(), pop.clone(), time.clone()],
        &names,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error fitting model: {}", e);
            return;
        },
    };

    // -------------------------------------------------------------------------
    // 2. Model Summary
    // -------------------------------------------------------------------------
    println!("2. MODEL SUMMARY");
    println!("{}\n", "─".repeat(70));

    println!("                    Model Fit Statistics");
    println!("                    ────────────────────");
    println!("  Observations:              {}", result.n);
    println!("  R-squared:                 {:.4}", result.r_squared);
    println!("  Adjusted R-squared:        {:.4}", result.adj_r_squared);
    println!(
        "  F-statistic:               {:.4} (df: {}.{}, p-value: {:.4})",
        result.f_statistic,
        result.k - 1,
        result.n - result.k,
        result.f_p_value
    );
    println!("  Mean Squared Error:        {:.2}\n", result.mse);

    // -------------------------------------------------------------------------
    // 3. Coefficient Table
    // -------------------------------------------------------------------------
    println!("3. COEFFICIENTS");
    println!("{}\n", "─".repeat(70));

    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "Estimate", "Std.Error", "t value", "Pr(>|t|)", "VIF"
    );
    println!("{}", "─".repeat(70));

    let vif_results = calculate_vif(&y, &[gnp.clone(), armed.clone(), pop.clone(), time.clone()]);

    for (i, name) in names.iter().enumerate() {
        let coef = result.coefficients[i];
        let se = result.std_errors[i];
        let t = result.t_stats[i];
        let p = result.p_values[i];

        // Get VIF if available
        let vif_str = if i == 0 {
            "-".to_string() // Intercept has no VIF
        } else {
            match vif_results.as_ref().and_then(|v| v.get(i - 1)) {
                Some(v) if *v > 10.0 => format!(" {:.2} ⚠", v),
                Some(v) => format!(" {:.2}", v),
                None => "-".to_string(),
            }
        };

        // Significance stars
        let stars = if p < 0.001 {
            "***"
        } else if p < 0.01 {
            "**"
        } else if p < 0.05 {
            "*"
        } else {
            ""
        };

        println!(
            "{:<16} {:>10.2} {:>10.2} {:>10.2} {:>10.4}{} {:>10}",
            name, coef, se, t, p, stars, vif_str
        );
    }
    println!("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05");

    // Print VIF interpretation
    if vif_results.is_some() {
        println!("\nVIF Interpretation:");
        println!("  - VIF < 5:   Low multicollinearity");
        println!("  - VIF 5-10:  Moderate multicollinearity");
        println!("  - VIF > 10:  High multicollinearity ⚠");
    }

    println!();

    // -------------------------------------------------------------------------
    // 4. Diagnostic Tests
    // -------------------------------------------------------------------------
    println!("4. DIAGNOSTIC TESTS");
    println!("{}\n", "─".repeat(70));

    run_diagnostics(&y, &[gnp, armed, pop, time]);

    // -------------------------------------------------------------------------
    // 5. Prediction Example
    // -------------------------------------------------------------------------
    println!("\n5. PREDICTION EXAMPLE");
    println!("{}\n", "─".repeat(70));

    // Predict for 1963 with hypothetical values
    let new_gnp = 580000.0;
    let new_armed = 2400.0;
    let new_pop = 132000.0;
    let new_year = 1963.0;

    let prediction = result.coefficients[0]
        + result.coefficients[1] * new_gnp
        + result.coefficients[2] * new_armed
        + result.coefficients[3] * new_pop
        + result.coefficients[4] * new_year;

    println!("Prediction for 1963:");
    println!("  GNP:          ${:.0}", new_gnp);
    println!("  Armed Forces: {:.0}", new_armed);
    println!("  Population:   {:.0}", new_pop);
    println!("  Year:         {:.0}", new_year);
    println!("  ──────────────────────────────────");
    println!("  Predicted Employment: {:.0}", prediction);
}

// -------------------------------------------------------------------------
// Helper Functions
// -------------------------------------------------------------------------

fn calculate_vif(y: &[f64], x_vars: &[Vec<f64>]) -> Option<Vec<f64>> {
    // Create dummy names for VIF calculation
    let dummy_names: Vec<String> = (0..=x_vars.len()).map(|_| "x".to_string()).collect();
    match ols_regression(y, x_vars, &dummy_names) {
        Ok(result) => Some(result.vif.iter().map(|v| v.vif).collect()),
        Err(_) => None,
    }
}

fn run_diagnostics(y: &[f64], x_vars: &[Vec<f64>]) {
    use linreg_core::diagnostics::{self, RainbowMethod};

    // Helper to print test result
    fn print_test(name: &str, stat: f64, p: f64, interpretation: &str) {
        let status = if p < 0.05 { "FAIL ⚠" } else { "PASS ✓" };
        println!(
            "{:<25} statistic={:>8.3}  p-value={:.4}  {}",
            name, stat, p, status
        );
        println!("  → {}", interpretation);
    }

    println!("Linearity Tests:");
    if let Ok(rainbow) = diagnostics::rainbow_test(y, x_vars, 0.5, RainbowMethod::R) {
        if let Some(r) = rainbow.r_result {
            print_test(
                "Rainbow Test",
                r.statistic,
                r.p_value,
                "Tests linear specification assumption",
            );
        }
    }
    if let Ok(hc) = diagnostics::harvey_collier_test(y, x_vars) {
        print_test(
            "Harvey-Collier",
            hc.statistic,
            hc.p_value,
            "Tests functional form using recursive residuals",
        );
    }

    println!("\nHeteroscedasticity Tests:");
    if let Ok(bp) = diagnostics::breusch_pagan_test(y, x_vars) {
        print_test(
            "Breusch-Pagan",
            bp.statistic,
            bp.p_value,
            "Tests constant variance assumption",
        );
    }

    println!("\nNormality Tests:");
    if let Ok(jb) = diagnostics::jarque_bera_test(y, x_vars) {
        print_test(
            "Jarque-Bera",
            jb.statistic,
            jb.p_value,
            "Tests normality via skewness/kurtosis",
        );
    }
    if let Ok(sw) = diagnostics::shapiro_wilk_test(y, x_vars) {
        print_test(
            "Shapiro-Wilk",
            sw.statistic,
            sw.p_value,
            "Powerful normality test for small/medium samples",
        );
    }

    println!("\nAutocorrelation:");
    if let Ok(dw) = diagnostics::durbin_watson_test(y, x_vars) {
        println!("{:<25} statistic={:>8.3}", "Durbin-Watson", dw.statistic);
        println!("  → Values near 2.0 indicate no autocorrelation");
    }
}
