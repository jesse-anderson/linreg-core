//! Diagnostic tests example.
//!
//! Demonstrates running various diagnostic tests on regression results
//! to validate assumptions like linearity, homoscedasticity, and normality.

use linreg_core::diagnostics::{self, RainbowMethod, WhiteMethod};

fn main() {
    // Sample data from the classic "mtcars" dataset
    let y = vec![
        21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4,
        10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7,
        15.0, 21.4,
    ]; // mpg

    let x1 = vec![
        6.0, 6.0, 4.0, 6.0, 8.0, 6.0, 8.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 4.0,
        4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0,
    ]; // cyl

    let x2 = vec![
        160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6, 167.6, 275.8, 275.8,
        275.8, 472.0, 460.0, 440.0, 97.0, 78.7, 75.7, 71.1, 120.1, 318.0, 304.0, 350.0, 79.0,
        120.3, 95.1, 351.0, 145.0, 301.0, 121.0,
    ]; // disp

    let x_vars = vec![x1, x2];

    println!("=== Regression Diagnostic Tests ===\n");

    // -------------------------------------------------------------------------
    // Linearity Tests
    // -------------------------------------------------------------------------
    println!("--- Linearity Tests ---");

    // Rainbow Test (tests if fit on central subset matches full dataset)
    match diagnostics::rainbow_test(&y, &x_vars, 0.5, RainbowMethod::R) {
        Ok(rainbow) => {
            if let Some(r) = rainbow.r_result {
                println!("Rainbow Test (R method):");
                println!("  statistic={:.4}, p-value={:.4}", r.statistic, r.p_value);
                println!("  Interpretation: p < 0.05 suggests non-linearity");
            }
        },
        Err(e) => println!("  Rainbow Test error: {}", e),
    }

    // Harvey-Collier Test (uses recursive residuals)
    match diagnostics::harvey_collier_test(&y, &x_vars) {
        Ok(hc) => {
            println!("Harvey-Collier Test:");
            println!("  statistic={:.4}, p-value={:.4}", hc.statistic, hc.p_value);
            println!("  Interpretation: p < 0.05 suggests functional form misspecification");
        },
        Err(e) => println!("  Harvey-Collier Test error: {}", e),
    }

    println!();

    // -------------------------------------------------------------------------
    // Heteroscedasticity Tests
    // -------------------------------------------------------------------------
    println!("--- Heteroscedasticity Tests ---");

    // Breusch-Pagan Test
    match diagnostics::breusch_pagan_test(&y, &x_vars) {
        Ok(bp) => {
            println!("Breusch-Pagan Test:");
            println!("  statistic={:.4}, p-value={:.4}", bp.statistic, bp.p_value);
            println!("  Interpretation: p < 0.05 suggests heteroscedasticity");
        },
        Err(e) => println!("  Breusch-Pagan Test error: {}", e),
    }

    // White Test (more general heteroscedasticity test)
    match diagnostics::white_test(&y, &x_vars, WhiteMethod::R) {
        Ok(white) => {
            if let Some(r) = white.r_result {
                println!("White Test (R method):");
                println!("  statistic={:.4}, p-value={:.4}", r.statistic, r.p_value);
            }
        },
        Err(e) => println!("  White Test error: {}", e),
    }

    println!();

    // -------------------------------------------------------------------------
    // Normality Tests
    // -------------------------------------------------------------------------
    println!("--- Normality Tests (Residuals should be normally distributed) ---");

    // Jarque-Bera Test
    match diagnostics::jarque_bera_test(&y, &x_vars) {
        Ok(jb) => {
            println!("Jarque-Bera Test:");
            println!("  statistic={:.4}, p-value={:.4}", jb.statistic, jb.p_value);
            println!("  Interpretation: p < 0.05 suggests non-normal residuals");
        },
        Err(e) => println!("  Jarque-Bera Test error: {}", e),
    }

    // Shapiro-Wilk Test (powerful for small/medium samples)
    match diagnostics::shapiro_wilk_test(&y, &x_vars) {
        Ok(sw) => {
            println!("Shapiro-Wilk Test:");
            println!("  statistic={:.4}, p-value={:.4}", sw.statistic, sw.p_value);
        },
        Err(e) => println!("  Shapiro-Wilk Test error: {}", e),
    }

    // Anderson-Darling Test (sensitive to tail deviations)
    match diagnostics::anderson_darling_test(&y, &x_vars) {
        Ok(ad) => {
            println!("Anderson-Darling Test:");
            println!("  statistic={:.4}, p-value={:.4}", ad.statistic, ad.p_value);
        },
        Err(e) => println!("  Anderson-Darling Test error: {}", e),
    }

    println!();

    // -------------------------------------------------------------------------
    // Autocorrelation Test
    // -------------------------------------------------------------------------
    println!("--- Autocorrelation Test ---");

    match diagnostics::durbin_watson_test(&y, &x_vars) {
        Ok(dw) => {
            println!("Durbin-Watson Test:");
            println!("  statistic={:.4}", dw.statistic);
            println!("  Interpretation:");
            println!("    - Value near 2.0: no autocorrelation");
            println!("    - Value < 1.5: positive autocorrelation");
            println!("    - Value > 2.5: negative autocorrelation");
        },
        Err(e) => println!("  Durbin-Watson Test error: {}", e),
    }

    println!();

    // -------------------------------------------------------------------------
    // Influential Observations
    // -------------------------------------------------------------------------
    println!("--- Influential Observations ---");

    match diagnostics::cooks_distance_test(&y, &x_vars) {
        Ok(cooks) => {
            println!("Cook's Distance:");
            println!("  Threshold (4/n): {:.4}", cooks.threshold_4_over_n);
            println!("  Observations above threshold:");
            if cooks.influential_4_over_n.is_empty() {
                println!("    None (no highly influential observations)");
            } else {
                for idx in &cooks.influential_4_over_n {
                    println!("    Observation {}: {:.4}", idx + 1, cooks.distances[*idx]);
                }
            }
        },
        Err(e) => println!("  Cook's Distance error: {}", e),
    }
}
