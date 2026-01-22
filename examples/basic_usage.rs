//! Basic OLS regression example.
//!
//! Demonstrates simple linear regression with one predictor.

use linreg_core::core::ols_regression;

fn main() {
    // Sample data: advertising spend vs sales
    let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0, 8.2, 9.1];
    let advertising = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let names = vec!["Intercept".to_string(), "Advertising".to_string()];

    match ols_regression(&y, &[advertising], &names) {
        Ok(result) => {
            println!("=== OLS Regression Results ===\n");

            println!("Coefficients:");
            for (i, name) in names.iter().enumerate() {
                let coef = result.coefficients[i];
                let se = result.std_errors[i];
                let t_stat = result.t_stats[i];
                let p_val = result.p_values[i];
                println!(
                    "  {:12}: {:.4} (SE: {:.4}, t: {:.4}, p: {:.4})",
                    name, coef, se, t_stat, p_val
                );
            }

            println!("\nModel Fit:");
            println!("  R-squared:       {:.4}", result.r_squared);
            println!("  Adjusted R-squared: {:.4}", result.adj_r_squared);
            println!("  F-statistic:     {:.4}", result.f_statistic);
            println!("  F p-value:       {:.6}", result.f_p_value);
            println!("  MSE:             {:.4}", result.mse);
            println!("  Observations:    {}", result.n);

            // Make a prediction
            let new_advertising = 10.0;
            let prediction = result.coefficients[0] + result.coefficients[1] * new_advertising;
            println!(
                "\nPrediction for advertising={} is {:.2}",
                new_advertising, prediction
            );
        },
        Err(e) => eprintln!("Error: {}", e),
    }
}
