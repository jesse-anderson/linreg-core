//! WASM bindings for feature importance metrics.
//!
//! This module provides WebAssembly bindings for computing feature importance
//! metrics in the browser.

use crate::core::RegressionOutput;
use crate::error::Result;
use crate::feature_importance::{
    permutation_importance_ols_named, shap_values_linear_named, standardized_coefficients_named,
    vif_ranking as core_vif_ranking, PermutationImportanceOptions, PermutationImportanceOutput,
    ShapOutput, StandardizedCoefficientsOutput, VifRankingOutput,
};
use wasm_bindgen::prelude::*;

/// Computes standardized coefficients for feature importance.
///
/// # Arguments
///
/// * `x_json` - JSON array of predictor arrays (each array is a column)
/// * `coefficients_json` - JSON array of coefficients including intercept
/// * `variable_names_json` - JSON array of variable names
/// * `y_std` - Standard deviation of response variable
///
/// # Returns
///
/// JSON string of [`StandardizedCoefficientsOutput`]
///
/// # Example
///
/// ```javascript
/// const result = JSON.parse(standardized_coefficients(
///     JSON.stringify([[1,2,3,4,5], [10,20,30,40,50]]),
///     JSON.stringify([1, 0.5, -0.3]),
///     JSON.stringify(["Temperature", "Pressure"]),
///     2.5
/// ));
/// console.log(result.standardized_coefficients);
/// ```
#[wasm_bindgen]
pub fn standardized_coefficients(
    x_json: &str,
    coefficients_json: &str,
    variable_names_json: &str,
    y_std: f64,
) -> String {
    let result: Result<StandardizedCoefficientsOutput> = (|| {
        let x_vars: Vec<Vec<f64>> = serde_json::from_str(x_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid x_json: {}", e)))?;
        let coefficients: Vec<f64> = serde_json::from_str(coefficients_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid coefficients_json: {}", e)))?;
        let variable_names: Vec<String> = serde_json::from_str(variable_names_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid variable_names_json: {}", e)))?;

        standardized_coefficients_named(&coefficients, &x_vars, &variable_names, y_std)
    })();

    match result {
        Ok(output) => serde_json::to_string(&output).unwrap_or_else(|_| crate::error::error_json("Serialization failed")),
        Err(e) => crate::error::error_json(&e.to_string()),
    }
}

/// Computes SHAP (SHapley Additive exPlanations) values for linear models.
///
/// # Arguments
///
/// * `x_json` - JSON array of predictor arrays (each array is a column)
/// * `coefficients_json` - JSON array of coefficients including intercept
/// * `variable_names_json` - JSON array of variable names
///
/// # Returns
///
/// JSON string of [`ShapOutput`]
///
/// # Example
///
/// ```javascript
/// const result = JSON.parse(shap_values_linear(
///     JSON.stringify([[1,2,3], [2,4,6]]),
///     JSON.stringify([5, 2, 3]),
///     JSON.stringify(["X1", "X2"])
/// ));
/// console.log(result.mean_abs_shap); // Global importance
/// console.log(result.shap_values); // Local contributions
/// ```
#[wasm_bindgen]
pub fn shap_values_linear(
    x_json: &str,
    coefficients_json: &str,
    variable_names_json: &str,
) -> String {
    let result: Result<ShapOutput> = (|| {
        let x_vars: Vec<Vec<f64>> = serde_json::from_str(x_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid x_json: {}", e)))?;
        let coefficients: Vec<f64> = serde_json::from_str(coefficients_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid coefficients_json: {}", e)))?;
        let variable_names: Vec<String> = serde_json::from_str(variable_names_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid variable_names_json: {}", e)))?;

        shap_values_linear_named(&x_vars, &coefficients, &variable_names)
    })();

    match result {
        Ok(output) => serde_json::to_string(&output).unwrap_or_else(|_| crate::error::error_json("Serialization failed")),
        Err(e) => crate::error::error_json(&e.to_string()),
    }
}

/// Computes permutation importance for OLS regression.
///
/// # Arguments
///
/// * `y_json` - JSON array of response values
/// * `x_json` - JSON array of predictor arrays (each array is a column)
/// * `fit_json` - JSON string of OLS fit result
/// * `n_permutations` - Number of permutation iterations
/// * `seed` - Random seed (use 0 for no seed)
///
/// # Returns
///
/// JSON string of [`PermutationImportanceOutput`]
///
/// # Example
///
/// ```javascript
/// const y = [2.5, 3.7, 4.2, 5.1, 6.3];
/// const x = [[1,2,3,4,5], [2,4,5,4,3]];
/// const fit = JSON.parse(ols_regression(...)); // from regression module
///
/// const result = JSON.parse(permutation_importance_ols(
///     JSON.stringify(y),
///     JSON.stringify(x),
///     JSON.stringify(fit),
///     50,  // n_permutations
///     42   // seed
/// ));
/// console.log(result.importance);
/// ```
#[wasm_bindgen]
pub fn permutation_importance_ols(
    y_json: &str,
    x_json: &str,
    fit_json: &str,
    n_permutations: usize,
    seed: u64,
) -> String {
    let result: Result<PermutationImportanceOutput> = (|| {
        let y: Vec<f64> = serde_json::from_str(y_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid y_json: {}", e)))?;
        let x_vars: Vec<Vec<f64>> = serde_json::from_str(x_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid x_json: {}", e)))?;
        let fit: RegressionOutput = serde_json::from_str(fit_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid fit_json: {}", e)))?;

        let variable_names: Vec<String> = (0..x_vars.len())
            .map(|i| format!("X{}", i + 1))
            .collect();

        let options = PermutationImportanceOptions {
            n_permutations,
            seed: if seed == 0 { None } else { Some(seed) },
            compute_intervals: false,
            interval_confidence: 0.95,
        };

        permutation_importance_ols_named(&y, &x_vars, &fit, &options, &variable_names)
    })();

    match result {
        Ok(output) => serde_json::to_string(&output).unwrap_or_else(|_| crate::error::error_json("Serialization failed")),
        Err(e) => crate::error::error_json(&e.to_string()),
    }
}

/// Computes VIF (Variance Inflation Factor) ranking.
///
/// # Arguments
///
/// * `vif_json` - JSON array of VIF results from OLS output
///
/// # Returns
///
/// JSON string of [`VifRankingOutput`]
///
/// # Example
///
/// ```javascript
/// const fit = JSON.parse(ols_regression(...));
/// const result = JSON.parse(vif_ranking(JSON.stringify(fit.vif)));
/// console.log(result.ranking); // Sorted by VIF (lowest first)
/// ```
#[wasm_bindgen]
pub fn vif_ranking(vif_json: &str) -> String {
    let result: Result<VifRankingOutput> = (|| {
        let vif_results: Vec<crate::core::VifResult> = serde_json::from_str(vif_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid vif_json: {}", e)))?;

        Ok(core_vif_ranking(&vif_results))
    })();

    match result {
        Ok(output) => serde_json::to_string(&output).unwrap_or_else(|_| crate::error::error_json("Serialization failed")),
        Err(e) => crate::error::error_json(&e.to_string()),
    }
}

/// Computes complete feature importance analysis for OLS regression.
///
/// This combines standardized coefficients, SHAP values, VIF ranking,
/// and permutation importance into a single call.
///
/// # Arguments
///
/// * `y_json` - JSON array of response values
/// * `x_json` - JSON array of predictor arrays (each array is a column)
/// * `variable_names_json` - JSON array of variable names
/// * `y_std` - Standard deviation of response variable
/// * `n_permutations` - Number of permutation iterations
/// * `seed` - Random seed (use 0 for no seed)
///
/// # Returns
///
/// JSON string with all feature importance metrics
///
/// # Example
///
/// ```javascript
/// const y = [2.5, 3.7, 4.2, 5.1, 6.3];
/// const x = [[1,2,3,4,5], [2,4,5,4,3]];
/// const names = ["Temperature", "Pressure"];
///
/// const result = JSON.parse(feature_importance_ols(
///     JSON.stringify(y),
///     JSON.stringify(x),
///     JSON.stringify(names),
///     2.5,   // y_std
///     50,    // n_permutations
///     42     // seed
/// ));
///
/// console.log(result.standardized_coefficients);
/// console.log(result.shap);
/// console.log(result.permutation_importance);
/// console.log(result.vif_ranking);
/// ```
#[wasm_bindgen]
pub fn feature_importance_ols(
    y_json: &str,
    x_json: &str,
    variable_names_json: &str,
    y_std: f64,
    n_permutations: usize,
    seed: u64,
) -> String {
    use serde_json::json;

    let result: Result<serde_json::Value> = (|| {
        let y: Vec<f64> = serde_json::from_str(y_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid y_json: {}", e)))?;
        let x_vars: Vec<Vec<f64>> = serde_json::from_str(x_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid x_json: {}", e)))?;
        let variable_names: Vec<String> = serde_json::from_str(variable_names_json)
            .map_err(|e| crate::error::Error::InvalidInput(format!("Invalid variable_names_json: {}", e)))?;

        // Create OLS fit
        let names_with_intercept = {
            let mut names = vec!["Intercept".to_string()];
            names.extend(variable_names.clone());
            names
        };

        let fit = crate::core::ols_regression(&y, &x_vars, &names_with_intercept)?;

        // Compute all metrics
        let std_coefs = standardized_coefficients_named(&fit.coefficients, &x_vars, &variable_names, y_std)?;
        let shap = shap_values_linear_named(&x_vars, &fit.coefficients, &variable_names)?;

        let perm_options = PermutationImportanceOptions {
            n_permutations,
            seed: if seed == 0 { None } else { Some(seed) },
            compute_intervals: false,
            interval_confidence: 0.95,
        };
        let perm_importance = permutation_importance_ols_named(&y, &x_vars, &fit, &perm_options, &variable_names)?;

        let vif_rank = core_vif_ranking(&fit.vif);

        Ok(json!({
            "standardized_coefficients": std_coefs,
            "shap": shap,
            "permutation_importance": perm_importance,
            "vif_ranking": vif_rank,
        }))
    })();

    match result {
        Ok(output) => serde_json::to_string(&output).unwrap_or_else(|_| crate::error::error_json("Serialization failed")),
        Err(e) => crate::error::error_json(&e.to_string()),
    }
}
