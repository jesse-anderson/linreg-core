// ============================================================================
// K-Fold Cross Validation Implementation
// ============================================================================

//! Core K-Fold Cross Validation implementation for linear regression models.
//!
//! This module provides the main cross-validation functions that:
//! 1. Split data into K folds
//! 2. For each fold: fit model on training data, evaluate on test data
//! 3. Aggregate metrics across folds

use crate::core::{self, RegressionOutput};
use crate::cross_validation::metrics::{compute_mae, compute_mse, compute_rmse, compute_r_squared};
use crate::cross_validation::splits::create_kfold_splits;
use crate::cross_validation::types::{CVResult, FoldResult, KFoldOptions};
use crate::error::{Error, Result};
use crate::linalg::Matrix;
use crate::regularized::{elastic_net, lasso, preprocess, ridge};

/// Extracts elements from a slice by indices.
///
/// # Arguments
///
/// * `data` — Source slice
/// * `indices` — Indices to extract
///
/// # Returns
///
/// A new vector containing the elements at the specified indices.
fn extract_by_indices<T: Clone>(data: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| data[i].clone()).collect()
}

/// Extracts predictor columns by indices.
///
/// Each predictor column is filtered to include only the specified indices.
///
/// # Arguments
///
/// * `x_vars` — Slice of predictor columns
/// * `indices` — Indices to extract
///
/// # Returns
///
/// A new vector of predictor columns with only the specified rows.
fn extract_x_vars_by_indices(x_vars: &[Vec<f64>], indices: &[usize]) -> Vec<Vec<f64>> {
    x_vars.iter().map(|col| extract_by_indices(col, indices)).collect()
}

/// Builds a design matrix from predictor columns.
///
/// Creates a matrix with an intercept column (all 1s) followed by the predictor columns.
///
/// # Arguments
///
/// * `x_vars` — Predictor columns
///
/// # Returns
///
/// A Matrix with n rows and (p + 1) columns (intercept + predictors).
fn build_design_matrix(x_vars: &[Vec<f64>], n: usize) -> Matrix {
    let p = x_vars.len();
    let mut data = vec![1.0; n * (p + 1)]; // Intercept column

    for (j, x_var) in x_vars.iter().enumerate() {
        for (i, &val) in x_var.iter().enumerate() {
            data[i * (p + 1) + j + 1] = val;
        }
    }

    Matrix::new(n, p + 1, data)
}

/// Performs K-Fold Cross Validation for OLS regression.
///
/// # Arguments
///
/// * `y` — Response variable values
/// * `x_vars` — Predictor variables (column vectors)
/// * `variable_names` — Names for the variables (for OLS fitting)
/// * `options` — CV configuration options
///
/// # Returns
///
/// Aggregated cross-validation results with mean/std metrics across all folds.
///
/// # Example
///
/// ```rust
/// use linreg_core::cross_validation::{kfold_cv_ols, KFoldOptions};
///
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let x2 = vec![2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
/// let names = vec!["Intercept".into(), "X1".into(), "X2".into()];
///
/// let options = KFoldOptions::new(4).with_shuffle(true).with_seed(42);
/// let result = kfold_cv_ols(&y, &[x1, x2], &names, &options)?;
///
/// println!("CV RMSE: {:.4} +/- {:.4}", result.mean_rmse, result.std_rmse);
/// println!("CV R²: {:.4}", result.mean_r_squared);
/// # Ok::<(), linreg_core::Error>(())
/// ```
pub fn kfold_cv_ols(
    y: &[f64],
    x_vars: &[Vec<f64>],
    variable_names: &[String],
    options: &KFoldOptions,
) -> Result<CVResult> {
    // Validate inputs
    let n_samples = y.len();
    options.validate(n_samples)?;

    // Create train/test splits
    let splits = create_kfold_splits(
        n_samples,
        options.n_folds,
        options.shuffle,
        options.seed,
    )?;

    let mut fold_results = Vec::with_capacity(splits.len());
    let mut fold_coefficients = Vec::with_capacity(splits.len());

    // Process each fold
    for (fold_idx, (train_indices, test_indices)) in splits.iter().enumerate() {
        // Extract training data
        let y_train = extract_by_indices(y, train_indices);
        let x_train = extract_x_vars_by_indices(x_vars, train_indices);

        // Extract test data
        let y_test = extract_by_indices(y, test_indices);
        let x_test = extract_x_vars_by_indices(x_vars, test_indices);

        // Fit OLS on training data
        let fit_result = core::ols_regression(&y_train, &x_train, variable_names)?;

        // Get predictions on test set
        let y_test_pred = predict_ols(&fit_result, &x_test);

        // Get predictions on training set (for overfitting detection)
        let y_train_pred = predict_ols(&fit_result, &x_train);

        // Compute metrics
        let test_mse = compute_mse(&y_test, &y_test_pred);
        let test_rmse = compute_rmse(&y_test, &y_test_pred);
        let test_mae = compute_mae(&y_test, &y_test_pred);
        let test_r_squared = compute_r_squared(&y_test, &y_test_pred);
        let train_r_squared = compute_r_squared(&y_train, &y_train_pred);

        // Store fold results
        fold_results.push(FoldResult::new(
            fold_idx + 1, // 1-based indexing
            train_indices.len(),
            test_indices.len(),
            test_mse,
            test_rmse,
            test_mae,
            test_r_squared,
            train_r_squared,
        ));

        // Store coefficients
        fold_coefficients.push(fit_result.coefficients.clone());
    }

    // Aggregate results
    Ok(CVResult::from_folds(
        n_samples,
        options.n_folds,
        fold_results,
        fold_coefficients,
    ))
}

/// Generates predictions using OLS coefficients.
///
/// # Arguments
///
/// * `fit_result` — Fitted OLS model
/// * `x_vars` — Predictor variables
///
/// # Returns
///
/// Predicted values for each observation.
fn predict_ols(fit_result: &RegressionOutput, x_vars: &[Vec<f64>]) -> Vec<f64> {
    let n = x_vars[0].len();
    let coefficients = &fit_result.coefficients;

    let mut predictions = Vec::with_capacity(n);

    for i in 0..n {
        // Start with intercept (coefficient 0)
        // NAN coefficients are treated as 0 (dropped by pivoted QR for rank-deficient data)
        let mut pred = if coefficients[0].is_nan() { 0.0 } else { coefficients[0] };

        // Add contribution from each predictor
        for (j, x_var) in x_vars.iter().enumerate() {
            let coef = coefficients[j + 1];
            if !coef.is_nan() {
                pred += coef * x_var[i];
            }
        }

        predictions.push(pred);
    }

    predictions
}

/// Performs K-Fold Cross Validation for Ridge regression.
///
/// # Arguments
///
/// * `x_vars` — Predictor variables (column vectors)
/// * `y` — Response variable values
/// * `lambda` — Regularization strength
/// * `standardize` — Whether to standardize predictors
/// * `options` — CV configuration options
///
/// # Returns
///
/// Aggregated cross-validation results.
///
/// # Example
///
/// ```ignore
/// # use linreg_core::cross_validation::{kfold_cv_ridge, KFoldOptions};
/// # use linreg_core::linalg::Matrix;
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0];
/// let options = KFoldOptions::new(5).with_shuffle(false);
/// let result = kfold_cv_ridge(&[x1], &y, 0.1, true, &options).unwrap();
/// println!("CV RMSE: {:.4}", result.mean_rmse);
/// ```
pub fn kfold_cv_ridge(
    x_vars: &[Vec<f64>],
    y: &[f64],
    lambda: f64,
    standardize: bool,
    options: &KFoldOptions,
) -> Result<CVResult> {
    let n_samples = y.len();
    options.validate(n_samples)?;

    if lambda < 0.0 {
        return Err(Error::InvalidInput("lambda must be non-negative".to_string()));
    }

    let splits = create_kfold_splits(
        n_samples,
        options.n_folds,
        options.shuffle,
        options.seed,
    )?;

    let mut fold_results = Vec::with_capacity(splits.len());
    let mut fold_coefficients = Vec::with_capacity(splits.len());

    for (fold_idx, (train_indices, test_indices)) in splits.iter().enumerate() {
        let y_train = extract_by_indices(y, train_indices);
        let x_train = extract_x_vars_by_indices(x_vars, train_indices);
        let y_test = extract_by_indices(y, test_indices);
        let x_test = extract_x_vars_by_indices(x_vars, test_indices);

        // Build design matrices
        let x_train_matrix = build_design_matrix(&x_train, y_train.len());
        let x_test_matrix = build_design_matrix(&x_test, y_test.len());

        // Fit Ridge
        let ridge_options = ridge::RidgeFitOptions {
            lambda,
            intercept: true,
            standardize,
            max_iter: 100000,
            tol: 1e-7,
            warm_start: None,
            weights: None,
        };

        let fit_result = ridge::ridge_fit(&x_train_matrix, &y_train, &ridge_options)?;

        // Predict using ridge predict function
        let y_test_pred = ridge::predict_ridge(&fit_result, &x_test_matrix);
        let y_train_pred = ridge::predict_ridge(&fit_result, &x_train_matrix);

        // Compute metrics
        let test_mse = compute_mse(&y_test, &y_test_pred);
        let test_rmse = compute_rmse(&y_test, &y_test_pred);
        let test_mae = compute_mae(&y_test, &y_test_pred);
        let test_r_squared = compute_r_squared(&y_test, &y_test_pred);
        let train_r_squared = compute_r_squared(&y_train, &y_train_pred);

        fold_results.push(FoldResult::new(
            fold_idx + 1,
            train_indices.len(),
            test_indices.len(),
            test_mse,
            test_rmse,
            test_mae,
            test_r_squared,
            train_r_squared,
        ));

        // Collect coefficients: intercept + slope coefficients
        let mut coeffs = vec![fit_result.intercept];
        coeffs.extend(fit_result.coefficients.clone());
        fold_coefficients.push(coeffs);
    }

    Ok(CVResult::from_folds(
        n_samples,
        options.n_folds,
        fold_results,
        fold_coefficients,
    ))
}

/// Performs K-Fold Cross Validation for Lasso regression.
///
/// # Arguments
///
/// * `x_vars` — Predictor variables (column vectors)
/// * `y` — Response variable values
/// * `lambda` — Regularization strength
/// * `standardize` — Whether to standardize predictors
/// * `options` — CV configuration options
///
/// # Returns
///
/// Aggregated cross-validation results.
///
/// # Example
///
/// ```ignore
/// # use linreg_core::cross_validation::{kfold_cv_lasso, KFoldOptions};
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0];
/// let options = KFoldOptions::new(3).with_shuffle(false);
/// let result = kfold_cv_lasso(&[x1], &y, 0.1, true, &options).unwrap();
/// println!("CV RMSE: {:.4}", result.mean_rmse);
/// ```
pub fn kfold_cv_lasso(
    x_vars: &[Vec<f64>],
    y: &[f64],
    lambda: f64,
    standardize: bool,
    options: &KFoldOptions,
) -> Result<CVResult> {
    let n_samples = y.len();
    options.validate(n_samples)?;

    if lambda < 0.0 {
        return Err(Error::InvalidInput("lambda must be non-negative".to_string()));
    }

    let splits = create_kfold_splits(
        n_samples,
        options.n_folds,
        options.shuffle,
        options.seed,
    )?;

    let mut fold_results = Vec::with_capacity(splits.len());
    let mut fold_coefficients = Vec::with_capacity(splits.len());

    for (fold_idx, (train_indices, test_indices)) in splits.iter().enumerate() {
        let y_train = extract_by_indices(y, train_indices);
        let x_train = extract_x_vars_by_indices(x_vars, train_indices);
        let y_test = extract_by_indices(y, test_indices);
        let x_test = extract_x_vars_by_indices(x_vars, test_indices);

        let x_train_matrix = build_design_matrix(&x_train, y_train.len());
        let x_test_matrix = build_design_matrix(&x_test, y_test.len());

        let lasso_options = lasso::LassoFitOptions {
            lambda,
            intercept: true,
            standardize,
            max_iter: 100000,
            tol: 1e-7,
            ..Default::default()
        };

        let fit_result = lasso::lasso_fit(&x_train_matrix, &y_train, &lasso_options)?;

        // Predict using lasso predict function
        let y_test_pred = lasso::predict_lasso(&fit_result, &x_test_matrix);
        let y_train_pred = lasso::predict_lasso(&fit_result, &x_train_matrix);

        let test_mse = compute_mse(&y_test, &y_test_pred);
        let test_rmse = compute_rmse(&y_test, &y_test_pred);
        let test_mae = compute_mae(&y_test, &y_test_pred);
        let test_r_squared = compute_r_squared(&y_test, &y_test_pred);
        let train_r_squared = compute_r_squared(&y_train, &y_train_pred);

        fold_results.push(FoldResult::new(
            fold_idx + 1,
            train_indices.len(),
            test_indices.len(),
            test_mse,
            test_rmse,
            test_mae,
            test_r_squared,
            train_r_squared,
        ));

        let mut coeffs = vec![fit_result.intercept];
        coeffs.extend(fit_result.coefficients.clone());
        fold_coefficients.push(coeffs);
    }

    Ok(CVResult::from_folds(
        n_samples,
        options.n_folds,
        fold_results,
        fold_coefficients,
    ))
}

/// Performs K-Fold Cross Validation for Elastic Net regression.
///
/// # Arguments
///
/// * `x_vars` — Predictor variables (column vectors)
/// * `y` — Response variable values
/// * `lambda` — Regularization strength
/// * `alpha` — Mixing parameter (0 = Ridge, 1 = Lasso)
/// * `standardize` — Whether to standardize predictors
/// * `options` — CV configuration options
///
/// # Returns
///
/// Aggregated cross-validation results.
///
/// # Example
///
/// ```ignore
/// # use linreg_core::cross_validation::{kfold_cv_elastic_net, KFoldOptions};
/// let y = vec![2.0, 4.0, 6.0, 8.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0];
/// let options = KFoldOptions::new(3).with_shuffle(false);
/// let result = kfold_cv_elastic_net(&[x1], &y, 0.1, 0.5, true, &options).unwrap();
/// println!("CV RMSE: {:.4}", result.mean_rmse);
/// ```
pub fn kfold_cv_elastic_net(
    x_vars: &[Vec<f64>],
    y: &[f64],
    lambda: f64,
    alpha: f64,
    standardize: bool,
    options: &KFoldOptions,
) -> Result<CVResult> {
    let n_samples = y.len();
    options.validate(n_samples)?;

    if lambda < 0.0 {
        return Err(Error::InvalidInput("lambda must be non-negative".to_string()));
    }
    if alpha < 0.0 || alpha > 1.0 {
        return Err(Error::InvalidInput("alpha must be between 0 and 1".to_string()));
    }

    let splits = create_kfold_splits(
        n_samples,
        options.n_folds,
        options.shuffle,
        options.seed,
    )?;

    let mut fold_results = Vec::with_capacity(splits.len());
    let mut fold_coefficients = Vec::with_capacity(splits.len());

    for (fold_idx, (train_indices, test_indices)) in splits.iter().enumerate() {
        let y_train = extract_by_indices(y, train_indices);
        let x_train = extract_x_vars_by_indices(x_vars, train_indices);
        let y_test = extract_by_indices(y, test_indices);
        let x_test = extract_x_vars_by_indices(x_vars, test_indices);

        let x_train_matrix = build_design_matrix(&x_train, y_train.len());
        let x_test_matrix = build_design_matrix(&x_test, y_test.len());

        let enet_options = elastic_net::ElasticNetOptions {
            lambda,
            alpha,
            intercept: true,
            standardize,
            max_iter: 100000,
            tol: 1e-7,
            ..Default::default()
        };

        let fit_result = elastic_net::elastic_net_fit(&x_train_matrix, &y_train, &enet_options)?;

        // Predict using preprocess predict function
        let y_test_pred = preprocess::predict(&x_test_matrix, fit_result.intercept, &fit_result.coefficients);
        let y_train_pred = preprocess::predict(&x_train_matrix, fit_result.intercept, &fit_result.coefficients);

        let test_mse = compute_mse(&y_test, &y_test_pred);
        let test_rmse = compute_rmse(&y_test, &y_test_pred);
        let test_mae = compute_mae(&y_test, &y_test_pred);
        let test_r_squared = compute_r_squared(&y_test, &y_test_pred);
        let train_r_squared = compute_r_squared(&y_train, &y_train_pred);

        fold_results.push(FoldResult::new(
            fold_idx + 1,
            train_indices.len(),
            test_indices.len(),
            test_mse,
            test_rmse,
            test_mae,
            test_r_squared,
            train_r_squared,
        ));

        let mut coeffs = vec![fit_result.intercept];
        coeffs.extend(fit_result.coefficients.clone());
        fold_coefficients.push(coeffs);
    }

    Ok(CVResult::from_folds(
        n_samples,
        options.n_folds,
        fold_results,
        fold_coefficients,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_by_indices() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = vec![0, 2, 4];
        let result = extract_by_indices(&data, &indices);
        assert_eq!(result, vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_extract_x_vars_by_indices() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let indices = vec![0, 2, 4];

        let result = extract_x_vars_by_indices(&[x1, x2], &indices);
        assert_eq!(result[0], vec![1.0, 3.0, 5.0]);
        assert_eq!(result[1], vec![2.0, 6.0, 10.0]);
    }

    #[test]
    fn test_predict_ols_simple() {
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let names = vec!["Intercept".into(), "X1".into()];

        let fit = core::ols_regression(&y, &[x], &names).unwrap();

        let x_test = vec![5.0, 6.0];
        let pred = predict_ols(&fit, &[x_test]);

        // y = 2*x, so predictions should be ~10 and ~12
        assert!((pred[0] - 10.0).abs() < 0.1);
        assert!((pred[1] - 12.0).abs() < 0.1);
    }
}
