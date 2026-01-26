// ============================================================================
// RESET Test (Regression Specification Error Test)
// ============================================================================
//
// H0: The model is correctly specified (no omitted variables/transformations)
// H1: The model is misspecified (omitted variables or non-linear terms needed)
//
// Ramsey's RESET test works by:
// 1. Fitting the original OLS model and computing fitted values
// 2. Adding powers of fitted values (or regressors, or first PC) to the model
// 3. Comparing the original vs augmented models via an F-test
//
// The test statistic is: F = (df2/df1) * ((RSS_restricted - RSS_unrestricted) / RSS_unrestricted)
// where df1 = number of additional terms, df2 = n - k - q

use super::helpers::{compute_rss, f_p_value, fit_ols};
use super::types::DiagnosticTestResult;
use crate::error::{Error, Result};
use crate::linalg::Matrix;

/// RESET test type specification.
///
/// Determines what terms to add for the RESET test. The original RESET test
/// uses powers of fitted values, but variants use powers of regressors or
/// the first principal component.
///
/// # Variants
///
/// * `Fitted` - Powers of fitted values (original Ramsey RESET, default in R)
/// * `Regressor` - Powers of each regressor variable
/// * `PrincipalComponent` - Powers of first principal component of regressors
///
/// # Example
///
/// ```
/// use linreg_core::diagnostics::ResetType;
///
/// let fitted = ResetType::Fitted;
/// let regressor = ResetType::Regressor;
/// let pc = ResetType::PrincipalComponent;
///
/// // Enum variants can be compared
/// assert_eq!(fitted, ResetType::Fitted);
/// assert_ne!(fitted, regressor);
/// assert_eq!(fitted.as_str(), "fitted");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResetType {
    /// Powers of fitted values (ŷ², ŷ³, ...) - original Ramsey RESET test
    Fitted,
    /// Powers of each regressor variable
    Regressor,
    /// Powers of first principal component of regressor matrix
    PrincipalComponent,
}

impl ResetType {
    /// Convert to string matching R's resettest type parameter
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Fitted => "fitted",
            Self::Regressor => "regressor",
            Self::PrincipalComponent => "princomp",
        }
    }
}

/// Performs Ramsey's RESET test for functional form misspecification.
///
/// The RESET (Regression Specification Error Test) test checks whether the model
/// is correctly specified by testing if additional terms (powers of fitted values,
/// regressors, or first principal component) significantly improve the model fit.
///
/// **Null hypothesis (H0):** The model is correctly specified (no omitted non-linear terms)
///
/// **Alternative hypothesis (H1):** The model is misspecified (additional terms needed)
///
/// # Arguments
///
/// * `y` - Dependent variable values (n observations)
/// * `x_vars` - Independent variables (each vec is a column, k predictors)
/// * `powers` - Powers to use for additional terms (e.g., `[2, 3]` for ŷ², ŷ³)
/// * `type_` - Type of terms to add ([`ResetType`])
///
/// # Returns
///
/// A [`DiagnosticTestResult`] containing the F-statistic, p-value, and interpretation.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if:
/// - n ≤ k + q + 1 (where q = number of additional terms)
/// - Any power value is ≤ 1
/// - powers array is empty
///
/// # Examples
///
/// ```ignore
/// use linreg_core::diagnostics::reset_test;
/// use linreg_core::diagnostics::ResetType;
///
/// let y = vec![2.5, 3.7, 4.2, 5.1, 6.3, 7.0];
/// let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let x2 = vec![2.0, 4.0, 5.0, 4.0, 3.0, 2.0];
///
/// // Standard RESET test with fitted values squared and cubed
/// let result = reset_test(&y, &[x1, x2], &[2, 3], ResetType::Fitted)?;
/// # Ok::<(), linreg_core::Error>(())
/// ```
///
/// # Algorithm
///
/// The test follows R's `lmtest::resettest` algorithm:
///
/// 1. **Fit restricted model:** y = Xβ + ε, compute RSS₁
/// 2. **Generate Z matrix:** powers of fitted values/regressors/PC
/// 3. **Fit unrestricted model:** y = Xβ + Zγ + ε, compute RSS₂
/// 4. **Compute F-statistic:** F = (df₂/df₁) × ((RSS₁ - RSS₂) / RSS₂)
///    - df₁ = q (number of additional terms)
///    - df₂ = n - k - q
///
/// # References
///
/// - Ramsey, J.B. (1969). "Tests for Specification Errors in Classical Linear
///   Least-Squares Regression Analysis". *Journal of the Royal Statistical
///   Society, Series B* 31: 350–371.
/// - R documentation: `?lmtest::resettest`
#[allow(clippy::similar_names)]
pub fn reset_test(
    y: &[f64],
    x_vars: &[Vec<f64>],
    powers: &[usize],
    type_: ResetType,
) -> Result<DiagnosticTestResult> {
    let n = y.len();
    let k = x_vars.len(); // number of non-intercept regressors
    let p = k + 1; // number of columns in design matrix including intercept

    // Validate inputs
    if powers.is_empty() {
        return Err(Error::InvalidInput(
            "Powers array cannot be empty".to_string(),
        ));
    }

    for &power in powers {
        if power <= 1 {
            return Err(Error::InvalidInput(format!(
                "All powers must be greater than 1, got {power}"
            )));
        }
    }

    // Validate dimensions and finite values using shared helper
    super::helpers::validate_regression_data(y, x_vars)?;

    // Create design matrix (with intercept)
    let mut x_data = vec![0.0; n * p];
    for row in 0..n {
        x_data[row * p] = 1.0; // intercept
        for (col, x_var) in x_vars.iter().enumerate() {
            x_data[row * p + col + 1] = x_var[row];
        }
    }
    let x = Matrix::new(n, p, x_data);

    // ========================================================================
    // Step 1: Fit restricted model (original model) and get RSS
    // ========================================================================
    let beta_restricted = fit_ols(y, &x)?;
    let rss_restricted = compute_rss(y, &x, &beta_restricted)?;

    // ========================================================================
    // Step 2: Generate Z matrix (additional terms)
    // ========================================================================
    let z = generate_z_matrix(y, x_vars, &x, powers, type_, &beta_restricted)?;

    // Number of additional terms
    let q = z.cols;

    // Check we have enough degrees of freedom
    if n <= p + q {
        return Err(Error::InsufficientData {
            required: p + q + 1,
            available: n,
        });
    }

    // ========================================================================
    // Step 3: Fit unrestricted model (original + Z terms) and get RSS
    // ========================================================================
    // Create augmented design matrix [X | Z]
    let xz_data = augment_x_with_z(&x, &z);
    let xz = Matrix::new(n, p + q, xz_data);

    let beta_unrestricted = fit_ols(y, &xz)?;
    let rss_unrestricted = compute_rss(y, &xz, &beta_unrestricted)?;

    // ========================================================================
    // Step 4: Compute F-statistic and p-value
    // ========================================================================
    // F = (df2/df1) * ((RSS1 - RSS2) / RSS2)
    // where df1 = q, df2 = n - k - q
    let df1 = q as f64;
    let df2 = (n - p - q) as f64;

    let f_stat = (df2 / df1) * ((rss_restricted - rss_unrestricted) / rss_unrestricted);

    // Handle numerical edge cases
    let f_stat = if !f_stat.is_finite() || f_stat < 0.0 {
        0.0
    } else {
        f_stat
    };

    let p_value = f_p_value(f_stat, df1, df2);

    // Standard interpretation
    let alpha = 0.05;
    let passed = p_value > alpha;

    let type_str = type_.as_str();
    let powers_str = powers
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let (interpretation, guidance) = if passed {
        (
            format!(
                "p-value = {:.4} is greater than {:.2}. Cannot reject H0. No significant evidence of model misspecification.",
                p_value, alpha
            ),
            "The current model specification appears adequate. Consider keeping the current functional form.",
        )
    } else {
        (
            format!(
                "p-value = {:.4} is less than or equal to {:.2}. Reject H0. Significant evidence of model misspecification.",
                p_value, alpha
            ),
            "Consider adding polynomial terms, transforming variables, including omitted predictors, or using non-linear modeling.",
        )
    };

    Ok(DiagnosticTestResult {
        test_name: format!(
            "RESET Test (type={}, powers=[{}])",
            type_str, powers_str
        ),
        statistic: f_stat,
        p_value,
        passed,
        interpretation,
        guidance: guidance.to_string(),
    })
}

/// Generates the Z matrix of additional terms for the RESET test.
///
/// The Z matrix contains powers of:
/// - fitted values (ŷ², ŷ³, ...) for `ResetType::Fitted`
/// - each regressor for `ResetType::Regressor` (e.g., x₁², x₁³, x₂², x₂³, ...)
/// - first principal component for `ResetType::PrincipalComponent`
///
/// # Arguments
///
/// * `y` - Response variable (for fitting model to get ŷ in `Fitted` mode)
/// * `x_vars` - Predictor variables
/// * `x` - Design matrix (with intercept)
/// * `powers` - Powers to compute (e.g., `[2, 3]` for squared and cubed)
/// * `type_` - Type of terms to generate
/// * `beta` - Coefficients from restricted model (used for `Fitted` type)
///
/// # Returns
///
/// A matrix with n rows and q columns, where q depends on the type:
/// - `Fitted`: q = len(powers)
/// - `Regressor`: q = len(powers) × len(x_vars)
/// - `PrincipalComponent`: q = len(powers)
#[allow(clippy::needless_range_loop)]
fn generate_z_matrix(
    y: &[f64],
    x_vars: &[Vec<f64>],
    x: &Matrix,
    powers: &[usize],
    type_: ResetType,
    beta: &[f64],
) -> Result<Matrix> {
    let n = y.len();
    let k = x_vars.len();

    match type_ {
        ResetType::Fitted => {
            // Powers of fitted values: ŷ², ŷ³, ...
            let fitted = x.mul_vec(beta);
            let mut z_data = Vec::with_capacity(n * powers.len());

            for &power in powers {
                for &f in &fitted {
                    z_data.push(f.powi(power as i32));
                }
            }

            // Transpose to get n rows × len(powers) cols (column-major to row-major)
            let mut z_transposed = vec![0.0; n * powers.len()];
            for row in 0..n {
                for (col, &power) in powers.iter().enumerate() {
                    z_transposed[row * powers.len() + col] = fitted[row].powi(power as i32);
                }
            }
            Ok(Matrix::new(n, powers.len(), z_transposed))
        }

        ResetType::Regressor => {
            // Powers of each regressor: x₁², x₁³, x₂², x₂³, ...
            let q = powers.len() * k;
            let mut z_data = vec![0.0; n * q];

            for (var_idx, x_var) in x_vars.iter().enumerate() {
                for (power_idx, &power) in powers.iter().enumerate() {
                    let col_idx = var_idx * powers.len() + power_idx;
                    for (row_idx, &x_val) in x_var.iter().enumerate() {
                        z_data[row_idx * q + col_idx] = x_val.powi(power as i32);
                    }
                }
            }
            Ok(Matrix::new(n, q, z_data))
        }

        ResetType::PrincipalComponent => {
            // First principal component of regressor matrix
            // Create regressor matrix (without intercept)
            let mut x_reg_data = vec![0.0; n * k];
            for row in 0..n {
                for (col, x_var) in x_vars.iter().enumerate() {
                    x_reg_data[row * k + col] = x_var[row];
                }
            }
            let x_reg = Matrix::new(n, k, x_reg_data.clone());

            // Compute covariance matrix: (X'X - n*mean*mean') / (n-1)
            // But since we center the data, it's simpler: cov(X) = (X_centered'X_centered) / (n-1)
            let mut x_centered_data = x_reg_data;

            // Center each column
            for col in 0..k {
                let col_mean = (0..n)
                    .map(|row| x_reg.get(row, col))
                    .sum::<f64>()
                    / n as f64;
                for row in 0..n {
                    x_centered_data[row * k + col] -= col_mean;
                }
            }

            // Compute X'X (proportional to covariance)
            let x_centered = Matrix::new(n, k, x_centered_data);
            let xt_x = x_centered.transpose().matmul(&x_centered);

            // Find first principal component (eigenvector of largest eigenvalue)
            // Using power iteration
            let pc1 = first_principal_component(&xt_x, k)?;

            // Compute PC1 scores: X * pc1
            let mut pc1_scores = vec![0.0; n];
            for row in 0..n {
                let mut sum = 0.0;
                for col in 0..k {
                    sum += x_reg.get(row, col) * pc1[col];
                }
                pc1_scores[row] = sum;
            }

            // Powers of PC1 scores
            let mut z_data = vec![0.0; n * powers.len()];
            for (power_idx, &power) in powers.iter().enumerate() {
                for row in 0..n {
                    z_data[row * powers.len() + power_idx] = pc1_scores[row].powi(power as i32);
                }
            }
            Ok(Matrix::new(n, powers.len(), z_data))
        }
    }
}

/// Computes the first principal component using power iteration.
///
/// # Arguments
///
/// * `matrix` - Symmetric matrix (e.g., X'X covariance-like matrix)
/// * `k` - Dimension of the matrix
///
/// # Returns
///
/// A vector of length k representing the eigenvector corresponding to the
/// largest eigenvalue.
#[allow(clippy::needless_range_loop)]
fn first_principal_component(matrix: &Matrix, k: usize) -> Result<Vec<f64>> {
    // Power iteration to find dominant eigenvector
    let mut v = vec![1.0 / k as f64; k]; // Initial guess (normalized)
    let max_iter = 1000;
    let tolerance = 1e-10;

    for _ in 0..max_iter {
        let v_old = v.clone();
        // v = matrix * v
        v = matrix.mul_vec(&v);

        // Normalize
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            return Err(Error::SingularMatrix);
        }
        for val in &mut v {
            *val /= norm;
        }

        // Check convergence
        let diff: f64 = v
            .iter()
            .zip(&v_old)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        if diff < tolerance {
            break;
        }
    }

    Ok(v)
}

/// Augments the design matrix X with Z matrix by column-binding.
///
/// Creates [X | Z] in row-major storage.
///
/// # Arguments
///
/// * `x` - Original design matrix (n × p)
/// * `z` - Additional terms matrix (n × q)
///
/// # Returns
///
/// A flat vector of length n × (p + q) containing [X | Z] in row-major order.
fn augment_x_with_z(x: &Matrix, z: &Matrix) -> Vec<f64> {
    let n = x.rows;
    let p = x.cols;
    let q = z.cols;

    let mut xz_data = vec![0.0; n * (p + q)];

    for row in 0..n {
        // Copy X row
        for col in 0..p {
            xz_data[row * (p + q) + col] = x.get(row, col);
        }
        // Copy Z row
        for col in 0..q {
            xz_data[row * (p + q) + p + col] = z.get(row, col);
        }
    }

    xz_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset_test_fitted() {
        // Simple linear relationship - should pass
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + 2.0 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = reset_test(&y, &[x], &[2, 3], ResetType::Fitted).unwrap();
        // For a perfectly linear relationship, p-value should be large (fail to reject H0)
        assert!(result.p_value > 0.01);
    }

    #[test]
    fn test_reset_test_quadratic() {
        // Quadratic relationship - should detect misspecification
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + i as f64 + 0.1 * i as f64 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = reset_test(&y, &[x], &[2, 3], ResetType::Fitted).unwrap();
        // For a quadratic relationship fit with linear model, p-value should be small
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_reset_invalid_powers() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 2.0, 3.0];

        // Power of 1 should fail
        let result = reset_test(&y, &[x.clone()], &[1], ResetType::Fitted);
        assert!(result.is_err());

        // Empty powers should fail
        let result = reset_test(&y, &[x], &[], ResetType::Fitted);
        assert!(result.is_err());
    }

    #[test]
    fn test_first_principal_component() {
        // Simple 2x2 identity matrix - PC1 should be [1/sqrt(2), 1/sqrt(2)]
        let m = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let pc1 = first_principal_component(&m, 2).unwrap();

        // PC1 should be approximately normalized
        let norm: f64 = pc1.iter().map(|&x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_reset_type_as_str() {
        assert_eq!(ResetType::Fitted.as_str(), "fitted");
        assert_eq!(ResetType::Regressor.as_str(), "regressor");
        assert_eq!(ResetType::PrincipalComponent.as_str(), "princomp");
    }

    #[test]
    fn test_reset_insufficient_data_for_additional_terms() {
        // Small dataset that can handle original model but not augmented
        let y = vec![1.0, 2.0, 3.0];
        let x1 = vec![1.0, 2.0, 3.0];
        let x2 = vec![2.0, 4.0, 6.0];

        // Original model: n=3, k=2, p=3 (including intercept)
        // With powers=[2,3,4], q=3, so we need n > p + q = 6, but n=3
        let result = reset_test(&y, &[x1, x2], &[2, 3, 4], ResetType::Fitted);
        assert!(result.is_err());

        if let Err(Error::InsufficientData { required, available }) = result {
            assert_eq!(required, 7); // p + q + 1 = 3 + 3 + 1
            assert_eq!(available, 3);
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_reset_regressor_type() {
        // Linear relationship - should pass
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + 2.0 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = reset_test(&y, &[x.clone()], &[2, 3], ResetType::Regressor).unwrap();
        // For linear relationship, p-value should be large
        assert!(result.p_value > 0.01);
        assert!(result.test_name.contains("regressor"));
    }

    #[test]
    fn test_reset_regressor_type_quadratic() {
        // Quadratic relationship - Regressor type should detect misspecification
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + i as f64 + 0.1 * i as f64 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = reset_test(&y, &[x], &[2, 3], ResetType::Regressor).unwrap();
        // For quadratic relationship with linear model, p-value should be small
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_reset_regressor_multiple_predictors() {
        // Multiple predictors with Regressor type
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + 2.0 * i as f64 + 0.5 * (i as f64 / 2.0)).collect();
        let x1: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let x2: Vec<f64> = (1..=30).map(|i| i as f64 / 2.0).collect();

        let result = reset_test(&y, &[x1, x2], &[2], ResetType::Regressor).unwrap();
        // Should pass for correctly specified model
        assert!(result.p_value > 0.01);
    }

    #[test]
    fn test_reset_principal_component_type() {
        // Linear relationship - should pass
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + 2.0 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = reset_test(&y, &[x.clone()], &[2, 3], ResetType::PrincipalComponent).unwrap();
        // For linear relationship, p-value should be large
        assert!(result.p_value > 0.01);
        assert!(result.test_name.contains("princomp"));
    }

    #[test]
    fn test_reset_principal_component_quadratic() {
        // Quadratic relationship - PC type should detect misspecification
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + i as f64 + 0.1 * i as f64 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = reset_test(&y, &[x], &[2, 3], ResetType::PrincipalComponent).unwrap();
        // For quadratic relationship, p-value should be small
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_reset_principal_component_multiple_predictors() {
        // Multiple predictors with PC type
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + 2.0 * i as f64 + 0.5 * (i as f64 / 2.0)).collect();
        let x1: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let x2: Vec<f64> = (1..=30).map(|i| i as f64 / 2.0).collect();

        let result = reset_test(&y, &[x1, x2], &[2], ResetType::PrincipalComponent).unwrap();
        // Should pass for correctly specified model
        assert!(result.p_value > 0.01);
    }

    #[test]
    fn test_reset_single_power() {
        // Test with single power (not 2, 3)
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + 2.0 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        // All three types with single power
        let result_fitted = reset_test(&y, &[x.clone()], &[4], ResetType::Fitted).unwrap();
        assert!(result_fitted.p_value > 0.01);

        let result_regressor = reset_test(&y, &[x.clone()], &[4], ResetType::Regressor).unwrap();
        assert!(result_regressor.p_value > 0.01);

        let result_pc = reset_test(&y, &[x], &[4], ResetType::PrincipalComponent).unwrap();
        assert!(result_pc.p_value > 0.01);
    }

    #[test]
    fn test_reset_higher_powers() {
        // Test with higher powers
        let y: Vec<f64> = (1..=50).map(|i| 1.0 + 2.0 * i as f64 + 0.01 * i as f64 * i as f64).collect();
        let x: Vec<f64> = (1..=50).map(|i| i as f64).collect();

        let result = reset_test(&y, &[x], &[3, 4, 5], ResetType::Fitted).unwrap();
        // Should detect the slight non-linearity with higher powers
        assert!(result.p_value < 0.1);
    }

    #[test]
    fn test_first_principal_component_singular_matrix() {
        // Create a near-zero matrix (should return SingularMatrix error)
        let m = Matrix::new(2, 2, vec![1e-20, 0.0, 0.0, 1e-20]);
        let result = first_principal_component(&m, 2);

        assert!(result.is_err());
        if let Err(Error::SingularMatrix) = result {
            // Expected error
        } else {
            panic!("Expected SingularMatrix error");
        }
    }

    #[test]
    fn test_reset_all_three_types_consistent() {
        // For a simple linear model, all three types should give similar results (pass)
        let y: Vec<f64> = (1..=40).map(|i| 1.0 + 2.0 * i as f64).collect();
        let x: Vec<f64> = (1..=40).map(|i| i as f64).collect();

        let result_fitted = reset_test(&y, &[x.clone()], &[2, 3], ResetType::Fitted).unwrap();
        let result_regressor = reset_test(&y, &[x.clone()], &[2, 3], ResetType::Regressor).unwrap();
        let result_pc = reset_test(&y, &[x], &[2, 3], ResetType::PrincipalComponent).unwrap();

        // All should pass (large p-values) for correctly specified linear model
        assert!(result_fitted.p_value > 0.05);
        assert!(result_regressor.p_value > 0.05);
        assert!(result_pc.p_value > 0.05);
    }

    #[test]
    fn test_reset_interpretation_and_guidance() {
        let y: Vec<f64> = (1..=30).map(|i| 1.0 + 2.0 * i as f64).collect();
        let x: Vec<f64> = (1..=30).map(|i| i as f64).collect();

        let result = reset_test(&y, &[x], &[2, 3], ResetType::Fitted).unwrap();

        // Check interpretation contains expected phrases
        assert!(result.interpretation.contains("p-value"));
        assert!(result.guidance.contains("adequate") || result.guidance.contains("polynomial"));

        // Test name should include type and powers
        assert!(result.test_name.contains("fitted"));
        assert!(result.test_name.contains("2"));
        assert!(result.test_name.contains("3"));
    }

    #[test]
    fn test_reset_boundary_power_values() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Power of 0 should fail
        let result = reset_test(&y, &[x.clone()], &[0], ResetType::Fitted);
        assert!(result.is_err());

        // Power of 1 should fail (must be > 1)
        let result = reset_test(&y, &[x.clone()], &[1], ResetType::Fitted);
        assert!(result.is_err());

        // Valid power should work
        let result = reset_test(&y, &[x], &[2], ResetType::Fitted);
        assert!(result.is_ok());
    }
}
