// ============================================================================
// LOESS (Locally Estimated Scatterplot Smoothing) - Native Types API
// ============================================================================


// ============================================================================
// LOESS Regression
// ============================================================================

/// Fit a LOESS (Locally Estimated Scatterplot Smoothing) regression model.
///
/// LOESS is a non-parametric regression method that fits multiple regressions
/// in local subsets of data to create a smooth curve through the data points.
///
/// Args:
///     y: Response variable values (list or array of floats).
///     x_vars: Predictor variables (list of lists, currently only 1 predictor supported).
///     span: Smoothing parameter (fraction of data used in each local fit).
///            Range: (0, 1]. Default: 0.75. Smaller values = wigglier curves.
///     degree: Degree of local polynomial: 0 (constant), 1 (linear), or 2 (quadratic).
///              Default: 1.
///     robust_iterations: Number of robustness iterations (0 for non-robust fit).
///                        Default: 0. Use 2 for standard robust fitting.
///     surface: Surface computation method. "direct" for exact fits at each point,
///              "interpolate" for faster interpolation from vertex fits.
///              Default: "direct".
///
/// Returns:
///     LoessResult: Object containing fitted values, residuals, and fit statistics.
///
/// Raises:
///     ValueError: If inputs are invalid (empty, mismatched lengths, invalid parameters).
///
/// Example:
///     >>> import linreg_core
///     >>> y = [1.0, 3.5, 4.8, 6.2, 8.5, 11.0]
///     >>> x = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
///     >>> result = linreg_core.loess_fit(y, x, span=0.75)
///     >>> print(result.fitted)
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (y, x_vars, span=0.75, degree=1, robust_iterations=0, surface="direct"))]
fn loess_fit(
    y: &Bound<PyAny>,
    x_vars: &Bound<PyAny>,
    span: f64,
    degree: usize,
    robust_iterations: usize,
    surface: &str,
) -> PyResult<PyLoessResult> {
    use crate::python::types::{extract_f64_sequence, extract_f64_matrix};

    let y_vec = extract_f64_sequence(y)?;
    let x_vars_vec = extract_f64_matrix(x_vars)?;

    // Validate inputs
    if y_vec.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("y cannot be empty"));
    }
    if x_vars_vec.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("x_vars cannot be empty"));
    }
    if x_vars_vec.len() > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "LOESS currently only supports single predictor (1 x variable)"
        ));
    }

    let x_vec = &x_vars_vec[0];

    // Validate x has same length as y
    if x_vec.len() != y_vec.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "x has {} elements, but y has {} elements",
            x_vec.len(),
            y_vec.len()
        )));
    }

    // Validate span
    if span <= 0.0 || span > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "span must be in (0, 1]"
        ));
    }

    // Validate degree
    if degree > 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "degree must be 0, 1, or 2"
        ));
    }

    // Parse surface parameter
    let surface_enum = match surface.to_lowercase().as_str() {
        "interpolate" => crate::loess::LoessSurface::Interpolate,
        "direct" | _ => crate::loess::LoessSurface::Direct,
    };

    // Build LOESS options
    let options = crate::loess::LoessOptions {
        span,
        degree,
        robust_iterations,
        n_predictors: 1,
        surface: surface_enum,
    };

    // Fit LOESS model
    let result = crate::loess::loess_fit(&y_vec, &x_vars_vec, &options)
        .map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;

    // Compute residuals
    let residuals: Vec<f64> = y_vec.iter()
        .zip(result.fitted.iter())
        .map(|(y_i, f_i)| y_i - f_i)
        .collect();

    // Compute MSE and RMSE
    let n = y_vec.len();
    let mse = residuals.iter()
        .map(|r| r * r)
        .sum::<f64>() / n as f64;
    let rmse = mse.sqrt();

    Ok(PyLoessResult {
        fitted: result.fitted,
        residuals,
        span: result.span,
        degree: result.degree,
        robust_iterations: result.robust_iterations,
        surface: result.surface.as_str().to_string(),
        mse,
        rmse,
        n_observations: n,
    })
}

// ============================================================================
// LOESS Prediction
// ============================================================================

/// Predict at new points using a LOESS model.
///
/// Performs LOESS prediction at arbitrary new points by redoing the local
/// fitting at each query point using the original training data.
///
/// Args:
///     new_x: New x values to predict at (list or array of floats).
///     original_x: Original training predictor values (list or array of floats).
///     original_y: Original training response values (list or array of floats).
///     span: Smoothing parameter used for the original fit. Default: 0.75.
///     degree: Polynomial degree used for the original fit. Default: 1.
///     robust_iterations: Robustness iterations used for the original fit.
///                        Default: 0.
///     surface: Surface computation method used for the original fit.
///              Default: "direct".
///
/// Returns:
///     List of predicted y values at the new x points.
///
/// Raises:
///     ValueError: If inputs are invalid or parameters don't match original fit.
///
/// Example:
///     >>> import linreg_core
///     >>> train_x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
///     >>> train_y = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
///     >>> new_x = [1.5, 2.5, 3.5]
///     >>> predictions = linreg_core.loess_predict(new_x, train_x, train_y, span=0.75)
///     >>> print(predictions)
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (new_x, original_x, original_y, span=0.75, degree=1, robust_iterations=0, surface="direct"))]
fn loess_predict(
    new_x: &Bound<PyAny>,
    original_x: &Bound<PyAny>,
    original_y: &Bound<PyAny>,
    span: f64,
    degree: usize,
    robust_iterations: usize,
    surface: &str,
) -> PyResult<Vec<f64>> {
    use crate::python::types::extract_f64_sequence;

    let new_x_vec = extract_f64_sequence(new_x)?;
    let orig_x_vec = extract_f64_sequence(original_x)?;
    let orig_y_vec = extract_f64_sequence(original_y)?;

    // Validate inputs
    if orig_x_vec.is_empty() || orig_y_vec.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "original_x and original_y cannot be empty"
        ));
    }
    if orig_x_vec.len() != orig_y_vec.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "original_x has {} elements, but original_y has {} elements",
            orig_x_vec.len(),
            orig_y_vec.len()
        )));
    }

    // Validate span
    if span <= 0.0 || span > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "span must be in (0, 1]"
        ));
    }

    // Validate degree
    if degree > 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "degree must be 0, 1, or 2"
        ));
    }

    // Parse surface parameter
    let surface_enum = match surface.to_lowercase().as_str() {
        "interpolate" => crate::loess::LoessSurface::Interpolate,
        "direct" | _ => crate::loess::LoessSurface::Direct,
    };

    // Build LOESS options
    let options = crate::loess::LoessOptions {
        span,
        degree,
        robust_iterations,
        n_predictors: 1,
        surface: surface_enum,
    };

    // Build new_x as a vector of vectors for the predict API
    let new_x_nested: Vec<Vec<f64>> = vec![new_x_vec.clone()];

    // Build original_x as a vector of vectors for the predict API
    let original_x_nested: Vec<Vec<f64>> = vec![orig_x_vec];

    // Fit LOESS model on original data (needed to get LoessFit)
    let fit_result = crate::loess::loess_fit(&orig_y_vec, &original_x_nested, &options)
        .map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;

    // Predict at new points using the LoessFit::predict method
    let predictions = fit_result.predict(&new_x_nested, &original_x_nested, &orig_y_vec, &options)
        .map_err(|e| pyo3::PyErr::from(crate::python::error::PythonError::from(e)))?;

    Ok(predictions)
}
