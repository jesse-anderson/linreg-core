// ============================================================================
// Utility Result Classes for Python Bindings
// ============================================================================
// LambdaPathResult, CSVResult, FoldResult, CVResult, PredictionIntervalResult

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ============================================================================
// LambdaPathResult - Lambda path generation results
// ============================================================================

/// Result class for lambda path generation.
#[cfg(feature = "python")]
#[pyclass(name = "LambdaPathResult")]
pub struct PyLambdaPathResult {
    /// Generated lambda sequence (decreasing order)
    #[pyo3(get, set)]
    pub lambda_path: Vec<f64>,

    /// Maximum lambda value
    #[pyo3(get, set)]
    pub lambda_max: f64,

    /// Minimum lambda value
    #[pyo3(get, set)]
    pub lambda_min: f64,

    /// Number of lambda values
    #[pyo3(get, set)]
    pub n_lambda: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyLambdaPathResult {
    #[new]
    fn new(lambda_path: Vec<f64>, lambda_max: f64, lambda_min: f64, n_lambda: usize) -> Self {
        Self {
            lambda_path,
            lambda_max,
            lambda_min,
            n_lambda,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Lambda Path Results\n\
             ===================\n\
             Lambda max: {:.6}\n\
             Lambda min: {:.6}\n\
             Number of values: {}",
            self.lambda_max, self.lambda_min, self.n_lambda
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("lambda_path", &self.lambda_path)?;
        dict.set_item("lambda_max", self.lambda_max)?;
        dict.set_item("lambda_min", self.lambda_min)?;
        dict.set_item("n_lambda", self.n_lambda)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "LambdaPathResult(n={}, max={:.2e}, min={:.2e})",
            self.n_lambda, self.lambda_max, self.lambda_min
        )
    }
}

// ============================================================================
// CSVResult - CSV parsing result
// ============================================================================

/// Result class for CSV parsing.
#[cfg(feature = "python")]
#[pyclass(name = "CSVResult")]
pub struct PyCSVResult {
    /// Column headers from the CSV
    #[pyo3(get, set)]
    pub headers: Vec<String>,

    /// Parsed data as Python object (list of dicts)
    #[pyo3(get)]
    pub data: PyObject,

    /// Names of columns that contain numeric data
    #[pyo3(get, set)]
    pub numeric_columns: Vec<String>,

    /// Number of rows parsed
    #[pyo3(get, set)]
    pub n_rows: usize,

    /// Number of columns parsed
    #[pyo3(get, set)]
    pub n_cols: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCSVResult {
    #[new]
    fn new(
        headers: Vec<String>,
        data: PyObject,
        numeric_columns: Vec<String>,
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        Self {
            headers,
            data,
            numeric_columns,
            n_rows,
            n_cols,
        }
    }

    /// Get the parsed data as a Python list of dicts.
    fn get_data(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.data.clone_ref(py))
    }

    fn summary(&self) -> String {
        format!(
            "CSV Parsing Results\n\
             ====================\n\
             Rows: {}\n\
             Columns: {}\n\
             Headers: {:?}\n\
             Numeric columns: {:?}",
            self.n_rows, self.n_cols, self.headers, self.numeric_columns
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        use pyo3::types::PyList;
        let dict = PyDict::new_bound(py);

        // Set headers as Python list
        let headers_list = PyList::new_bound(py, &self.headers);
        dict.set_item("headers", headers_list)?;

        // Set numeric_columns as Python list
        let numeric_list = PyList::new_bound(py, &self.numeric_columns);
        dict.set_item("numeric_columns", numeric_list)?;

        // Set n_rows and n_cols
        dict.set_item("n_rows", self.n_rows)?;
        dict.set_item("n_cols", self.n_cols)?;

        // Set data (already a Python object)
        dict.set_item("data", self.data.clone_ref(py))?;

        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "CSVResult(n_rows={}, n_cols={}, numeric_cols={})",
            self.n_rows, self.n_cols, self.numeric_columns.len()
        )
    }
}

// ============================================================================
// FoldResult - Single fold result from cross-validation
// ============================================================================

/// Result class for a single fold in cross-validation.
#[cfg(feature = "python")]
#[pyclass(name = "FoldResult")]
pub struct PyFoldResult {
    /// Fold index (1-based)
    #[pyo3(get, set)]
    pub fold_index: usize,

    /// Number of training observations
    #[pyo3(get, set)]
    pub train_size: usize,

    /// Number of test observations
    #[pyo3(get, set)]
    pub test_size: usize,

    /// Mean Squared Error on test set
    #[pyo3(get, set)]
    pub mse: f64,

    /// Root Mean Squared Error on test set
    #[pyo3(get, set)]
    pub rmse: f64,

    /// Mean Absolute Error on test set
    #[pyo3(get, set)]
    pub mae: f64,

    /// R-squared on test set
    #[pyo3(get, set)]
    pub r_squared: f64,

    /// Training R-squared (for detecting overfitting)
    #[pyo3(get, set)]
    pub train_r_squared: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFoldResult {
    #[new]
    fn new(
        fold_index: usize,
        train_size: usize,
        test_size: usize,
        mse: f64,
        rmse: f64,
        mae: f64,
        r_squared: f64,
        train_r_squared: f64,
    ) -> Self {
        Self {
            fold_index,
            train_size,
            test_size,
            mse,
            rmse,
            mae,
            r_squared,
            train_r_squared,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Fold {}: train={}, test={}, RMSE={:.4}, R²={:.4}",
            self.fold_index, self.train_size, self.test_size, self.rmse, self.r_squared
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("fold_index", self.fold_index)?;
        dict.set_item("train_size", self.train_size)?;
        dict.set_item("test_size", self.test_size)?;
        dict.set_item("mse", self.mse)?;
        dict.set_item("rmse", self.rmse)?;
        dict.set_item("mae", self.mae)?;
        dict.set_item("r_squared", self.r_squared)?;
        dict.set_item("train_r_squared", self.train_r_squared)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "FoldResult(fold={}, rmse={:.4}, r2={:.4})",
            self.fold_index, self.rmse, self.r_squared
        )
    }
}

// ============================================================================
// CVResult - Aggregated cross-validation results
// ============================================================================

/// Result class for aggregated K-Fold cross-validation results.
#[cfg(feature = "python")]
#[pyclass(name = "CVResult")]
pub struct PyCVResult {
    /// Number of folds used
    #[pyo3(get, set)]
    pub n_folds: usize,

    /// Total number of observations
    #[pyo3(get, set)]
    pub n_samples: usize,

    /// Mean MSE across all folds
    #[pyo3(get, set)]
    pub mean_mse: f64,

    /// Standard deviation of MSE across folds
    #[pyo3(get, set)]
    pub std_mse: f64,

    /// Mean RMSE across all folds
    #[pyo3(get, set)]
    pub mean_rmse: f64,

    /// Standard deviation of RMSE across folds
    #[pyo3(get, set)]
    pub std_rmse: f64,

    /// Mean MAE across all folds
    #[pyo3(get, set)]
    pub mean_mae: f64,

    /// Standard deviation of MAE across folds
    #[pyo3(get, set)]
    pub std_mae: f64,

    /// Mean R-squared across all folds
    #[pyo3(get, set)]
    pub mean_r_squared: f64,

    /// Standard deviation of R-squared across folds
    #[pyo3(get, set)]
    pub std_r_squared: f64,

    /// Mean training R-squared (for overfitting detection)
    #[pyo3(get, set)]
    pub mean_train_r_squared: f64,

    /// Per-fold results as a Python list of FoldResult objects
    #[pyo3(get)]
    pub fold_results: PyObject,

    /// Coefficient estimates from each fold (for stability analysis)
    #[pyo3(get, set)]
    pub fold_coefficients: Vec<Vec<f64>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCVResult {
    // No #[new] - constructed from Rust CV functions

    fn summary(&self) -> String {
        format!(
            "K-Fold Cross Validation Results\n\
             ================================\n\
             Folds: {}\n\
             Observations: {}\n\
             Mean RMSE: {:.4} (+/- {:.4})\n\
             Mean MAE:  {:.4} (+/- {:.4})\n\
             Mean R²:   {:.4} (+/- {:.4})\n\
             Mean Train R²: {:.4}",
            self.n_folds,
            self.n_samples,
            self.mean_rmse,
            self.std_rmse,
            self.mean_mae,
            self.std_mae,
            self.mean_r_squared,
            self.std_r_squared,
            self.mean_train_r_squared
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("n_folds", self.n_folds)?;
        dict.set_item("n_samples", self.n_samples)?;
        dict.set_item("mean_mse", self.mean_mse)?;
        dict.set_item("std_mse", self.std_mse)?;
        dict.set_item("mean_rmse", self.mean_rmse)?;
        dict.set_item("std_rmse", self.std_rmse)?;
        dict.set_item("mean_mae", self.mean_mae)?;
        dict.set_item("std_mae", self.std_mae)?;
        dict.set_item("mean_r_squared", self.mean_r_squared)?;
        dict.set_item("std_r_squared", self.std_r_squared)?;
        dict.set_item("mean_train_r_squared", self.mean_train_r_squared)?;
        dict.set_item("fold_results", &self.fold_results)?;
        dict.set_item("fold_coefficients", &self.fold_coefficients)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "CVResult(folds={}, rmse={:.4}+/-{:.4}, r2={:.4})",
            self.n_folds, self.mean_rmse, self.std_rmse, self.mean_r_squared
        )
    }
}

// ============================================================================
// PredictionIntervalResult - Prediction interval output
// ============================================================================

/// Result class for prediction interval computation.
#[cfg(feature = "python")]
#[pyclass(name = "PredictionIntervalResult")]
pub struct PyPredictionIntervalResult {
    /// Point predictions (fitted values) for new observations
    #[pyo3(get, set)]
    pub predicted: Vec<f64>,

    /// Lower bounds of prediction intervals
    #[pyo3(get, set)]
    pub lower_bound: Vec<f64>,

    /// Upper bounds of prediction intervals
    #[pyo3(get, set)]
    pub upper_bound: Vec<f64>,

    /// Standard errors for predictions
    #[pyo3(get, set)]
    pub se_pred: Vec<f64>,

    /// Leverage values for the new observations
    #[pyo3(get, set)]
    pub leverage: Vec<f64>,

    /// Significance level used
    #[pyo3(get, set)]
    pub alpha: f64,

    /// Residual degrees of freedom from the fitted model
    #[pyo3(get, set)]
    pub df_residuals: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPredictionIntervalResult {
    #[new]
    fn new(
        predicted: Vec<f64>,
        lower_bound: Vec<f64>,
        upper_bound: Vec<f64>,
        se_pred: Vec<f64>,
        leverage: Vec<f64>,
        alpha: f64,
        df_residuals: f64,
    ) -> Self {
        Self {
            predicted,
            lower_bound,
            upper_bound,
            se_pred,
            leverage,
            alpha,
            df_residuals,
        }
    }

    fn summary(&self) -> String {
        format!(
            "Prediction Intervals\n\
             ====================\n\
             Alpha: {:.4}\n\
             Df residuals: {:.1}\n\
             Number of predictions: {}",
            self.alpha, self.df_residuals, self.predicted.len()
        )
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let dict = PyDict::new_bound(py);
        dict.set_item("predicted", &self.predicted)?;
        dict.set_item("lower_bound", &self.lower_bound)?;
        dict.set_item("upper_bound", &self.upper_bound)?;
        dict.set_item("se_pred", &self.se_pred)?;
        dict.set_item("leverage", &self.leverage)?;
        dict.set_item("alpha", self.alpha)?;
        dict.set_item("df_residuals", self.df_residuals)?;
        Ok(dict.into())
    }

    fn __str__(&self) -> String {
        self.summary()
    }

    fn __repr__(&self) -> String {
        format!(
            "PredictionIntervalResult(n={}, alpha={:.4})",
            self.predicted.len(), self.alpha
        )
    }
}
