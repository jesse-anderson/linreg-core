# Type stubs for linreg-core Python bindings
#
# This file provides type hints for IDE autocomplete support.

from typing import List, Tuple, Optional, Union, Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    NDArray = np.ndarray
else:
    # Fallback if numpy is not available
    NDArray = object

# Type aliases for flexibility
FloatArray = Union[List[float], NDArray]
FloatMatrix = Union[List[List[float]], NDArray]
StringArray = Union[List[str], NDArray]

# ============================================================================
# Result Classes
# ============================================================================

class OLSResult:
    """Ordinary Least Squares regression results.

    Attributes
    ----------
    coefficients : List[float]
        Estimated regression coefficients (including intercept)
    standard_errors : List[float]
        Standard errors of coefficients
    t_statistics : List[float]
        t-statistics for coefficient significance tests
    p_values : List[float]
        Two-tailed p-values for coefficients
    r_squared : float
        Coefficient of determination (R²)
    r_squared_adjusted : float
        Adjusted R² (accounts for number of predictors)
    f_statistic : float
        F-statistic for overall model significance
    f_p_value : float
        p-value for F-statistic
    residuals : List[float]
        Raw residuals (y - ŷ)
    standardized_residuals : List[float]
        Residuals standardized by their standard errors
    leverage : List[float]
        Leverage values (hat matrix diagonal)
    vif : List[float]
        Variance Inflation Factors (excludes intercept)
    n_observations : int
        Number of observations
    n_predictors : int
        Number of predictor variables (excluding intercept)
    degrees_of_freedom : int
        Residual degrees of freedom
    mse : float
        Mean squared error
    rmse : float
        Root mean squared error
    """
    coefficients: List[float]
    standard_errors: List[float]
    t_statistics: List[float]
    p_values: List[float]
    r_squared: float
    r_squared_adjusted: float
    f_statistic: float
    f_p_value: float
    residuals: List[float]
    standardized_residuals: List[float]
    leverage: List[float]
    vif: List[float]
    n_observations: int
    n_predictors: int
    degrees_of_freedom: int
    mse: float
    rmse: float

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class RidgeResult:
    """Ridge regression results.

    Attributes
    ----------
    intercept : float
        Intercept coefficient
    coefficients : List[float]
        Slope coefficients (excluding intercept)
    lambda : float
        Lambda (regularization strength) used
    fitted_values : List[float]
        Fitted values (predictions on training data)
    residuals : List[float]
        Residuals (y - fitted_values)
    r_squared : float
        R-squared
    mse : float
        Mean squared error
    effective_df : float
        Effective degrees of freedom
    log_likelihood : float
        Log-likelihood of the model (for model comparison)
    aic : float
        Akaike Information Criterion (lower = better)
    bic : float
        Bayesian Information Criterion (lower = better)
    """
    intercept: float
    coefficients: List[float]
    lambda: float
    fitted_values: List[float]
    residuals: List[float]
    r_squared: float
    mse: float
    effective_df: float
    log_likelihood: float
    aic: float
    bic: float

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class LassoResult:
    """Lasso regression results.

    Attributes
    ----------
    intercept : float
        Intercept coefficient
    coefficients : List[float]
        Slope coefficients (some may be exactly zero)
    lambda : float
        Lambda (regularization strength) used
    n_nonzero : int
        Number of non-zero coefficients (excluding intercept)
    converged : bool
        Whether the algorithm converged
    n_iterations : int
        Number of iterations performed
    r_squared : float
        R-squared
    log_likelihood : float
        Log-likelihood of the model (for model comparison)
    aic : float
        Akaike Information Criterion (lower = better)
    bic : float
        Bayesian Information Criterion (lower = better)
    """
    intercept: float
    coefficients: List[float]
    lambda: float
    n_nonzero: int
    converged: bool
    n_iterations: int
    r_squared: float
    log_likelihood: float
    aic: float
    bic: float

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class ElasticNetResult:
    """Elastic Net regression results.

    Attributes
    ----------
    intercept : float
        Intercept coefficient
    coefficients : List[float]
        Slope coefficients
    lambda : float
        Lambda (regularization strength) used
    alpha : float
        Alpha mixing parameter (0 = Ridge, 1 = Lasso)
    n_nonzero : int
        Number of non-zero coefficients (excluding intercept)
    converged : bool
        Whether the algorithm converged
    n_iterations : int
        Number of iterations performed
    r_squared : float
        R-squared
    log_likelihood : float
        Log-likelihood of the model (for model comparison)
    aic : float
        Akaike Information Criterion (lower = better)
    bic : float
        Bayesian Information Criterion (lower = better)
    """
    intercept: float
    coefficients: List[float]
    lambda: float
    alpha: float
    n_nonzero: int
    converged: bool
    n_iterations: int
    r_squared: float
    log_likelihood: float
    aic: float
    bic: float

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class LambdaPathResult:
    """Lambda path generation results.

    Attributes
    ----------
    lambda_path : List[float]
        Generated lambda sequence (decreasing order)
    lambda_max : float
        Maximum lambda value
    lambda_min : float
        Minimum lambda value
    n_lambda : int
        Number of lambda values
    """
    lambda_path: List[float]
    lambda_max: float
    lambda_min: float
    n_lambda: int

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class DiagnosticResult:
    """Base result class for diagnostic tests.

    Attributes
    ----------
    statistic : float
        Test statistic
    p_value : float
        p-value for the test
    test_name : str
        Name of the test
    """
    statistic: float
    p_value: float
    test_name: str

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class DurbinWatsonResult:
    """Durbin-Watson test result.

    Attributes
    ----------
    statistic : float
        DW statistic (0 to 4, ~2 indicates no autocorrelation)
    """
    statistic: float

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class RainbowTestResult:
    """Rainbow test result with both R and Python variants.

    Attributes
    ----------
    test_name : str
        Name of the test
    has_r_result : bool
        Whether R result is available
    r_statistic : float
        R method statistic (available if has_r_result is true)
    r_p_value : float
        R method p-value (available if has_r_result is true)
    has_python_result : bool
        Whether Python result is available
    python_statistic : float
        Python method statistic (available if has_python_result is true)
    python_p_value : float
        Python method p-value (available if has_python_result is true)
    interpretation : str
        Interpretation of the test result
    guidance : str
        Guidance based on the test result
    """
    test_name: str
    has_r_result: bool
    r_statistic: float
    r_p_value: float
    has_python_result: bool
    python_statistic: float
    python_p_value: float
    interpretation: str
    guidance: str

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class WhiteTestResult:
    """White test result with both R and Python variants.

    Attributes
    ----------
    test_name : str
        Name of the test
    has_r_result : bool
        Whether R result is available
    r_statistic : float
        R method statistic (available if has_r_result is true)
    r_p_value : float
        R method p-value (available if has_r_result is true)
    has_python_result : bool
        Whether Python result is available
    python_statistic : float
        Python method statistic (available if has_python_result is true)
    python_p_value : float
        Python method p-value (available if has_python_result is true)
    interpretation : str
        Interpretation of the test result
    guidance : str
        Guidance based on the test result
    """
    test_name: str
    has_r_result: bool
    r_statistic: float
    r_p_value: float
    has_python_result: bool
    python_statistic: float
    python_p_value: float
    interpretation: str
    guidance: str

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class BreuschGodfreyResult:
    """Breusch-Godfrey test result for higher-order serial correlation.

    Attributes
    ----------
    test_name : str
        Name of the test
    order : int
        Maximum order of serial correlation tested
    test_type : str
        Type of test statistic computed ("chisq" or "f")
    statistic : float
        Test statistic value (LM or F)
    p_value : float
        P-value for the test
    df : List[float]
        Degrees of freedom
    passed : bool
        Whether the null hypothesis was not rejected (no serial correlation)
    interpretation : str
        Interpretation of the test result
    guidance : str
        Guidance for the user
    """
    test_name: str
    order: int
    test_type: str
    statistic: float
    p_value: float
    df: List[float]
    passed: bool
    interpretation: str
    guidance: str

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class CooksDistanceResult:
    """Cook's distance analysis result.

    Attributes
    ----------
    distances : List[float]
        Cook's distances (one per observation)
    p : int
        Number of parameters (including intercept)
    mse : float
        Mean squared error of the model
    threshold_4_over_n : float
        Common threshold: 4/n (observations above this are potentially influential)
    threshold_4_over_df : float
        Conservative threshold: 4/(n-p-1)
    threshold_1 : float
        Absolute threshold: D_i > 1 indicates high influence
    influential_4_over_n : List[int]
        Indices of observations exceeding 4/n threshold
    influential_4_over_df : List[int]
        Indices of observations exceeding conservative threshold
    influential_1 : List[int]
        Indices of observations exceeding D_i > 1 threshold
    interpretation : str
        Interpretation of results
    guidance : str
        Guidance for handling influential observations
    """
    distances: List[float]
    p: int
    mse: float
    threshold_4_over_n: float
    threshold_4_over_df: float
    threshold_1: float
    influential_4_over_n: List[int]
    influential_4_over_df: List[int]
    influential_1: List[int]
    interpretation: str
    guidance: str

    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


class CSVResult:
    """CSV parsing result.

    Attributes
    ----------
    headers : List[str]
        Column headers from the CSV
    data : Any
        Parsed data as Python object (list of dicts)
    numeric_columns : List[str]
        Names of columns that contain numeric data
    n_rows : int
        Number of rows parsed
    n_cols : int
        Number of columns parsed
    """
    headers: List[str]
    data: Any
    numeric_columns: List[str]
    n_rows: int
    n_cols: int

    def get_data(self) -> Any: ...
    def summary(self) -> str: ...
    def to_dict(self) -> dict: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


# ============================================================================
# Exception Classes
# ============================================================================

class LinregError(Exception):
    """Base exception for linreg-core errors."""
    message: str
    def __init__(self, message: str) -> None: ...
    def __str__(self) -> str: ...


class DataValidationError(LinregError):
    """Exception raised for data validation errors."""
    message: str
    def __init__(self, message: str) -> None: ...


# ============================================================================
# OLS Regression Functions
# ============================================================================

def ols_regression(
    y: FloatArray,
    x: FloatMatrix,
    variable_names: List[str]
) -> OLSResult: ...


# ============================================================================
# Regularized Regression Functions
# ============================================================================

def ridge_regression(
    y: FloatArray,
    x: FloatMatrix,
    lambda_val: float = 1.0,
    standardize: bool = True
) -> RidgeResult: ...


def lasso_regression(
    y: FloatArray,
    x: FloatMatrix,
    lambda_val: float = 0.1,
    standardize: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-7
) -> LassoResult: ...


def elastic_net_regression(
    y: FloatArray,
    x: FloatMatrix,
    lambda_val: float = 0.1,
    alpha: float = 0.5,
    standardize: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-7
) -> ElasticNetResult: ...


def make_lambda_path(
    y: FloatArray,
    x: FloatMatrix,
    n_lambda: int = 100,
    lambda_min_ratio: float = 0.01,
    alpha: float = 1.0
) -> LambdaPathResult: ...


# ============================================================================
# Statistical Utilities
# ============================================================================

def get_t_cdf(t: float, df: int) -> float: ...

def get_t_critical(alpha: float, df: int) -> float: ...

def get_normal_inverse(p: float) -> float: ...

def get_version() -> str: ...


# ============================================================================
# Diagnostic Test Functions
# ============================================================================

def rainbow_test(
    y: FloatArray,
    x: FloatMatrix,
    fraction: float = 0.5,
    method: str = "r"
) -> RainbowTestResult: ...


def harvey_collier_test(
    y: FloatArray,
    x: FloatMatrix
) -> DiagnosticResult: ...


def breusch_pagan_test(
    y: FloatArray,
    x: FloatMatrix
) -> DiagnosticResult: ...


def white_test(
    y: FloatArray,
    x: FloatMatrix,
    method: str = "r"
) -> WhiteTestResult: ...

def r_white_test(
    y: FloatArray,
    x: FloatMatrix
) -> DiagnosticResult: ...

def python_white_test(
    y: FloatArray,
    x: FloatMatrix
) -> DiagnosticResult: ...

def jarque_bera_test(
    y: FloatArray,
    x: FloatMatrix
) -> DiagnosticResult: ...


def durbin_watson_test(
    y: FloatArray,
    x: FloatMatrix
) -> DurbinWatsonResult: ...


def shapiro_wilk_test(
    y: FloatArray,
    x: FloatMatrix
) -> DiagnosticResult: ...


def anderson_darling_test(
    y: FloatArray,
    x: FloatMatrix
) -> DiagnosticResult: ...


def cooks_distance_test(
    y: FloatArray,
    x: FloatMatrix
) -> CooksDistanceResult: ...


def reset_test(
    y: FloatArray,
    x: FloatMatrix,
    powers: List[int],
    test_type: str = "fitted"
) -> DiagnosticResult: ...


def breusch_godfrey_test(
    y: FloatArray,
    x: FloatMatrix,
    order: int = 1,
    test_type: str = "chisq"
) -> BreuschGodfreyResult: ...


# ============================================================================
# Descriptive Statistics Functions
# ============================================================================

def stats_mean(data: FloatArray) -> float: ...

def stats_variance(data: FloatArray) -> float: ...

def stats_stddev(data: FloatArray) -> float: ...

def stats_median(data: FloatArray) -> float: ...

def stats_quantile(data: FloatArray, q: float) -> float: ...

def stats_correlation(x: FloatArray, y: FloatArray) -> float: ...


# ============================================================================
# CSV Parsing Functions
# ============================================================================

def parse_csv(content: str) -> CSVResult: ...


# ============================================================================
# Model Serialization Functions
# ============================================================================

def save_model(
    result: Union[OLSResult, RidgeResult, LassoResult, ElasticNetResult, LoessResult, WlsResult],
    path: str,
    name: Optional[str] = None
) -> dict:
    """Save a trained model to a file.

    Args:
        result: A regression result object (OLSResult, RidgeResult, etc.)
        path: File path to save the model (will be created/overwritten)
        name: Optional custom name for the model

    Returns:
        A dictionary with metadata about the saved model:
        - model_type: str - Type of model that was saved
        - path: str - Path where the model was saved
        - format_version: str - Serialization format version
        - library_version: str - linreg-core version used
        - name: str - Optional model name (if provided)

    Example:
        >>> result = ols_regression(y, [x1, x2], ["Intercept", "X1", "X2"])
        >>> metadata = save_model(result, "my_model.json", name="My Housing Model")
        >>> print(metadata["model_type"])
        OLS
    """

def load_model(path: str) -> Union[OLSResult, RidgeResult, LassoResult, ElasticNetResult, LoessResult, WlsResult]:
    """Load a trained model from a file.

    Args:
        path: File path to load the model from

    Returns:
        The appropriate result object (OLSResult, RidgeResult, etc.)
        based on the model type stored in the file

    Raises:
        IOError: If the file cannot be read
        ValueError: If the file contains invalid data or an unsupported model type

    Example:
        >>> result = load_model("my_model.json")
        >>> print(result.coefficients)
        [2.5, 1.3]
    """
