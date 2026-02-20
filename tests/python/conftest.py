import pytest
import numpy as np

# Sample data fixtures
@pytest.fixture
def sample_y():
    return [1.0, 2.0, 3.0, 4.0, 5.0]

@pytest.fixture
def sample_x():
    return [[1.0, 2.0, 3.0, 4.0, 5.0]]

@pytest.fixture
def sample_names():
    return ["Intercept", "X1"]

# Multi-predictor fixtures
@pytest.fixture
def housing_y():
    return [245.5, 312.8, 198.4, 425.6, 278.9, 356.2, 189.5, 512.3, 234.7, 298.1]

@pytest.fixture
def housing_x():
    return [
        [1200.0, 1800.0, 950.0, 2400.0, 1450.0, 2000.0, 1100.0, 2800.0, 1350.0, 1650.0],
        [3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 3.0]
    ]

# Diagnostic test fixtures (data with variance for proper residual analysis)
@pytest.fixture
def diagnostic_y():
    # 10 observations with some noise - not a perfect linear relationship
    return [2.1, 4.3, 5.8, 8.2, 9.7, 12.1, 13.8, 16.2, 17.9, 20.3]

@pytest.fixture
def diagnostic_x():
    # Predictor with some variance from perfect linearity
    return [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]

# ============================================================================
# Polynomial Regression Fixtures
# ============================================================================

@pytest.fixture
def poly_x():
    """Single predictor for polynomial regression (evenly spaced 1-10)."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

@pytest.fixture
def poly_y_quadratic():
    """Quadratic relationship: y = 1 + 2x + 0.5x² with small noise."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, size=len(x))
    y = 1.0 + 2.0 * x + 0.5 * x**2 + noise
    return y.tolist()

@pytest.fixture
def poly_y_cubic():
    """Cubic relationship: y = 1 + x - 0.3x² + 0.05x³ with small noise."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    np.random.seed(123)
    noise = np.random.normal(0, 1.0, size=len(x))
    y = 1.0 + 1.0 * x - 0.3 * x**2 + 0.05 * x**3 + noise
    return y.tolist()

@pytest.fixture
def poly_y_perfect_quadratic():
    """Perfect quadratic relationship (no noise) for validation: y = 2 + x + 0.5x²."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = 2.0 + 1.0 * x + 0.5 * x**2
    return y.tolist()

@pytest.fixture
def poly_x_new():
    """New x values for prediction testing."""
    return [11.0, 12.0, 15.0, 20.0]

@pytest.fixture
def poly_small_x():
    """Small dataset for edge case testing (5 observations)."""
    return [1.0, 2.0, 3.0, 4.0, 5.0]

@pytest.fixture
def poly_small_y():
    """Small quadratic dataset (5 observations)."""
    return [3.5, 6.0, 9.5, 14.0, 19.5]  # y = 2 + x + 0.5x²

@pytest.fixture
def poly_centered_x():
    """X values that benefit from centering (values far from zero)."""
    return [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0]

@pytest.fixture
def poly_centered_y():
    """Quadratic response for centered x values."""
    x = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0])
    # Center at mean (109) for stability
    x_centered = x - 109.0
    np.random.seed(456)
    noise = np.random.normal(0, 2.0, size=len(x))
    y = 10.0 + 3.0 * x_centered + 0.1 * x_centered**2 + noise
    return y.tolist()

# Numpy array versions for testing numpy integration
@pytest.fixture
def poly_x_np():
    """Single predictor as numpy array."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

@pytest.fixture
def poly_y_np():
    """Quadratic response as numpy array."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, size=len(x))
    y = 1.0 + 2.0 * x + 0.5 * x**2 + noise
    return y
