import pytest

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
