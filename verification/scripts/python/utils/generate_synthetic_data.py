"""
Generate synthetic datasets for OLS regression verification.

Each dataset has known properties to validate specific aspects of the regression:
- Simple linear: basic single predictor regression
- Multiple regression: multiple predictors with known coefficients
- Perfect collinearity: tests error handling for singular matrices
- Heteroscedastic: tests Breusch-Pagan and White detection
- Nonlinear: tests Rainbow and Harvey-Collier linearity tests
- Non-normal residuals: tests Jarque-Bera normality test
- Autocorrelated: tests Durbin-Watson statistic
"""

import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = "verification/datasets/csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_simple_linear(n=100):
    """Simple linear regression: y = 2 + 3*x + noise"""
    x = np.linspace(0, 10, n)
    noise = np.random.normal(0, 1, n)
    y = 2 + 3 * x + noise

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_simple_linear.csv", index=False)
    print(f"Created synthetic_simple_linear.csv (n={n}, true intercept=2, true slope=3)")


def generate_multiple_regression(n=100):
    """Multiple regression: y = 1 + 2*x1 - 1.5*x2 + 0.5*x3 + noise"""
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(-5, 5, n)
    x3 = np.random.uniform(0, 100, n)
    noise = np.random.normal(0, 0.5, n)
    y = 1 + 2 * x1 - 1.5 * x2 + 0.5 * x3 + noise

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_multiple.csv", index=False)
    print(f"Created synthetic_multiple.csv (n={n}, 3 predictors)")


def generate_perfect_collinearity(n=100):
    """Data with perfect collinearity: x3 = 2*x1 + x2"""
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 5, n)
    x3 = 2 * x1 + x2  # Perfect linear combination
    noise = np.random.normal(0, 0.5, n)
    y = 1 + x1 + 0.5 * x2 + 2 * x3 + noise

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_collinear.csv", index=False)
    print(f"Created synthetic_collinear.csv (n={n}, x3 = 2*x1 + x2)")


def generate_heteroscedastic(n=100):
    """Heteroscedastic data: variance increases with x"""
    x = np.linspace(0, 10, n)
    # Variance increases with x (sigma = 0.5 + 0.3*x)
    noise = np.random.normal(0, 1, n) * (0.5 + 0.3 * x)
    y = 2 + 1.5 * x + noise

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_heteroscedastic.csv", index=False)
    print(f"Created synthetic_heteroscedastic.csv (n={n}, variance increases with x)")


def generate_nonlinear(n=100):
    """Nonlinear relationship: y = 1 + x + 0.3*x^2 + noise"""
    x = np.linspace(-5, 5, n)
    noise = np.random.normal(0, 0.5, n)
    y = 1 + x + 0.3 * x**2 + noise

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_nonlinear.csv", index=False)
    print(f"Created synthetic_nonlinear.csv (n={n}, quadratic relationship)")


def generate_non_normal_residuals(n=100):
    """Data with non-normal residuals (exponential noise)"""
    x = np.linspace(0, 10, n)
    # Exponential noise (skewed distribution)
    noise = np.random.exponential(1, n) - 1
    y = 2 + 1.5 * x + noise * 0.5

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_nonnormal.csv", index=False)
    print(f"Created synthetic_nonnormal.csv (n={n}, exponential noise)")


def generate_autocorrelated(n=100, phi=0.7):
    """Data with autocorrelated residuals: AR(1) process"""
    x = np.linspace(0, 10, n)
    # AR(1) noise: epsilon_t = phi * epsilon_{t-1} + u_t
    noise = np.zeros(n)
    u = np.random.normal(0, 0.5, n)
    noise[0] = u[0]
    for t in range(1, n):
        noise[t] = phi * noise[t-1] + u[t]

    y = 2 + 1.5 * x + noise

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_autocorrelated.csv", index=False)
    print(f"Created synthetic_autocorrelated.csv (n={n}, phi={phi})")


def generate_high_multicollinearity(n=100):
    """Data with high (but not perfect) multicollinearity"""
    x1 = np.random.uniform(0, 10, n)
    # x2 highly correlated with x1 but not perfectly
    x2 = x1 + np.random.normal(0, 0.3, n)
    x3 = np.random.uniform(0, 5, n)
    noise = np.random.normal(0, 0.5, n)
    y = 1 + 2 * x1 + 1.5 * x2 + 0.5 * x3 + noise

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_high_vif.csv", index=False)
    print(f"Created synthetic_high_vif.csv (n={n}, x2 ~ x1 + noise)")


def generate_outliers(n=100, n_outliers=5):
    """Data with outliers for influence testing"""
    x = np.linspace(0, 10, n)
    noise = np.random.normal(0, 0.5, n)
    y = 2 + 1.5 * x + noise

    # Add outliers at the end
    y[-n_outliers:] += np.random.uniform(5, 10, n_outliers)

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_outliers.csv", index=False)
    print(f"Created synthetic_outliers.csv (n={n}, {n_outliers} outliers)")


def generate_small_sample(n=15):
    """Small sample for edge case testing"""
    x = np.random.uniform(0, 10, n)
    noise = np.random.normal(0, 0.5, n)
    y = 2 + 1.5 * x + noise

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_small.csv", index=False)
    print(f"Created synthetic_small.csv (n={n}, small sample)")


def generate_interaction(n=100):
    """Data with interaction effect: y = a + b1*x1 + b2*x2 + b3*x1*x2 + noise"""
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 10, n)
    noise = np.random.normal(0, 0.5, n)
    y = 1 + 2 * x1 + 1.5 * x2 + 0.5 * x1 * x2 + noise

    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    df.to_csv(f"{OUTPUT_DIR}/synthetic_interaction.csv", index=False)
    print(f"Created synthetic_interaction.csv (n={n}, x1*x2 interaction)")


if __name__ == "__main__":
    print("Generating synthetic datasets for OLS regression verification...")
    print("=" * 60)

    generate_simple_linear()
    generate_multiple_regression()
    generate_perfect_collinearity()
    generate_heteroscedastic()
    generate_nonlinear()
    generate_non_normal_residuals()
    generate_autocorrelated()
    generate_high_multicollinearity()
    generate_outliers()
    generate_small_sample()
    generate_interaction()

    print("=" * 60)
    print(f"All synthetic datasets saved to: {OUTPUT_DIR}/")
