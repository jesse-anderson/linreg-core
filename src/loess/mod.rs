//! LOESS (Locally Estimated Scatterplot Smoothing)
//!
//! Non-parametric regression method that fits multiple regressions in local
//! subsets of data to create a smooth curve through the data points.
//!
//! # References
//!
//! - Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing Scatterplots".
//!   *Journal of the American Statistical Association*. 74 (368): 829-836.
//!
//! # Overview
//!
//! LOESS works by:
//! 1. For each point to be fitted, finding the k nearest neighbors (where k = span * n)
//! 2. Assigning weights to neighbors using the tricube kernel function
//! 3. Fitting a weighted least squares polynomial (constant, linear, or quadratic)
//! 4. Using the fitted value as the smoothed value at that point
//!
//! Optionally, robust fitting iterations can be performed to reduce the influence
//! of outliers using a biweight function.
//!
//! # Example
//!
//! ```rust,no_run
//! use linreg_core::loess::{loess_fit, LoessOptions};
//!
//! let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
//! let y = vec![1.0, 3.5, 4.8, 6.2, 8.5, 11.0, 13.2, 14.8, 17.5, 19.0, 22.0];
//!
//! let options = LoessOptions::default();
//! let result = loess_fit(&y, &[x], &options).unwrap();
//! ```

pub mod core;
pub mod neighbors;
pub mod normalize;
pub mod predict;
pub mod robust;
pub mod types;
pub mod weights;
pub mod wls;

// Public API re-exports
pub use core::loess_fit;
pub use predict::{fit_at_point, fit_at_point_impl};
pub use types::{LoessFit, LoessOptions, LoessSurface};
pub use weights::tricube_weight;

#[cfg(test)]
mod tests;
