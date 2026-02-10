//! Weighted regression methods
//!
//! This module contains regression methods that incorporate observation weights,
//! allowing for heteroscedastic data, robust fitting, and other applications where
//! observations have different levels of importance.

pub mod wls;

pub use wls::{wls_regression, WlsFit};
