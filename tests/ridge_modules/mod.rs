//! Ridge regression validation tests
//!
//! This module contains comprehensive validation tests for ridge regression:
//! - `baseline` - Basic smoke tests and baseline values
//! - `verification` - Manual calculation verification
//! - `glmnet_audit` - Comparison with R's glmnet

mod baseline;
mod verification;
mod glmnet_audit;
