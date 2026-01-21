// ============================================================================
// Linear Algebra Unit Tests
// ============================================================================
//
// Comprehensive tests for matrix operations and QR decomposition.
// Tests are organized into focused modules:
//
// - common:                 Shared constants and helper functions
// - basic_tests:            Constructor, element access, transpose tests
// - ops_tests:              Matrix multiplication, vector operations
// - qr_tests:               QR decomposition tests
// - inversion_tests:        Matrix inversion, chol2inv tests
// - property_tests:         Property-based tests with proptest
// - nalgebra_comparison:    Comparison tests against nalgebra (disabled)
// - column_ops_tests:       Column operations (col_dot, col_axpy_inplace, etc.)
// - vector_functions_tests: Vector helper functions (vec_add, vec_scale, etc.)
// - tolerance_tests:        Custom tolerance variants
// - linpack_qr_tests:       R's LINPACK QR decomposition implementation

pub mod common;

mod basic_tests;
mod ops_tests;
mod qr_tests;
mod inversion_tests;
mod property_tests;
mod nalgebra_comparison;
mod column_ops_tests;
mod vector_functions_tests;
mod tolerance_tests;
mod linpack_qr_tests;
mod edge_case_tests;
mod numerical_accuracy_tests;
