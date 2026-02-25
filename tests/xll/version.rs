// XLL tests for LINREG.VERSION() UDF.

use super::common::*;

#[test]
fn test_version_returns_string() {
    let result = XlResultGuard::new(xll_version());
    assert!(result.is_string(), "VERSION should return a string XLOPER12");
}

#[test]
fn test_version_contains_dots() {
    let result = XlResultGuard::new(xll_version());
    let version = result.as_string();
    assert!(version.contains('.'), "Version should contain dots: {}", version);
}

#[test]
fn test_version_matches_cargo_pkg() {
    let result = XlResultGuard::new(xll_version());
    let version = result.as_string();
    assert_eq!(version, env!("CARGO_PKG_VERSION"), "Should match Cargo.toml version");
}

#[test]
fn test_version_consistency() {
    let r1 = XlResultGuard::new(xll_version());
    let r2 = XlResultGuard::new(xll_version());
    assert_eq!(r1.as_string(), r2.as_string(), "Repeated calls should return same version");
}

#[test]
fn test_version_has_dll_free_bit() {
    let ptr = xll_version();
    let xltype = unsafe { (*ptr).xltype };
    assert!(
        xltype & linreg_core::xll::types::XLBIT_DLL_FREE != 0,
        "Returned XLOPER12 should have xlbitDLLFree set"
    );
    // Clean up
    xll_free(ptr);
}
