//! XLL Integration Tests
//!
//! Tests the XLL add-in UDFs by calling them directly with constructed
//! XLOPER12 inputs, simulating how Excel invokes worksheet functions.
//!
//! # Running the Tests
//!
//! ```bash
//! cargo test --features xll --test xll_tests
//! ```

#[cfg(feature = "xll")]
mod xll {
    include!("xll/mod.rs");
}
