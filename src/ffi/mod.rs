//! Windows DLL (FFI) bindings for VBA / Excel use.
//!
//! Compiled only when the `ffi` feature is enabled:
//!
//! ```bash
//! cargo build --release --target x86_64-pc-windows-msvc --features ffi
//! cargo build --release --target i686-pc-windows-msvc   --features ffi
//! ```
//!
//! # Handle-based API
//!
//! All fit functions return an opaque `usize` handle (>= 1).  A return value
//! of `0` means an error occurred — call `LR_GetLastError` to retrieve the
//! message.  Call `LR_Free` when you are done with a handle.
//!
//! # Calling convention
//!
//! All exported functions use `extern "system"` with `#[no_mangle]`.
//! On 32-bit Windows `"system"` maps to stdcall; on 64-bit Windows and other
//! platforms it maps to the platform's default C calling convention.
//! On 32-bit Windows a linker `.def` file (see the build script) exports
//! plain names without the `_FunctionName@N` stdcall decoration.

pub mod cross_validation;
pub mod diagnostics;
pub mod ols;
pub mod prediction_intervals;
pub mod regularized;
pub mod store;
pub mod types;
pub mod utils;
