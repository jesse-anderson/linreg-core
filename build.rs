// build.rs — linreg-core
//
// On 32-bit Windows MSVC (i686-pc-windows-msvc) the linker decorates
// `extern "system"` (stdcall) exports as `_FunctionName@N`.
// VBA `Declare` statements need plain names, so we supply a .def file that
// exports every symbol with its undecorated name.
//
// This only fires when:
//   • the `ffi` feature is enabled (i.e. you're building the DLL), AND
//   • the target is i686 / x86 Windows MSVC.
//
// x86_64 Windows does not use stdcall decoration, so no .def file is needed
// there.

fn main() {
    // Only relevant for the ffi (DLL) build.
    if std::env::var("CARGO_FEATURE_FFI").is_err() {
        return;
    }

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os   = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let env  = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    if arch == "x86" && os == "windows" && env == "msvc" {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        // cargo:rustc-cdylib-link-arg is forwarded only to cdylib link steps,
        // so it won't affect rlib / wasm builds.
        println!("cargo:rustc-cdylib-link-arg=/DEF:{manifest}/linreg_core.def");
    }
}
