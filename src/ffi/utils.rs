//! Utility FFI functions.

use std::slice;

/// Write the library version string into a caller-supplied buffer.
///
/// The buffer is always null-terminated if `out_len` > 0.
/// Returns the number of bytes written (not including the null terminator),
/// or -1 on invalid arguments.
#[no_mangle]
pub extern "system" fn LR_Version(out_ptr: *mut u8, out_len: i32) -> i32 {
    if out_ptr.is_null() || out_len <= 0 {
        return -1;
    }
    let version = env!("CARGO_PKG_VERSION");
    let bytes = version.as_bytes();
    let cap = (out_len as usize).saturating_sub(1);
    let count = bytes.len().min(cap);
    unsafe {
        let dst = slice::from_raw_parts_mut(out_ptr, count + 1);
        dst[..count].copy_from_slice(&bytes[..count]);
        dst[count] = 0;
    }
    count as i32
}

/// Confirm the DLL is loaded and functional.
///
/// Always returns 1.  Call this from `Workbook_Open` or `LR_Init` in VBA to
/// verify the DLL path is correct before making any other calls.
#[no_mangle]
pub extern "system" fn LR_Init() -> i32 {
    1
}
