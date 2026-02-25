//! Excel12v trampoline — Rust implementation of Excel12v trampoline.
//!
//! Excel exports `MdCallBack12` from the host process.  When the XLL is loaded
//! we look it up via `GetModuleHandleA` / `GetProcAddress` and cache the
//! pointer.  All calls to `excel12v` go through this cached pointer.
//!
//! This replaces the need for any outside compilation nonsense...

#![allow(non_snake_case)]

use super::types::XLOPER12;
use std::sync::atomic::{AtomicPtr, Ordering};

// ── Win32 FFI ───────────────────────────────────────────────────────────────

type HMODULE = *mut u8;
type FARPROC = *mut u8;

extern "system" {
    fn GetModuleHandleA(lpModuleName: *const u8) -> HMODULE;
    fn GetProcAddress(hModule: HMODULE, lpProcName: *const u8) -> FARPROC;
}

// ── MdCallBack12 signature ──────────────────────────────────────────────────

/// Signature of Excel's internal callback:
///   `int PASCAL MdCallBack12(int xlfn, int coper, LPXLOPER12 *rgpxloper12, LPXLOPER12 xloper12Res)`
type Excel12Proc = unsafe extern "system" fn(
    xlfn: i32,
    coper: i32,
    rgpxloper12: *const *mut XLOPER12,
    xloper12Res: *mut XLOPER12,
) -> i32;

// ── Cached entry point ─────────────────────────────────────────────────────

static ENTRY_PT: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());

fn fetch_entry_pt() -> Option<Excel12Proc> {
    let mut ptr = ENTRY_PT.load(Ordering::Acquire);
    if ptr.is_null() {
        unsafe {
            let hmod = GetModuleHandleA(std::ptr::null());
            if hmod.is_null() {
                return None;
            }
            ptr = GetProcAddress(hmod, b"MdCallBack12\0".as_ptr());
            if ptr.is_null() {
                return None;
            }
            ENTRY_PT.store(ptr, Ordering::Release);
        }
    }
    Some(unsafe { std::mem::transmute(ptr) })
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Return codes from Excel12v.
pub const XLRET_SUCCESS: i32 = 0;
pub const XLRET_FAILED: i32 = 32;

/// Call into Excel via the `MdCallBack12` entry point.
///
///
/// # Safety
///
/// `opers` must contain `count` valid pointers.  `oper_res` must point to a
/// valid (possibly zeroed) XLOPER12.
pub unsafe fn excel12v(
    xlfn: i32,
    oper_res: *mut XLOPER12,
    count: i32,
    opers: *const *mut XLOPER12,
) -> i32 {
    match fetch_entry_pt() {
        Some(f) => f(xlfn, count, opers, oper_res),
        None => XLRET_FAILED,
    }
}

/// Convenience: call Excel12v with a slice of XLOPER12 pointers.
pub fn excel12(xlfn: i32, args: &mut [*mut XLOPER12]) -> (i32, XLOPER12) {
    let mut result = XLOPER12::nil();
    let ret = unsafe {
        excel12v(
            xlfn,
            &mut result,
            args.len() as i32,
            args.as_ptr(),
        )
    };
    (ret, result)
}

/// Free an XLOPER12 that was returned by Excel (has `xlbitXLFree` set).
pub fn excel_free(oper: &mut XLOPER12) {
    unsafe {
        let mut p = oper as *mut XLOPER12;
        excel12v(super::types::XL_FREE, std::ptr::null_mut(), 1, &mut p);
    }
}
