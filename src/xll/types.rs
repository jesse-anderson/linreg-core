//! XLOPER12 type definitions
//!
//! These are `#[repr(C)]` structs that match the exact memory layout Excel
//! expects.  Only the variants we actually use are included; flow, bigdata,
//! and legacy XLOPER (pre-2007) types are omitted. Note, when we inevitably
//! decide to code xll-rs and xllgen we need to accomodate ALL types. : (

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use std::mem::ManuallyDrop;

// ── xltype constants ────────────────────────────────────────────────────────

pub const XLTYPE_NUM: u32 = 0x0001;
pub const XLTYPE_STR: u32 = 0x0002;
pub const XLTYPE_BOOL: u32 = 0x0004;
pub const XLTYPE_REF: u32 = 0x0008;
pub const XLTYPE_ERR: u32 = 0x0010;
pub const XLTYPE_MULTI: u32 = 0x0040;
pub const XLTYPE_MISSING: u32 = 0x0080;
pub const XLTYPE_NIL: u32 = 0x0100;
pub const XLTYPE_SREF: u32 = 0x0400;
pub const XLTYPE_INT: u32 = 0x0800;

pub const XLBIT_XL_FREE: u32 = 0x1000;
pub const XLBIT_DLL_FREE: u32 = 0x4000;

// ── Excel error codes ───────────────────────────────────────────────────────

pub const XLERR_NULL: i32 = 0;
pub const XLERR_DIV0: i32 = 7;
pub const XLERR_VALUE: i32 = 15;
pub const XLERR_REF: i32 = 23;
pub const XLERR_NAME: i32 = 29;
pub const XLERR_NUM: i32 = 36;
pub const XLERR_NA: i32 = 42;

// ── Excel C API function numbers ────────────────────────────────────────────

const XL_SPECIAL: i32 = 0x4000;

pub const XL_FREE: i32 = 0 | XL_SPECIAL;
pub const XL_GET_NAME: i32 = 9 | XL_SPECIAL;
pub const XLF_REGISTER: i32 = 149;

// ── XLOPER12 ────────────────────────────────────────────────────────────────

/// The core XLOPER12 struct.
#[repr(C)]
pub struct XLOPER12 {
    pub val: XLOPER12Val,
    pub xltype: u32,
}

/// Union of all XLOPER12 value types.
///
/// We use `ManuallyDrop` wrappers for struct variants to allow them in a
/// `Copy` union without implementing `Drop` on the union itself (memory
/// is managed externally via `xlAutoFree12`).
#[repr(C)]
pub union XLOPER12Val {
    pub num: f64,
    pub str_: *mut u16,
    pub xbool: i32,
    pub err: i32,
    pub w: i32,
    pub array: ManuallyDrop<XLOPER12Array>,
    pub sref: ManuallyDrop<XLOPER12SRef>,
    pub mref: ManuallyDrop<XLOPER12MRef>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XLOPER12Array {
    pub lparray: *mut XLOPER12,
    pub rows: i32,
    pub columns: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XLOPER12SRef {
    pub count: u16,
    pub ref_: XLREF12,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XLREF12 {
    pub rw_first: i32,
    pub rw_last: i32,
    pub col_first: i32,
    pub col_last: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XLOPER12MRef {
    pub lpmref: *mut XLMREF12,
    pub id_sheet: usize,
}

#[repr(C)]
pub struct XLMREF12 {
    pub count: u16,
    pub reftbl: [XLREF12; 1],
}

// ── Convenience constructors ────────────────────────────────────────────────

impl XLOPER12 {
    /// Create a nil (empty) XLOPER12.
    pub fn nil() -> Self {
        Self {
            val: XLOPER12Val { w: 0 },
            xltype: XLTYPE_NIL,
        }
    }

    /// Create a numeric XLOPER12.
    pub fn from_f64(v: f64) -> Self {
        Self {
            val: XLOPER12Val { num: v },
            xltype: XLTYPE_NUM,
        }
    }

    /// Create an integer XLOPER12.
    pub fn from_int(v: i32) -> Self {
        Self {
            val: XLOPER12Val { w: v },
            xltype: XLTYPE_INT,
        }
    }

    /// Create an error XLOPER12.
    pub fn from_err(code: i32) -> Self {
        Self {
            val: XLOPER12Val { err: code },
            xltype: XLTYPE_ERR,
        }
    }

    /// Create a missing-argument XLOPER12.
    pub fn missing() -> Self {
        Self {
            val: XLOPER12Val { w: 0 },
            xltype: XLTYPE_MISSING,
        }
    }

    /// Create a string XLOPER12 from a Rust `&str`.
    ///
    /// Allocates a length-counted UTF-16 buffer: `[len, char0, char1, ...]`.
    /// Sets `xlbitDLLFree` so the memory is reclaimed in `xlAutoFree12`.
    pub fn from_str(s: &str) -> Self {
        let utf16: Vec<u16> = s.encode_utf16().collect();
        let len = utf16.len();
        if len > 32767 {
            return Self::from_err(XLERR_VALUE);
        }

        // Build pascal-style string: length prefix followed by chars
        let mut buf: Vec<u16> = Vec::with_capacity(len + 1);
        buf.push(len as u16);
        buf.extend_from_slice(&utf16);

        let ptr = buf.as_mut_ptr();
        std::mem::forget(buf);

        Self {
            val: XLOPER12Val { str_: ptr },
            xltype: XLTYPE_STR | XLBIT_DLL_FREE,
        }
    }

    /// Extract a Rust `String` from this XLOPER12 if it is a string type.
    pub fn as_string(&self) -> Option<String> {
        if (self.xltype & 0x0FFF) != XLTYPE_STR {
            return None;
        }
        unsafe {
            let ptr = self.val.str_;
            if ptr.is_null() {
                return None;
            }
            let len = *ptr as usize;
            let slice = std::slice::from_raw_parts(ptr.add(1), len);
            Some(String::from_utf16_lossy(slice))
        }
    }

    /// Extract an `f64` from this XLOPER12 if it is numeric or integer type.
    pub fn as_f64(&self) -> Option<f64> {
        match self.xltype & 0x0FFF {
            XLTYPE_NUM => Some(unsafe { self.val.num }),
            XLTYPE_INT => Some(unsafe { self.val.w } as f64),
            _ => None,
        }
    }

    /// Returns `true` if this XLOPER12 represents a missing argument.
    pub fn is_missing(&self) -> bool {
        (self.xltype & 0x0FFF) == XLTYPE_MISSING
    }

    /// Base type with memory flags masked off.
    pub fn base_type(&self) -> u32 {
        self.xltype & 0x0FFF
    }
}
