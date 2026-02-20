// FFI tests for utility functions (version, init, error handling).

#[cfg(feature = "ffi")]

use super::common::*;

// ============================================================================
// Version Tests
// ============================================================================

#[test]
fn test_version_returns_string() {
    let mut buffer = vec![0u8; 64];
    let written = unsafe { LR_Version(buffer.as_mut_ptr(), 64) };

    assert!(written > 0, "Version should write at least one byte");

    // Version should be valid UTF-8
    let version_str = String::from_utf8_lossy(&buffer[..written as usize]);
    assert!(!version_str.is_empty(), "Version string should not be empty");

    // Version should contain a dot (e.g., "0.7.0")
    assert!(version_str.contains('.'), "Version should contain dots: {}", version_str);
}

#[test]
fn test_version_buffer_too_small() {
    // Create a very small buffer
    let mut buffer = vec![0u8; 3];
    let written = unsafe { LR_Version(buffer.as_mut_ptr(), 3) };

    // Should write at most 2 bytes (leaving room for null terminator)
    assert!(written <= 2, "Should respect buffer size");
}

#[test]
fn test_version_zero_buffer() {
    let written = unsafe { LR_Version(std::ptr::null_mut(), 0) };
    // Should return -1 for null buffer or 0 for zero length
    assert!(written == 0 || written == -1, "Should handle zero/null buffer");
}

#[test]
fn test_version_consistency() {
    // Multiple calls should return the same version
    let mut buffer1 = vec![0u8; 64];
    let mut buffer2 = vec![0u8; 64];

    let written1 = unsafe { LR_Version(buffer1.as_mut_ptr(), 64) };
    let written2 = unsafe { LR_Version(buffer2.as_mut_ptr(), 64) };

    assert_eq!(written1, written2, "Version length should be consistent");

    let version1 = String::from_utf8_lossy(&buffer1[..written1 as usize]);
    let version2 = String::from_utf8_lossy(&buffer2[..written2 as usize]);

    assert_eq!(version1, version2, "Version string should be consistent");
}

// ============================================================================
// Init Tests
// ============================================================================

#[test]
fn test_init_returns_success() {
    // LR_Init is primarily for Windows/DLL loading
    // On non-Windows or when already loaded, it should still return a valid value
    let result = unsafe { LR_Init() };

    // Return value should be non-negative (0 = success, positive = some status)
    assert!(result >= 0, "LR_Init should return non-negative value");
}

#[test]
fn test_init_multiple_calls() {
    // Multiple init calls should be safe
    let r1 = unsafe { LR_Init() };
    let r2 = unsafe { LR_Init() };
    let r3 = unsafe { LR_Init() };

    assert!(r1 >= 0);
    assert!(r2 >= 0);
    assert!(r3 >= 0);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_get_last_error_with_no_error() {
    // Before any error, get last error should return empty or unknown
    let mut buffer = vec![0u8; 256];
    let written = unsafe { LR_GetLastError(buffer.as_mut_ptr(), 256) };

    // Either 0 (no error) or some message
    assert!(written >= 0, "GetLastError should handle no-error state");
}

#[test]
fn test_get_last_error_after_ols_error() {
    // Trigger an error
    let handle = unsafe { LR_OLS(std::ptr::null(), 10, std::ptr::null(), 1) };
    assert_eq!(handle, 0);

    // Get the error message
    let mut buffer = vec![0u8; 256];
    let written = unsafe { LR_GetLastError(buffer.as_mut_ptr(), 256) };

    assert!(written > 0, "Should have error message after failed call");

    let msg = String::from_utf8_lossy(&buffer[..written as usize]);
    assert!(!msg.is_empty(), "Error message should not be empty");
}

#[test]
fn test_get_last_error_persists() {
    // Trigger an error
    let _handle = unsafe { LR_OLS(std::ptr::null(), 10, std::ptr::null(), 1) };

    // Get the error message
    let mut buffer1 = vec![0u8; 256];
    let written1 = unsafe { LR_GetLastError(buffer1.as_mut_ptr(), 256) };

    // Get it again - should be the same
    let mut buffer2 = vec![0u8; 256];
    let written2 = unsafe { LR_GetLastError(buffer2.as_mut_ptr(), 256) };

    assert_eq!(written1, written2, "Error should persist across calls");

    let msg1 = String::from_utf8_lossy(&buffer1[..written1 as usize]);
    let msg2 = String::from_utf8_lossy(&buffer2[..written2 as usize]);

    assert_eq!(msg1, msg2, "Error message should be identical");
}

#[test]
fn test_get_last_error_buffer_sizes() {
    // Trigger an error
    let _handle = unsafe { LR_OLS(std::ptr::null(), 10, std::ptr::null(), 1) };

    // Test with various buffer sizes
    for size in [1, 10, 64, 256] {
        let mut buffer = vec![0u8; size];
        let written = unsafe { LR_GetLastError(buffer.as_mut_ptr(), size as i32) };

        assert!(written >= 0, "Should handle buffer size {}", size);
        assert!(written <= size as i32, "Should not write beyond buffer");

        if size > 1 {
            let msg = String::from_utf8_lossy(&buffer[..written as usize]);
            assert!(!msg.is_empty() || written == 0);
        }
    }
}

#[test]
fn test_get_last_error_null_buffer() {
    // Null buffer should be handled gracefully
    let written = unsafe { LR_GetLastError(std::ptr::null_mut(), 256) };
    assert!(written <= 0, "Null buffer should return 0 or negative");
}

// ============================================================================
// Handle Management Tests
// ============================================================================

#[test]
fn test_free_zero_handle_is_safe() {
    // Freeing handle 0 (error handle) should not crash
    unsafe {
        LR_Free(0);
        LR_Free(0);
        LR_Free(0);
    }
}

#[test]
fn test_free_invalid_handle_is_safe() {
    // Freeing invalid handles should not crash
    unsafe {
        LR_Free(999999);
        LR_Free(12345);
        LR_Free(std::usize::MAX);
    }
}

#[test]
fn test_handle_uniqueness() {
    let (y, x) = simple_linear_data();
    let x_matrix = columns_to_row_major(&[x]);

    let handles: Vec<usize> = (0..10)
        .map(|_| unsafe {
            LR_OLS(
                y.as_ptr(),
                y.len() as i32,
                x_matrix.as_ptr(),
                1,
            )
        })
        .collect();

    // All handles should be non-zero
    for &h in &handles {
        assert_ne!(h, 0, "Each OLS call should return valid handle");
    }

    // All handles should be unique
    let mut unique_handles = std::collections::HashSet::new();
    for &h in &handles {
        assert!(
            unique_handles.insert(h),
            "Handle {} should be unique",
            h
        );
    }

    // Clean up
    for h in handles {
        unsafe { LR_Free(h) };
    }
}

// ============================================================================
// Thread Safety Tests (Basic)
// ============================================================================

#[test]
fn test_concurrent_handle_creation() {
    use std::thread;

    let handles: Vec<_> = (0..5)
        .map(|_| {
            thread::spawn(|| {
                let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                let x_matrix = columns_to_row_major(&[x]);

                let handle = unsafe {
                    LR_OLS(
                        y.as_ptr(),
                        y.len() as i32,
                        x_matrix.as_ptr(),
                        1,
                    )
                };

                // Clean up in the thread
                if handle != 0 {
                    let r2 = unsafe { LR_GetRSquared(handle) };
                    unsafe { LR_Free(handle) };
                    r2
                } else {
                    0.0
                }
            })
            .join()
            .unwrap()
        })
        .collect();

    // All threads should have succeeded
    for r2 in handles {
        assert!(r2 > 0.0, "Concurrent OLS calls should all succeed");
    }
}

// ============================================================================
// NaN and Infinity Handling
// ============================================================================

#[test]
fn test_ols_with_nan_data() {
    let y = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_matrix = columns_to_row_major(&[x]);

    let handle = unsafe { LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1) };

    // Should either error or handle gracefully
    if handle != 0 {
        unsafe { LR_Free(handle) };
    }
}

#[test]
fn test_ols_with_inf_data() {
    let y = vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0];
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_matrix = columns_to_row_major(&[x]);

    let handle = unsafe { LR_OLS(y.as_ptr(), y.len() as i32, x_matrix.as_ptr(), 1) };

    // Should either error or handle gracefully
    if handle != 0 {
        unsafe { LR_Free(handle) };
    }
}

// ============================================================================
// Utility Integration Tests
// ============================================================================

#[test]
fn test_full_workflow() {
    // A complete workflow using multiple FFI functions

    // 1. Get version
    let mut ver_buf = vec![0u8; 64];
    let ver_len = unsafe { LR_Version(ver_buf.as_mut_ptr(), 64) };
    assert!(ver_len > 0);

    // 2. Initialize
    let init_result = unsafe { LR_Init() };
    assert!(init_result >= 0);

    // 3. Run OLS
    let (y, x) = simple_linear_data();
    let x_matrix = columns_to_row_major(&[x]);

    let handle = unsafe {
        LR_OLS(
            y.as_ptr(),
            y.len() as i32,
            x_matrix.as_ptr(),
            1,
        )
    };
    assert_ne!(handle, 0);

    // 4. Get results
    let r2 = unsafe { LR_GetRSquared(handle) };
    assert!(r2 > 0.9);

    let n_coef = unsafe { LR_GetNumCoefficients(handle) };
    assert_eq!(n_coef, 2);

    // 5. Clean up
    unsafe { LR_Free(handle) };
}
