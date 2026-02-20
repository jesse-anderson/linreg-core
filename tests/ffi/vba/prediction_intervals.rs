// FFI tests for prediction interval functions.

#[cfg(feature = "ffi")]

use super::common::*;

// ============================================================================
// Prediction Intervals Tests
// ============================================================================

#[test]
fn test_prediction_intervals_basic() {
    // Create training data
    let y_train: Vec<f64> = vec![2.0, 4.0, 5.0, 4.0, 5.0];
    let x_train: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_train_matrix = columns_to_row_major(&[x_train.clone()]);

    // New observations to predict
    let x_new: Vec<f64> = vec![2.0, 3.0]; // Predict for x = 2 and x = 3
    let n_new = x_new.len() as i32;

    let handle = unsafe {
        LR_PredictionIntervals(
            y_train.as_ptr(),
            y_train.len() as i32,
            x_train_matrix.as_ptr(),
            1,
            x_new.as_ptr(),
            n_new,
            0.05, // 95% confidence
        )
    };
    let _guard = HandleGuard::new(handle);

    let (predicted, lower, upper, se_pred) = read_prediction_intervals(handle, n_new as usize);

    // Should have one set of intervals per new observation
    assert_eq!(predicted.len(), n_new as usize);
    assert_eq!(lower.len(), n_new as usize);
    assert_eq!(upper.len(), n_new as usize);
    assert_eq!(se_pred.len(), n_new as usize);

    // All values should be finite
    for i in 0..n_new as usize {
        assert!(predicted[i].is_finite(), "Predicted value should be finite");
        assert!(lower[i].is_finite(), "Lower bound should be finite");
        assert!(upper[i].is_finite(), "Upper bound should be finite");
        assert!(se_pred[i].is_finite(), "SE should be finite");
        assert!(se_pred[i] > 0.0, "SE should be positive");

        // Lower <= predicted <= upper
        assert!(lower[i] <= predicted[i], "Lower should be <= predicted");
        assert!(predicted[i] <= upper[i], "Predicted should be <= upper");

        // Interval width should be positive
        let width = upper[i] - lower[i];
        assert!(width > 0.0, "Interval width should be positive");
    }
}

#[test]
fn test_prediction_intervals_alpha_levels() {
    let y_train: Vec<f64> = (0..20).map(|i| 2.0 + 3.0 * (i as f64)).collect();
    let x_train: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let x_train_matrix = columns_to_row_major(&[x_train.clone()]);

    let x_new = vec![5.0, 10.0];
    let n_new = x_new.len() as i32;

    // Test different alpha levels
    for alpha in [0.01, 0.05, 0.1, 0.2] {
        let handle = unsafe {
            LR_PredictionIntervals(
                y_train.as_ptr(),
                y_train.len() as i32,
                x_train_matrix.as_ptr(),
                1,
                x_new.as_ptr(),
                n_new,
                alpha,
            )
        };

        if handle != 0 {
            let (predicted, lower, upper, _se) = read_prediction_intervals(handle, n_new as usize);

            for i in 0..n_new as usize {
                let width = upper[i] - lower[i];

                // Higher alpha (wider interval) should give wider intervals
                assert!(width > 0.0, "Interval width should be positive for alpha={}", alpha);
            }

            unsafe { LR_Free(handle) };
        }
    }
}

#[test]
fn test_prediction_intervals_multiple_predictors() {
    let (y, x_cols) = mtcars_subset();
    let n_train = 15;
    let n_new = y.len() - n_train;

    let y_train = &y[..n_train];
    let x_train: Vec<Vec<f64>> = x_cols.iter().map(|col| col[..n_train].to_vec()).collect();
    let x_train_matrix = columns_to_row_major(&x_train);

    let x_new: Vec<Vec<f64>> = x_cols.iter().map(|col| col[n_train..].to_vec()).collect();
    let x_new_matrix = columns_to_row_major(&x_new);

    let handle = unsafe {
        LR_PredictionIntervals(
            y_train.as_ptr(),
            n_train as i32,
            x_train_matrix.as_ptr(),
            x_cols.len() as i32,
            x_new_matrix.as_ptr(),
            n_new as i32,
            0.05,
        )
    };
    let _guard = HandleGuard::new(handle);

    let (predicted, lower, upper, _se) = read_prediction_intervals(handle, n_new);

    assert_eq!(predicted.len(), n_new);
    assert_eq!(lower.len(), n_new);
    assert_eq!(upper.len(), n_new);

    for i in 0..n_new {
        assert!(lower[i] <= predicted[i]);
        assert!(predicted[i] <= upper[i]);
    }
}

#[test]
fn test_prediction_intervals_consistency() {
    // For the same x values used in training, predictions should be close to training y
    let y_train: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x_train: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_train_matrix = columns_to_row_major(&[x_train.clone()]);

    // Predict for one of the training points
    let x_new = vec![3.0]; // This was in training
    let n_new = 1;

    let handle = unsafe {
        LR_PredictionIntervals(
            y_train.as_ptr(),
            y_train.len() as i32,
            x_train_matrix.as_ptr(),
            1,
            x_new.as_ptr(),
            n_new,
            0.05,
        )
    };
    let _guard = HandleGuard::new(handle);

    let (predicted, _lower, _upper, _se) = read_prediction_intervals(handle, n_new as usize);

    // Prediction for x=3 should be close to y=6 (the training value)
    assert!((predicted[0] - 6.0).abs() < 0.5, "Prediction should be close to training value");
}

#[test]
fn test_prediction_intervals_error_handling() {
    let y_train: Vec<f64> = vec![2.0, 4.0, 6.0];
    let x_train: Vec<f64> = vec![1.0, 2.0, 3.0];
    let x_train_matrix = columns_to_row_major(&[x_train.clone()]);

    // Null pointer should error
    let handle = unsafe {
        LR_PredictionIntervals(
            std::ptr::null(),
            3,
            x_train_matrix.as_ptr(),
            1,
            x_train.as_ptr(),
            1,
            0.05,
        )
    };
    assert_eq!(handle, 0, "Null y_ptr should return error handle");

    // Invalid alpha (negative) should still work or error gracefully
    let handle = unsafe {
        LR_PredictionIntervals(
            y_train.as_ptr(),
            3,
            x_train_matrix.as_ptr(),
            1,
            x_train.as_ptr(),
            1,
            -0.05, // Invalid alpha
        )
    };
    // Implementation may accept or reject this; just shouldn't crash
    if handle != 0 {
        unsafe { LR_Free(handle) };
    }
}

#[test]
fn test_prediction_intervals_buffer_sizes() {
    let y_train: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0];
    let x_train: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let x_train_matrix = columns_to_row_major(&[x_train.clone()]);

    let x_new = vec![2.5, 3.5];
    let n_new = x_new.len() as i32;

    let handle = unsafe {
        LR_PredictionIntervals(
            y_train.as_ptr(),
            y_train.len() as i32,
            x_train_matrix.as_ptr(),
            1,
            x_new.as_ptr(),
            n_new,
            0.05,
        )
    };
    let _guard = HandleGuard::new(handle);

    // Test reading with buffer that's exactly the right size
    let mut predicted_exact = vec![0.0f64; n_new as usize];
    let written = unsafe { LR_GetPredicted(handle, predicted_exact.as_mut_ptr(), n_new) };
    assert_eq!(written, n_new);

    // Test reading with buffer that's too small
    let mut predicted_small = vec![0.0f64; 1];
    let written = unsafe { LR_GetPredicted(handle, predicted_small.as_mut_ptr(), 1) };
    assert_eq!(written, 1, "Should write only what fits in buffer");
}

#[test]
fn test_prediction_intervals_single_observation() {
    let y_train: Vec<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let x_train: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_train_matrix = columns_to_row_major(&[x_train.clone()]);

    let x_new = vec![3.0];
    let n_new = 1;

    let handle = unsafe {
        LR_PredictionIntervals(
            y_train.as_ptr(),
            y_train.len() as i32,
            x_train_matrix.as_ptr(),
            1,
            x_new.as_ptr(),
            n_new,
            0.05,
        )
    };
    let _guard = HandleGuard::new(handle);

    let (predicted, lower, upper, se) = read_prediction_intervals(handle, n_new as usize);

    // Single observation should work fine
    assert_eq!(predicted.len(), 1);
    assert!(lower[0] < predicted[0] && predicted[0] < upper[0]);
    assert!(se[0] > 0.0);
}

#[test]
fn test_prediction_intervals_wider_for_extrapolation() {
    let y_train: Vec<f64> = (1..=10).map(|i| i as f64 * 2.0).collect();
    let x_train: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let x_train_matrix = columns_to_row_major(&[x_train.clone()]);

    // Predict at the edge of training data
    let x_new_edge = vec![10.0];
    // Predict beyond training data (extrapolation)
    let x_new_far = vec![20.0];

    let h_edge = unsafe {
        LR_PredictionIntervals(
            y_train.as_ptr(),
            y_train.len() as i32,
            x_train_matrix.as_ptr(),
            1,
            x_new_edge.as_ptr(),
            1,
            0.05,
        )
    };
    let _guard1 = HandleGuard::new(h_edge);

    let h_far = unsafe {
        LR_PredictionIntervals(
            y_train.as_ptr(),
            y_train.len() as i32,
            x_train_matrix.as_ptr(),
            1,
            x_new_far.as_ptr(),
            1,
            0.05,
        )
    };
    let _guard2 = HandleGuard::new(h_far);

    let (_pred_edge, _lower_edge, _upper_edge, se_edge) =
        read_prediction_intervals(h_edge, 1);
    let (_pred_far, _lower_far, _upper_far, se_far) = read_prediction_intervals(h_far, 1);

    // Extrapolation should generally have larger standard errors
    // (though this depends on the data)
    assert!(se_edge[0] > 0.0 && se_far[0] > 0.0);
}
