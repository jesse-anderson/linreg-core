//! Neighbor finding and bandwidth computation

use crate::linalg::Matrix;
use super::weights::{EPSILON, MIN_NEIGHBORS_QUADRATIC};

/// Euclidean distance between two points
///
/// Computes the square root of the sum of squared differences between
/// corresponding coordinates of two points.
///
/// # Arguments
///
/// * `a` - First point (slice of coordinates)
/// * `b` - Second point (slice of coordinates, same length as `a`)
///
/// # Returns
///
/// The Euclidean distance between the two points
///
/// # Panics
///
/// Panics if the slices have different lengths.
///
/// # Example
///
/// ```
/// use linreg_core::loess::neighbors::euclidean_distance;
///
/// // 1D distance
/// assert_eq!(euclidean_distance(&[0.0], &[3.0]), 3.0);
///
/// // 2D distance: sqrt(3² + 4²) = 5
/// let d = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
/// assert!((d - 5.0).abs() < 1e-10);
///
/// // Distance from point to itself is 0
/// assert_eq!(euclidean_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
/// ```
#[inline]
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "Points must have the same dimensionality"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Find k nearest neighbors for a query point
///
/// Computes distances from the query point to all data points and returns
/// the indices of the k points with the smallest distances.
///
/// # Arguments
///
/// * `query` - Query point coordinates (normalized, p-dimensional)
/// * `data` - Data matrix (normalized, n obs × p predictors)
/// * `k` - Number of neighbors to find
///
/// # Returns
///
/// Vector of indices of the k nearest neighbors, sorted by distance (closest first)
///
/// # Panics
///
/// Panics if k is 0 or greater than the number of data points.
///
/// # Example
///
/// ```
/// use linreg_core::loess::neighbors::find_nearest_neighbors;
/// use linreg_core::linalg::Matrix;
///
/// // 1D data: [0, 2, 4, 6, 8]
/// let data = vec![0.0, 2.0, 4.0, 6.0, 8.0];
/// let x = Matrix::new(5, 1, data);
///
/// // Query at 3.0, find 2 nearest neighbors
/// let query = vec![3.0];
/// let neighbors = find_nearest_neighbors(&query, &x, 2);
///
/// // Should return indices [1, 2] for points [2, 4] with distances [1, 1]
/// assert_eq!(neighbors.len(), 2);
/// assert!(neighbors.contains(&1)); // Point 2.0
/// assert!(neighbors.contains(&2)); // Point 4.0
/// ```
pub fn find_nearest_neighbors(query: &[f64], data: &Matrix, k: usize) -> Vec<usize> {
    let n = data.rows;
    let p = data.cols;

    assert!(k > 0, "k must be positive");
    assert!(k <= n, "k cannot exceed number of data points");
    assert_eq!(
        query.len(),
        p,
        "Query dimension must match number of predictors"
    );

    // Compute distances to all points
    let mut distances: Vec<(usize, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let mut point = Vec::with_capacity(p);
        for j in 0..p {
            point.push(data.get(i, j));
        }
        let dist = euclidean_distance(query, &point);
        distances.push((i, dist));
    }

    // Sort by distance and take k smallest
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Return indices of k nearest neighbors
    distances.iter().take(k).map(|(idx, _)| *idx).collect()
}

/// Compute neighborhood size from span
///
/// Computes k = floor(span * n), with bounds checking.
///
/// # Arguments
///
/// * `n` - Number of data points
/// * `span` - Span parameter
/// * `degree` - Polynomial degree (affects minimum neighborhood)
///
/// # Returns
///
/// The neighborhood size k
#[inline]
pub fn compute_neighborhood_size(n: usize, span: f64, degree: usize) -> usize {
    // Use floor with epsilon for floating point safety
    let k = (span * n as f64 + EPSILON).floor() as usize;

    // Minimum depends on degree: 2 for constant/linear, 3 for quadratic
    let min_k = if degree == 2 {
        MIN_NEIGHBORS_QUADRATIC
    } else {
        2
    };

    k.max(min_k).min(n)
}

/// Compute bandwidth for local fitting
///
/// The bandwidth is determined by the distance to the k-th nearest neighbor,
/// where k = max(2, floor(span * n)). This ensures each local fit uses
/// approximately `span * n` data points.
///
/// # Arguments
///
/// * `query` - Query point (normalized, p-dimensional)
/// * `data` - Data matrix (normalized, n obs × p predictors)
/// * `span` - Span parameter (0.0 to 1.0)
/// * `degree` - Polynomial degree
///
/// # Returns
///
/// A tuple of:
/// - Bandwidth value (distance to k-th nearest neighbor)
/// - Vector of neighbor indices within the bandwidth
///
/// # Panics
///
/// Panics if span is not in (0, 1] or if data is empty.
pub fn compute_bandwidth(query: &[f64], data: &Matrix, span: f64, degree: usize) -> (f64, Vec<usize>) {
    let n = data.rows;

    assert!(n > 0, "Data matrix must have at least one row");
    assert!(
        span > 0.0 && span <= 1.0,
        "Span must be in (0, 1], got {}",
        span
    );

    // Compute k = floor(span * n) with appropriate minimum
    let k = compute_neighborhood_size(n, span, degree);

    // Find k nearest neighbors
    let neighbors = find_nearest_neighbors(query, data, k);

    // Bandwidth is distance to k-th neighbor (furthest in the neighborhood)
    let kth_idx = neighbors[k - 1];
    let p = data.cols;
    let mut kth_point = Vec::with_capacity(p);
    for j in 0..p {
        kth_point.push(data.get(kth_idx, j));
    }
    let bandwidth = euclidean_distance(query, &kth_point);

    (bandwidth, neighbors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance_1d() {
        // 1D distance
        let a = vec![0.0];
        let b = vec![3.0];
        assert_eq!(euclidean_distance(&a, &b), 3.0);
    }

    #[test]
    fn test_euclidean_distance_2d() {
        // 2D distance: sqrt(3² + 4²) = 5
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance_same_point() {
        // Distance from point to itself is 0
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(euclidean_distance(&a, &b), 0.0);
    }

    #[test]
    fn test_find_nearest_neighbors_simple() {
        // 1D data: [0, 2, 4, 6, 8]
        let data = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let x = Matrix::new(5, 1, data);

        // Query at 3.0, find 2 nearest neighbors
        let query = vec![3.0];
        let neighbors = find_nearest_neighbors(&query, &x, 2);

        // Should return indices [1, 2] for points [2, 4] with distances [1, 1]
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1)); // Point 2.0
        assert!(neighbors.contains(&2)); // Point 4.0
    }

    #[test]
    fn test_find_nearest_neighbors_2d() {
        // 2D data points with clear distance ordering
        let data = vec![
            0.0, 0.0,  // Point 0 - far
            1.0, 0.0,  // Point 1 - close
            0.0, 1.0,  // Point 2 - close
            5.0, 5.0,  // Point 3 - very far
        ];
        let x = Matrix::new(4, 2, data);

        // Query at (0.1, 0.1), find 2 nearest neighbors
        let query = vec![0.1, 0.1];
        let neighbors = find_nearest_neighbors(&query, &x, 2);

        // Nearest should be points 0 and 1
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&0)); // (0, 0) is closest
        // Second closest is either (1, 0) or (0, 1), both at similar distance
    }

    #[test]
    fn test_compute_neighborhood_size() {
        // Test neighborhood size computation
        // span = 0.5, n = 10 -> k = floor(5.0) = 5
        assert_eq!(compute_neighborhood_size(10, 0.5, 1), 5);

        // Small span: span = 0.1, n = 10 -> k = floor(1.0) = 1 -> min to 2
        assert_eq!(compute_neighborhood_size(10, 0.1, 1), 2);

        // Quadratic requires min 3
        assert_eq!(compute_neighborhood_size(10, 0.1, 2), 3);

        // Full span: span = 1.0, n = 10 -> k = 10
        assert_eq!(compute_neighborhood_size(10, 1.0, 1), 10);
    }

    #[test]
    fn test_compute_bandwidth_span() {
        // Test bandwidth computation with different span values
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x = Matrix::new(10, 1, data);
        let query = vec![5.0];

        // Span 0.5 should use k = floor(0.5 * 10) = 5 neighbors
        let (bw, neighbors) = compute_bandwidth(&query, &x, 0.5, 1);
        assert_eq!(neighbors.len(), 5);
        // Bandwidth should be distance to 5th nearest neighbor
        assert!(bw > 0.0);
    }

    #[test]
    fn test_compute_bandwidth_small_span() {
        // Small span should use few neighbors (minimum 2)
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x = Matrix::new(10, 1, data);
        let query = vec![5.0];

        // Span 0.1 would give k=1, but we enforce minimum of 2
        let (bw, neighbors) = compute_bandwidth(&query, &x, 0.1, 1);
        assert_eq!(neighbors.len(), 2);
        assert!(bw > 0.0);
    }

    #[test]
    fn test_compute_bandwidth_full_span() {
        // Full span should use all data points
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let x = Matrix::new(5, 1, data);
        let query = vec![2.0];

        let (_bw, neighbors) = compute_bandwidth(&query, &x, 1.0, 1);
        assert_eq!(neighbors.len(), 5);
        // All points should be included
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
        assert!(neighbors.contains(&4));
    }

    #[test]
    fn test_neighbors_sorted_by_distance() {
        // Neighbors should be returned sorted by distance (closest first)
        let data = vec![0.0, 10.0, 20.0, 30.0, 40.0];
        let x = Matrix::new(5, 1, data);
        let query = vec![25.0];

        let neighbors = find_nearest_neighbors(&query, &x, 3);

        // First neighbor should be closest (20.0)
        let p0 = vec![x.get(neighbors[0], 0)];
        let d0 = euclidean_distance(&query, &p0);

        // Last neighbor should be furthest
        let p2 = vec![x.get(neighbors[2], 0)];
        let d2 = euclidean_distance(&query, &p2);

        assert!(d0 <= d2);
    }
}
