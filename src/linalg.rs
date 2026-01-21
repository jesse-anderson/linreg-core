//! Minimal Linear Algebra module to replace nalgebra dependency.
//!
//! Implements matrix operations, QR decomposition, and solvers needed for OLS.
//! Uses row-major storage for compatibility with statistical computing conventions.
//!
//! # Numerical Stability Considerations
//!
//! This implementation uses Householder QR decomposition with careful attention to
//! numerical stability:
//!
//! - **Sign convention**: Uses the numerically stable Householder sign choice
//!   (v = x + sgn(x₀)||x||e₁) to avoid cancellation
//! - **Tolerance checking**: Uses predefined tolerances to detect near-singular matrices
//! - **Zero-skipping**: Skips transformations when columns are already zero-aligned
//!
//! # Scaling Recommendations
//!
//! For optimal numerical stability when predictor variables have vastly different
//! scales (e.g., one variable in millions, another in thousandths), consider
//! standardizing predictors before regression. Z-score standardization
//! (`x_scaled = (x - mean) / std`) is already done in VIF calculation.
//!
//! However, the current implementation handles typical OLS cases without explicit
//! scaling, as QR decomposition is generally stable for well-conditioned matrices.

// ============================================================================
// Numerical Constants
// ============================================================================

/// Machine epsilon threshold for detecting zero values in QR decomposition.
/// Values below this are treated as zero to avoid numerical instability.
const QR_ZERO_TOLERANCE: f64 = 1e-12;

/// Threshold for detecting singular matrices during inversion.
/// Diagonal elements below this value indicate a near-singular matrix.
const SINGULAR_TOLERANCE: f64 = 1e-10;

/// A dense matrix stored in row-major order.
///
/// # Storage
///
/// Elements are stored in a single flat vector in row-major order:
/// `data[row * cols + col]`
///
/// # Example
///
/// ```
/// # use linreg_core::linalg::Matrix;
/// let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// assert_eq!(m.rows, 2);
/// assert_eq!(m.cols, 3);
/// assert_eq!(m.get(0, 0), 1.0);
/// assert_eq!(m.get(1, 2), 6.0);
/// ```
#[derive(Clone, Debug)]
pub struct Matrix {
    /// Number of rows in the matrix
    pub rows: usize,
    /// Number of columns in the matrix
    pub cols: usize,
    /// Flat vector storing matrix elements in row-major order
    pub data: Vec<f64>,
}

impl Matrix {
    /// Creates a new matrix from the given dimensions and data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `data` - Flat vector of elements in row-major order
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::linalg::Matrix;
    /// let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(m.get(0, 0), 1.0);
    /// assert_eq!(m.get(0, 1), 2.0);
    /// assert_eq!(m.get(1, 0), 3.0);
    /// assert_eq!(m.get(1, 1), 4.0);
    /// ```
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Data length must match dimensions");
        Matrix { rows, cols, data }
    }

    /// Creates a matrix filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::linalg::Matrix;
    /// let m = Matrix::zeros(3, 2);
    /// assert_eq!(m.rows, 3);
    /// assert_eq!(m.cols, 2);
    /// assert_eq!(m.get(1, 1), 0.0);
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    // NOTE: Currently unused but kept as reference implementation.
    // Uncomment if needed for convenience constructor.
    /*
    /// Creates a matrix from a row-major slice.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `slice` - Slice containing matrix elements in row-major order
    pub fn from_row_slice(rows: usize, cols: usize, slice: &[f64]) -> Self {
        Matrix::new(rows, cols, slice.to_vec())
    }
    */

    /// Gets the element at the specified row and column.
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// Sets the element at the specified row and column.
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    /// * `val` - Value to set
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.cols + col] = val;
    }

    /// Returns the transpose of this matrix.
    ///
    /// Swaps rows with columns: `result[col][row] = self[row][col]`.
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::linalg::Matrix;
    /// let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let t = m.transpose();
    /// assert_eq!(t.rows, 3);
    /// assert_eq!(t.cols, 2);
    /// assert_eq!(t.get(0, 1), 4.0);
    /// ```
    pub fn transpose(&self) -> Matrix {
        let mut t_data = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                t_data[c * self.rows + r] = self.get(r, c);
            }
        }
        Matrix::new(self.cols, self.rows, t_data)
    }

    /// Performs matrix multiplication: `self * other`.
    ///
    /// # Panics
    ///
    /// Panics if `self.cols != other.rows`.
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::linalg::Matrix;
    /// let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let c = a.matmul(&b);
    /// assert_eq!(c.rows, 2);
    /// assert_eq!(c.cols, 2);
    /// assert_eq!(c.get(0, 0), 22.0); // 1*1 + 2*3 + 3*5
    /// ```
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Dimension mismatch for multiplication");
        let mut result = Matrix::zeros(self.rows, other.cols);

        for r in 0..self.rows {
            for c in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(r, k) * other.get(k, c);
                }
                result.set(r, c, sum);
            }
        }
        result
    }

    /// Multiplies this matrix by a vector (treating vector as column matrix).
    ///
    /// Computes `self * vec` where vec is treated as an n×1 column matrix.
    ///
    /// # Panics
    ///
    /// Panics if `self.cols != vec.len()`.
    ///
    /// # Arguments
    ///
    /// * `vec` - Vector to multiply by
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::linalg::Matrix;
    /// let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let v = vec![1.0, 2.0, 3.0];
    /// let result = m.mul_vec(&v);
    /// assert_eq!(result.len(), 2);
    /// assert_eq!(result[0], 14.0); // 1*1 + 2*2 + 3*3
    /// ```
    #[allow(clippy::needless_range_loop)]
    pub fn mul_vec(&self, vec: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, vec.len(), "Dimension mismatch for matrix-vector multiplication");
        let mut result = vec![0.0; self.rows];

        for r in 0..self.rows {
            let mut sum = 0.0;
            for c in 0..self.cols {
                sum += self.get(r, c) * vec[c];
            }
            result[r] = sum;
        }
        result
    }

    /// Computes the dot product of a column with a vector: `Σ(data[i * cols + col] * v[i])`.
    ///
    /// For a row-major matrix, this iterates through all rows at a fixed column.
    ///
    /// # Arguments
    ///
    /// * `col` - Column index
    /// * `v` - Vector to dot with (must have length equal to rows)
    ///
    /// # Panics
    ///
    /// Panics if `col >= cols` or `v.len() != rows`.
    #[allow(clippy::needless_range_loop)]
    pub fn col_dot(&self, col: usize, v: &[f64]) -> f64 {
        assert!(col < self.cols, "Column index out of bounds");
        assert_eq!(self.rows, v.len(), "Vector length must match number of rows");

        let mut sum = 0.0;
        for row in 0..self.rows {
            sum += self.get(row, col) * v[row];
        }
        sum
    }

    /// Performs the column-vector operation in place: `v += alpha * column_col`.
    ///
    /// This is the AXPY operation where the column is treated as a vector.
    /// For row-major storage, we iterate through rows at a fixed column.
    ///
    /// # Arguments
    ///
    /// * `col` - Column index
    /// * `alpha` - Scaling factor for the column
    /// * `v` - Vector to modify in place (must have length equal to rows)
    ///
    /// # Panics
    ///
    /// Panics if `col >= cols` or `v.len() != rows`.
    #[allow(clippy::needless_range_loop)]
    pub fn col_axpy_inplace(&self, col: usize, alpha: f64, v: &mut [f64]) {
        assert!(col < self.cols, "Column index out of bounds");
        assert_eq!(self.rows, v.len(), "Vector length must match number of rows");

        for row in 0..self.rows {
            v[row] += alpha * self.get(row, col);
        }
    }

    /// Computes the squared L2 norm of a column: `Σ(data[i * cols + col]²)`.
    ///
    /// # Arguments
    ///
    /// * `col` - Column index
    ///
    /// # Panics
    ///
    /// Panics if `col >= cols`.
    #[allow(clippy::needless_range_loop)]
    pub fn col_norm2(&self, col: usize) -> f64 {
        assert!(col < self.cols, "Column index out of bounds");

        let mut sum = 0.0;
        for row in 0..self.rows {
            let val = self.get(row, col);
            sum += val * val;
        }
        sum
    }

    /// Adds a value to diagonal elements starting from a given index.
    ///
    /// This is useful for ridge regression where we add `lambda * I` to `X^T X`,
    /// but the intercept column should not be penalized.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Value to add to diagonal elements
    /// * `start_index` - Starting diagonal index (0 = first diagonal element)
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Example
    ///
    /// For a 3×3 identity matrix with intercept in first column (unpenalized):
    /// ```text
    /// add_diagonal_in_place(lambda, 1) on:
    /// [1.0, 0.0, 0.0]       [1.0,   0.0,   0.0  ]
    /// [0.0, 1.0, 0.0]  ->   [0.0,  1.0+λ, 0.0  ]
    /// [0.0, 0.0, 1.0]       [0.0,   0.0,  1.0+λ]
    /// ```
    pub fn add_diagonal_in_place(&mut self, alpha: f64, start_index: usize) {
        assert_eq!(self.rows, self.cols, "Matrix must be square");
        let n = self.rows;
        for i in start_index..n {
            let current = self.get(i, i);
            self.set(i, i, current + alpha);
        }
    }
}

// ============================================================================
// QR Decomposition
// ============================================================================

impl Matrix {
    /// Computes the QR decomposition using Householder reflections.
    ///
    /// Factorizes the matrix as `A = QR` where Q is orthogonal and R is upper triangular.
    ///
    /// # Requirements
    ///
    /// This implementation requires `rows >= cols` (tall matrix). For OLS regression,
    /// we always have more observations than predictors, so this requirement is satisfied.
    ///
    /// # Returns
    ///
    /// A tuple `(Q, R)` where:
    /// - `Q` is an orthogonal matrix (QᵀQ = I) of size m×m
    /// - `R` is an upper triangular matrix of size m×n
    #[allow(clippy::needless_range_loop)]
    pub fn qr(&self) -> (Matrix, Matrix) {
        let m = self.rows;
        let n = self.cols;
        let mut q = Matrix::identity(m);
        let mut r = self.clone();

        for k in 0..n.min(m - 1) {
            // Create vector x = R[k:, k]
            let mut x = vec![0.0; m - k];
            for i in k..m {
                x[i - k] = r.get(i, k);
            }

            // Norm of x
            let norm_x: f64 = x.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm_x < QR_ZERO_TOLERANCE { continue; } // Already zero

            // Create vector v = x + sign(x[0]) * ||x|| * e1
            //
            // NOTE: Numerical stability consideration (Householder sign choice)
            // According to Overton & Yu (2023), the numerically stable choice is
            // σ = -sgn(x₁) in the formula v = x - σ‖x‖e₁.
            //
            // This means: v = x - (-sgn(x₁))‖x‖e₁ = x + sgn(x₁)‖x‖e₁
            //
            // Equivalently: u₁ = x₁ + sgn(x₁)‖x‖
            //
            // Current implementation uses this formula (the "correct" choice for stability):
            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };  // sgn(x₀) as defined (sgn(0) = +1)
            let u1 = x[0] + sign * norm_x;
            
            // Normalize v to get Householder vector
            let mut v = x; // Re-use storage
            v[0] = u1;

            let norm_v: f64 = v.iter().map(|&val| val * val).sum::<f64>().sqrt();
            for val in &mut v { *val /= norm_v; }

            // Apply Householder transformation to R: R = H * R = (I - 2vv^T)R = R - 2v(v^T R)
            // Focus on submatrix R[k:, k:]
            for j in k..n {
                let mut dot = 0.0;
                for i in 0..m-k {
                    dot += v[i] * r.get(k+i, j);
                }
                
                for i in 0..m-k {
                    let val = r.get(k+i, j) - 2.0 * v[i] * dot;
                    r.set(k+i, j, val);
                }
            }

            // Update Q: Q = Q * H = Q(I - 2vv^T) = Q - 2(Qv)v^T
            // Focus on Q[:, k:]
            for i in 0..m {
                let mut dot = 0.0;
                for l in 0..m-k {
                    dot += q.get(i, k+l) * v[l];
                }
                
                for l in 0..m-k {
                    let val = q.get(i, k+l) - 2.0 * dot * v[l];
                    q.set(i, k+l, val);
                }
            }
        }

        (q, r)
    }

    /// Creates an identity matrix of the given size.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of rows and columns (square matrix)
    ///
    /// # Example
    ///
    /// ```
    /// # use linreg_core::linalg::Matrix;
    /// let i = Matrix::identity(3);
    /// assert_eq!(i.get(0, 0), 1.0);
    /// assert_eq!(i.get(1, 1), 1.0);
    /// assert_eq!(i.get(2, 2), 1.0);
    /// assert_eq!(i.get(0, 1), 0.0);
    /// ```
    pub fn identity(size: usize) -> Self {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }
        Matrix::new(size, size, data)
    }

    /// Inverts an upper triangular matrix (such as R from QR decomposition).
    ///
    /// Uses back-substitution to compute the inverse. This is efficient for
    /// triangular matrices compared to general matrix inversion.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Returns
    ///
    /// `None` if the matrix is singular (has a zero or near-zero diagonal element).
    /// A matrix is considered singular if any diagonal element is below the
    /// internal tolerance (1e-10), which indicates the matrix does not have full rank.
    ///
    /// # Note
    ///
    /// For upper triangular matrices, singularity is equivalent to having a
    /// zero (or near-zero) diagonal element. This is much simpler to check than
    /// for general matrices, which would require computing the condition number.
    pub fn invert_upper_triangular(&self) -> Option<Matrix> {
        let n = self.rows;
        assert_eq!(n, self.cols, "Matrix must be square");

        // Check for singularity using relative tolerance
        // This scales with the magnitude of diagonal elements, handling matrices
        // of different scales better than a fixed absolute tolerance.
        //
        // Previous implementation used absolute tolerance:
        //   if self.get(i, i).abs() < SINGULAR_TOLERANCE { return None; }
        //
        // New implementation uses relative tolerance similar to LAPACK:
        //   tolerance = max_diag * epsilon * n
        // where epsilon is machine epsilon (~2.2e-16 for f64)
        let max_diag: f64 = (0..n)
            .map(|i| self.get(i, i).abs())
            .fold(0.0_f64, |acc, val| acc.max(val));

        // Use a relative tolerance based on the maximum diagonal element
        // This is similar to LAPACK's dlamch machine epsilon approach
        let epsilon = 2.0_f64 * f64::EPSILON;  // ~4.4e-16 for f64
        let relative_tolerance = max_diag * epsilon * n as f64;
        let tolerance = SINGULAR_TOLERANCE.max(relative_tolerance);

        for i in 0..n {
            if self.get(i, i).abs() < tolerance {
                return None; // Singular matrix - cannot invert
            }
        }

        let mut inv = Matrix::zeros(n, n);

        for i in 0..n {
            inv.set(i, i, 1.0 / self.get(i, i));

            for j in (0..i).rev() {
                let mut sum = 0.0;
                for k in j+1..=i {
                    sum += self.get(j, k) * inv.get(k, i);
                }
                inv.set(j, i, -sum / self.get(j, j));
            }
        }

        Some(inv)
    }

    /// Inverts an upper triangular matrix with a custom tolerance multiplier.
    ///
    /// The tolerance is computed as `max_diag * epsilon * n * tolerance_mult`.
    /// A higher tolerance_mult allows more tolerance for near-singular matrices.
    ///
    /// # Arguments
    ///
    /// * `tolerance_mult` - Multiplier for the tolerance (1.0 = standard, higher = more tolerant)
    pub fn invert_upper_triangular_with_tolerance(&self, tolerance_mult: f64) -> Option<Matrix> {
        let n = self.rows;
        assert_eq!(n, self.cols, "Matrix must be square");

        // Check for singularity using relative tolerance
        let max_diag: f64 = (0..n)
            .map(|i| self.get(i, i).abs())
            .fold(0.0_f64, |acc, val| acc.max(val));

        // Use a relative tolerance based on the maximum diagonal element
        let epsilon = 2.0_f64 * f64::EPSILON;
        let relative_tolerance = max_diag * epsilon * n as f64 * tolerance_mult;
        let tolerance = SINGULAR_TOLERANCE.max(relative_tolerance);

        for i in 0..n {
            if self.get(i, i).abs() < tolerance {
                return None;
            }
        }

        let mut inv = Matrix::zeros(n, n);

        for i in 0..n {
            inv.set(i, i, 1.0 / self.get(i, i));

            for j in (0..i).rev() {
                let mut sum = 0.0;
                for k in j+1..=i {
                    sum += self.get(j, k) * inv.get(k, i);
                }
                inv.set(j, i, -sum / self.get(j, j));
            }
        }

        Some(inv)
    }

    /// Computes the inverse of a square matrix using QR decomposition.
    ///
    /// For an invertible matrix A, computes A⁻¹ such that A * A⁻¹ = I.
    /// Uses QR decomposition for numerical stability.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square (i.e., `self.rows != self.cols`).
    /// Check dimensions before calling if the matrix shape is not guaranteed.
    ///
    /// # Returns
    ///
    /// Returns `Some(inverse)` if the matrix is invertible, or `None` if
    /// the matrix is singular (non-invertible).
    pub fn invert(&self) -> Option<Matrix> {
        let n = self.rows;
        if n != self.cols {
            panic!("Matrix must be square for inversion");
        }

        // Use QR decomposition: A = Q * R
        let (q, r) = self.qr();

        // Compute R⁻¹ (upper triangular inverse)
        let r_inv = r.invert_upper_triangular()?;

        // A⁻¹ = R⁻¹ * Q^T
        let q_transpose = q.transpose();
        let mut result = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r_inv.get(i, k) * q_transpose.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        Some(result)
    }

    /// Computes the inverse of X'X given the QR decomposition of X (R's chol2inv).
    ///
    /// This is equivalent to computing `(X'X)^(-1)` using the QR decomposition of X.
    /// R's `chol2inv` function is used for numerical stability in recursive residuals.
    ///
    /// # Arguments
    ///
    /// * `x` - Input matrix (must have rows >= cols)
    ///
    /// # Returns
    ///
    /// `Some((X'X)^(-1))` if X has full rank, `None` otherwise.
    ///
    /// # Algorithm
    ///
    /// Given QR decomposition X = QR where R is upper triangular:
    /// 1. Extract the upper p×p portion of R (denoted R₁)
    /// 2. Invert R₁ (upper triangular inverse)
    /// 3. Compute (X'X)^(-1) = R₁^(-1) × R₁^(-T)
    ///
    /// This works because X'X = R'Q'QR = R'R, and R₁ contains the Cholesky factor.
    pub fn chol2inv_from_qr(&self) -> Option<Matrix> {
        self.chol2inv_from_qr_with_tolerance(1.0)
    }

    /// Computes the inverse of X'X given the QR decomposition with custom tolerance.
    ///
    /// Similar to `chol2inv_from_qr` but allows specifying a tolerance multiplier
    /// for handling near-singular matrices.
    ///
    /// # Arguments
    ///
    /// * `tolerance_mult` - Multiplier for the tolerance (higher = more tolerant)
    pub fn chol2inv_from_qr_with_tolerance(&self, tolerance_mult: f64) -> Option<Matrix> {
        let p = self.cols;

        // QR decomposition: X = QR
        // For X (m×n, m≥n), R is m×n upper triangular
        // The upper n×n block of R contains the meaningful values
        let (_, r_full) = self.qr();

        // Extract upper p×p portion from R
        // For tall matrices (m > p), R has zeros below diagonal in first p rows
        // For square matrices (m = p), R is p×p upper triangular
        let mut r1 = Matrix::zeros(p, p);
        for i in 0..p {
            // Row i of R1 is row i of R_full, columns 0..p
            // But we only copy the upper triangular part (columns i..p)
            for j in i..p {
                r1.set(i, j, r_full.get(i, j));
            }
            // Also copy diagonal if not yet copied
            if i < p {
                r1.set(i, i, r_full.get(i, i));
            }
        }

        // Invert R₁ (upper triangular) with custom tolerance
        let r1_inv = r1.invert_upper_triangular_with_tolerance(tolerance_mult)?;

        // Compute (X'X)^(-1) = R₁^(-1) × R₁^(-T)
        let mut result = Matrix::zeros(p, p);
        for i in 0..p {
            for j in 0..p {
                let mut sum = 0.0;
                // result[i,j] = sum(R1_inv[i,k] * R1_inv[j,k] for k=0..p)
                // R1_inv is upper triangular, but we iterate full range
                for k in 0..p {
                    sum += r1_inv.get(i, k) * r1_inv.get(j, k);
                }
                result.set(i, j, sum);
            }
        }

        Some(result)
    }
}

// ============================================================================
// Vector Helper Functions
// ============================================================================

/// Computes the arithmetic mean of a slice of f64 values.
///
/// Returns 0.0 for empty slices.
///
/// # Examples
///
/// ```
/// use linreg_core::linalg::vec_mean;
///
/// assert_eq!(vec_mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
/// assert_eq!(vec_mean(&[]), 0.0);
/// ```
///
/// # Arguments
///
/// * `v` - Slice of values
pub fn vec_mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Computes element-wise subtraction of two slices: `a - b`.
///
/// # Examples
///
/// ```
/// use linreg_core::linalg::vec_sub;
///
/// let a = vec![5.0, 4.0, 3.0];
/// let b = vec![1.0, 1.0, 1.0];
/// let result = vec_sub(&a, &b);
/// assert_eq!(result, vec![4.0, 3.0, 2.0]);
/// ```
///
/// # Arguments
///
/// * `a` - Minuend slice
/// * `b` - Subtrahend slice
///
/// # Panics
///
/// Panics if slices have different lengths.
pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "vec_sub: slice lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Computes the dot product of two slices: `Σ(a[i] * b[i])`.
///
/// # Examples
///
/// ```
/// use linreg_core::linalg::vec_dot;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// assert_eq!(vec_dot(&a, &b), 32.0);  // 1*4 + 2*5 + 3*6
/// ```
///
/// # Arguments
///
/// * `a` - First slice
/// * `b` - Second slice
///
/// # Panics
///
/// Panics if slices have different lengths.
pub fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vec_dot: slice lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes element-wise addition of two slices: `a + b`.
///
/// # Examples
///
/// ```
/// use linreg_core::linalg::vec_add;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// assert_eq!(vec_add(&a, &b), vec![5.0, 7.0, 9.0]);
/// ```
///
/// # Arguments
///
/// * `a` - First slice
/// * `b` - Second slice
///
/// # Panics
///
/// Panics if slices have different lengths.
pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "vec_add: slice lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Computes a scaled vector addition in place: `dst += alpha * src`.
///
/// This is the classic BLAS AXPY operation.
///
/// # Arguments
///
/// * `dst` - Destination slice (modified in place)
/// * `alpha` - Scaling factor for src
/// * `src` - Source slice
///
/// # Panics
///
/// Panics if slices have different lengths.
pub fn vec_axpy_inplace(dst: &mut [f64], alpha: f64, src: &[f64]) {
    assert_eq!(dst.len(), src.len(), "vec_axpy_inplace: slice lengths must match");
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += alpha * s;
    }
}

/// Scales a vector in place: `v *= alpha`.
///
/// # Arguments
///
/// * `v` - Vector to scale (modified in place)
/// * `alpha` - Scaling factor
pub fn vec_scale_inplace(v: &mut [f64], alpha: f64) {
    for val in v.iter_mut() {
        *val *= alpha;
    }
}

/// Returns a scaled copy of a vector: `v * alpha`.
///
/// # Examples
///
/// ```
/// use linreg_core::linalg::vec_scale;
///
/// let v = vec![1.0, 2.0, 3.0];
/// let scaled = vec_scale(&v, 2.5);
/// assert_eq!(scaled, vec![2.5, 5.0, 7.5]);
/// // Original is unchanged
/// assert_eq!(v, vec![1.0, 2.0, 3.0]);
/// ```
///
/// # Arguments
///
/// * `v` - Vector to scale
/// * `alpha` - Scaling factor
pub fn vec_scale(v: &[f64], alpha: f64) -> Vec<f64> {
    v.iter().map(|&x| x * alpha).collect()
}

/// Computes the L2 norm (Euclidean norm) of a vector: `sqrt(Σ(v[i]²))`.
///
/// # Examples
///
/// ```
/// use linreg_core::linalg::vec_l2_norm;
///
/// // Pythagorean triple: 3-4-5
/// assert_eq!(vec_l2_norm(&[3.0, 4.0]), 5.0);
/// // Unit vector
/// assert_eq!(vec_l2_norm(&[1.0, 0.0, 0.0]), 1.0);
/// ```
///
/// # Arguments
///
/// * `v` - Vector slice
pub fn vec_l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Computes the maximum absolute value in a vector.
///
/// # Arguments
///
/// * `v` - Vector slice
pub fn vec_max_abs(v: &[f64]) -> f64 {
    v.iter().map(|&x| x.abs()).fold(0.0_f64, f64::max)
}

// ============================================================================
// R-Compatible QR Decomposition (LINPACK dqrdc2 with Column Pivoting)
// ============================================================================

/// QR decomposition result using R's LINPACK dqrdc2 algorithm.
///
/// This implements the QR decomposition with column pivoting as used by R's
/// `qr()` function with `LAPACK=FALSE`. The algorithm is a modification of
/// LINPACK's DQRDC that:
/// - Uses Householder transformations
/// - Implements limited column pivoting based on 2-norms of reduced columns
/// - Moves columns with near-zero norm to the right-hand edge
/// - Computes the rank (number of linearly independent columns)
///
/// # Fields
///
/// * `qr` - The QR factorization (upper triangle contains R, below diagonal
///   contains Householder vector information)
/// * `qraux` - Auxiliary information for recovering the orthogonal part Q
/// * `pivot` - Column permutation: `pivot\[j\]` contains the original column index
///   now in column j
/// * `rank` - Number of linearly independent columns (the computed rank)
#[derive(Clone, Debug)]
pub struct QRLinpack {
    /// QR factorization matrix (same dimensions as input)
    pub qr: Matrix,
    /// Auxiliary information for Q recovery
    pub qraux: Vec<f64>,
    /// Column pivot vector (1-based indices like R)
    pub pivot: Vec<usize>,
    /// Computed rank (number of linearly independent columns)
    pub rank: usize,
}

impl Matrix {
    /// Computes QR decomposition using R's LINPACK dqrdc2 algorithm with column pivoting.
    ///
    /// This is a port of R's dqrdc2.f, which is a modification of LINPACK's DQRDC.
    /// The algorithm:
    /// 1. Uses Householder transformations for QR factorization
    /// 2. Implements limited column pivoting based on column 2-norms
    /// 3. Moves columns with near-zero norm to the right-hand edge
    /// 4. Computes the rank (number of linearly independent columns)
    ///
    /// # Arguments
    ///
    /// * `tol` - Tolerance for determining linear independence. Default is 1e-7 (R's default).
    ///   Columns with norm < tol * original_norm are considered negligible.
    ///
    /// # Returns
    ///
    /// A [`QRLinpack`] struct containing the QR factorization, auxiliary information,
    /// pivot vector, and computed rank.
    ///
    /// # Algorithm Details
    ///
    /// The decomposition is A * P = Q * R where:
    /// - P is the permutation matrix coded by `pivot`
    /// - Q is orthogonal (m × m)
    /// - R is upper triangular in the first `rank` rows
    ///
    /// The `qr` matrix contains:
    /// - Upper triangle: R matrix (if pivoting was performed, this is R of the permuted matrix)
    /// - Below diagonal: Householder vector information
    ///
    /// # Reference
    ///
    /// - R source: src/appl/dqrdc2.f
    /// - LINPACK documentation: <https://www.netlib.org/linpack/dqrdc.f>
    pub fn qr_linpack(&self, tol: Option<f64>) -> QRLinpack {
        let n = self.rows;
        let p = self.cols;
        let lup = n.min(p);

        // Default tolerance matches R's qr.default: tol = 1e-07
        let tol = tol.unwrap_or(1e-07);

        // Initialize working matrices
        let mut x = self.clone(); // Working copy that will be modified
        let mut qraux = vec![0.0; p];
        let mut pivot: Vec<usize> = (1..=p).collect(); // 1-based indices like R
        let mut work = vec![(0.0, 0.0); p]; // (work[j,1], work[j,2])

        // Compute the norms of the columns of x (initialization)
        if n > 0 {
            for j in 0..p {
                let mut norm = 0.0;
                for i in 0..n {
                    norm += x.get(i, j) * x.get(i, j);
                }
                norm = norm.sqrt();
                qraux[j] = norm;
                let original_norm = if norm == 0.0 { 1.0 } else { norm };
                work[j] = (norm, original_norm);
            }
        }

        let mut k = p + 1; // Will be decremented to get the final rank

        // Perform the Householder reduction of x
        for l in 0..lup {
            // Cycle columns from l to p until one with non-negligible norm is found
            // A column is negligible if its norm has fallen below tol * original_norm
            while l < k - 1 && qraux[l] < work[l].1 * tol {
                // Move column l to the end (it's negligible)
                let lp1 = l + 1;

                // Shift columns in x: x(i, l..p-1) = x(i, l+1..p)
                for i in 0..n {
                    let t = x.get(i, l);
                    for j in lp1..p {
                        x.set(i, j - 1, x.get(i, j));
                    }
                    x.set(i, p - 1, t);
                }

                // Shift pivot, qraux, and work arrays
                let saved_pivot = pivot[l];
                let saved_qraux = qraux[l];
                let saved_work = work[l];

                for j in lp1..p {
                    pivot[j - 1] = pivot[j];
                    qraux[j - 1] = qraux[j];
                    work[j - 1] = work[j];
                }

                pivot[p - 1] = saved_pivot;
                qraux[p - 1] = saved_qraux;
                work[p - 1] = saved_work;

                k -= 1;
            }

            if l == n - 1 {
                // Last row - skip transformation
                break;
            }

            // Compute the Householder transformation for column l
            // nrmxl = norm of x[l:, l]
            let mut nrmxl = 0.0;
            for i in l..n {
                let val = x.get(i, l);
                nrmxl += val * val;
            }
            nrmxl = nrmxl.sqrt();

            if nrmxl == 0.0 {
                // Zero column - continue to next
                continue;
            }

            // Apply sign for numerical stability (dsign in Fortran)
            let x_ll = x.get(l, l);
            if x_ll != 0.0 {
                nrmxl = nrmxl.copysign(x_ll);
            }

            // Scale the column
            let scale = 1.0 / nrmxl;
            for i in l..n {
                x.set(i, l, x.get(i, l) * scale);
            }
            x.set(l, l, 1.0 + x.get(l, l));

            // Apply the transformation to remaining columns, updating the norms
            let lp1 = l + 1;
            if p > lp1 {
                for j in lp1..p {
                    // Compute t = -dot(x[l:, l], x[l:, j]) / x(l, l)
                    let mut dot = 0.0;
                    for i in l..n {
                        dot += x.get(i, l) * x.get(i, j);
                    }
                    let t = -dot / x.get(l, l);

                    // x[l:, j] = x[l:, j] + t * x[l:, l]
                    for i in l..n {
                        let val = x.get(i, j) + t * x.get(i, l);
                        x.set(i, j, val);
                    }

                    // Update the norm
                    if qraux[j] != 0.0 {
                        // tt = 1.0 - (x(l, j) / qraux[j])^2
                        let x_lj = x.get(l, j).abs();
                        let mut tt = 1.0 - (x_lj / qraux[j]).powi(2);
                        tt = tt.max(0.0);

                        // Recompute norm if there is large reduction (BDR mod 9/99)
                        // The tolerance here is on the squared norm
                        if tt.abs() < 1e-6 {
                            // Re-compute norm directly
                            let mut new_norm = 0.0;
                            for i in (l + 1)..n {
                                let val = x.get(i, j);
                                new_norm += val * val;
                            }
                            new_norm = new_norm.sqrt();
                            qraux[j] = new_norm;
                            work[j].0 = new_norm;
                        } else {
                            qraux[j] = qraux[j] * tt.sqrt();
                        }
                    }
                }
            }

            // Save the transformation
            qraux[l] = x.get(l, l);
            x.set(l, l, -nrmxl);
        }

        // Compute final rank
        let rank = k - 1;
        let rank = rank.min(n);

        QRLinpack {
            qr: x,
            qraux,
            pivot,
            rank,
        }
    }

    /// Solves a linear system using the QR decomposition with column pivoting.
    ///
    /// This implements a least squares solver using the pivoted QR decomposition.
    /// For rank-deficient cases, coefficients corresponding to linearly dependent
    /// columns are set to `f64::NAN`.
    ///
    /// # Arguments
    ///
    /// * `qr_result` - QR decomposition from [`Matrix::qr_linpack`]
    /// * `y` - Right-hand side vector
    ///
    /// # Returns
    ///
    /// A vector of coefficients, or `None` if the system is exactly singular.
    ///
    /// # Algorithm
    ///
    /// This solver uses the standard QR decomposition approach:
    /// 1. Compute the QR decomposition of the permuted matrix
    /// 2. Extract R matrix (upper triangular with positive diagonal)
    /// 3. Compute qty = Q^T * y
    /// 4. Solve R * coef = qty using back substitution
    /// 5. Apply the pivot permutation to restore original column order
    ///
    /// # Note
    ///
    /// The LINPACK QR algorithm stores R with mixed signs on the diagonal.
    /// This solver corrects for that by taking the absolute value of R's diagonal.
    pub fn qr_solve_linpack(&self, qr_result: &QRLinpack, y: &[f64]) -> Option<Vec<f64>> {
        let n = self.rows;
        let p = self.cols;
        let k = qr_result.rank;

        if y.len() != n {
            return None;
        }

        if k == 0 {
            return None;
        }

        // Step 1: Compute Q^T * y using the Householder vectors directly
        // This is more efficient than reconstructing the full Q matrix
        let mut qty = y.to_vec();

        for j in 0..k {
            // Check if this Householder transformation is valid
            let r_jj = qr_result.qr.get(j, j);
            if r_jj == 0.0 {
                continue;
            }

            // Compute dot = v_j^T * qty[j:]
            // where v_j is the Householder vector stored in qr[j:, j]
            // The storage convention:
            // - qr[j,j] = -nrmxl (after final overwrite)
            // - qr[i,j] for i > j is the scaled Householder vector element
            // - qraux[j] = 1 + original_x[j,j]/nrmxl (the unscaled first element)

            // Reconstruct the Householder vector v_j
            // After scaling by 1/nrmxl, we have:
            // v_scaled[j] = 1 + x[j,j]/nrmxl
            // v_scaled[i] = x[i,j]/nrmxl for i > j
            // The actual unit vector is v = v_scaled / ||v_scaled||

            let mut v = vec![0.0; n - j];
            // Copy the scaled Householder vector from qr
            for i in j..n {
                v[i - j] = qr_result.qr.get(i, j);
            }

            // The j-th element was modified during the QR decomposition
            // We need to reconstruct it from qraux
            let alpha = qr_result.qraux[j];
            if alpha != 0.0 {
                v[0] = alpha;
            }

            // Compute the norm of v
            let v_norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if v_norm < 1e-14 {
                continue;
            }

            // Compute dot = v^T * qty[j:]
            let mut dot = 0.0;
            for i in j..n {
                dot += v[i - j] * qty[i];
            }

            // Apply Householder transformation: qty[j:] = qty[j:] - 2 * v * (v^T * qty[j:]) / (v^T * v)
            // Since v is already scaled, we use: t = 2 * dot / (v_norm^2)
            let t = 2.0 * dot / (v_norm * v_norm);

            for i in j..n {
                qty[i] -= t * v[i - j];
            }
        }

        // Step 2: Back substitution on R (solve R * coef = qty)
        // The R matrix is stored in the upper triangle of qr
        // Note: The diagonal elements of R are negative (from -nrmxl)
        // We use them as-is since the signs cancel out in the computation
        let mut coef_permuted = vec![f64::NAN; p];

        for row in (0..k).rev() {
            let r_diag = qr_result.qr.get(row, row);
            // Use relative tolerance for singularity check
            let max_abs = (0..k).map(|i| qr_result.qr.get(i, i).abs()).fold(0.0_f64, f64::max);
            let tolerance = 1e-14 * max_abs.max(1.0);

            if r_diag.abs() < tolerance {
                return None;  // Singular
            }

            let mut sum = qty[row];
            for col in (row + 1)..k {
                sum -= qr_result.qr.get(row, col) * coef_permuted[col];
            }
            coef_permuted[row] = sum / r_diag;
        }

        // Step 3: Apply pivot permutation to get coefficients in original order
        // pivot[j] is 1-based, indicating which original column is now in position j
        let mut result = vec![0.0; p];
        for j in 0..p {
            let original_col = qr_result.pivot[j] - 1;  // Convert to 0-based
            result[original_col] = coef_permuted[j];
        }

        Some(result)
    }
}

/// Performs OLS regression using R's LINPACK QR algorithm.
///
/// This function is a drop-in replacement for `fit_ols` that uses the
/// R-compatible QR decomposition with column pivoting. It handles
/// rank-deficient matrices more gracefully than the standard QR decomposition.
///
/// # Arguments
///
/// * `y` - Response variable (n observations)
/// * `x` - Design matrix (n rows, p columns including intercept)
///
/// # Returns
///
/// * `Some(Vec<f64>)` - OLS coefficient vector (p elements)
/// * `None` - If the matrix is exactly singular or dimensions don't match
///
/// # Note
///
/// For rank-deficient systems, this function uses the pivoted QR which
/// automatically handles multicollinearity by selecting a linearly
/// independent subset of columns.
pub fn fit_ols_linpack(y: &[f64], x: &Matrix) -> Option<Vec<f64>> {
    let qr_result = x.qr_linpack(None);
    x.qr_solve_linpack(&qr_result, y)
}

/// Fits OLS and predicts using R's LINPACK QR with rank-deficient handling.
///
/// This function matches R's `lm.fit` behavior for rank-deficient cases:
/// coefficients for linearly dependent columns are set to NA, and predictions
/// are computed using only the valid (non-NA) coefficients and their corresponding
/// columns. This matches how R handles rank-deficient models in prediction.
///
/// # Arguments
///
/// * `y` - Response variable (n observations)
/// * `x` - Design matrix (n rows, p columns including intercept)
///
/// # Returns
///
/// * `Some(Vec<f64>)` - Predictions (n elements)
/// * `None` - If the matrix is exactly singular or dimensions don't match
///
/// # Algorithm
///
/// For rank-deficient systems (rank < p):
/// 1. Compute QR decomposition with column pivoting
/// 2. Get coefficients (rank-deficient columns will have NaN)
/// 3. Build a reduced design matrix with only pivoted, non-singular columns
/// 4. Compute predictions using only the valid columns
///
/// This matches R's behavior where `predict(lm.fit(...))` handles NA coefficients
/// by excluding the corresponding columns from the prediction.
pub fn fit_and_predict_linpack(y: &[f64], x: &Matrix) -> Option<Vec<f64>> {
    let n = x.rows;
    let p = x.cols;

    // Compute QR decomposition
    let qr_result = x.qr_linpack(None);
    let k = qr_result.rank;

    // Solve for coefficients
    let beta_permuted = x.qr_solve_linpack(&qr_result, y)?;

    // Check for rank deficiency
    if k == p {
        // Full rank - use standard prediction
        return Some(x.mul_vec(&beta_permuted));
    }

    // Rank-deficient case: some columns are collinear and have NaN coefficients
    // We compute predictions using only columns with valid (non-NaN) coefficients
    // This matches R's behavior where NA coefficients exclude columns from prediction

    let mut pred = vec![0.0; n];

    for row in 0..n {
        let mut sum = 0.0;
        for j in 0..p {
            let b_val = beta_permuted[j];
            if b_val.is_nan() {
                continue;  // Skip collinear columns (matches R's NA coefficient behavior)
            }
            sum += x.get(row, j) * b_val;
        }
        pred[row] = sum;
    }

    Some(pred)
}
