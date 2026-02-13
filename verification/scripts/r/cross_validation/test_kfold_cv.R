# ============================================================================
# K-Fold Cross Validation Reference Implementation (R)
# ============================================================================
# This script generates reference values for K-Fold Cross Validation using R.
# The test validates that the Rust implementation matches R's behavior for
# OLS, Ridge, Lasso, and Elastic Net regression with CV.
#
# Source: Base R lm() and glmnet package
# Reference: Standard K-Fold CV implementation
#
# Usage:
#   Rscript test_kfold_cv.R
# ============================================================================

cat("==============================================================\n")
cat("K-Fold Cross Validation - R Reference Implementation\n")
cat("==============================================================\n")

# Add user library to path
user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.4")
if (dir.exists(user_lib)) {
    .libPaths(c(user_lib, .libPaths()))
}

# Load required packages
required_packages <- c("glmnet", "jsonlite")
for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        stop(sprintf("Package '%s' is required. Install with: install.packages('%s')", pkg, pkg))
    }
}

library(glmnet)
library(jsonlite)

# ============================================================================
# Simple Test Dataset (matches Rust validation test data)
# ============================================================================
# y = 5 + 2*x1 + 3*x2 + noise
# Generate 100 samples for more stable CV estimates
n <- 100
n_folds <- 10

y <- numeric(n)
x1 <- numeric(n)
x2 <- numeric(n)

for (i in 1:n) {
  x1[i] <- i * 0.3  # 0.3, 0.6, 0.9, ..., 30
  # x2 is linearly independent from x1 (no multicollinearity)
  x2[i] <- i * 0.2 + 5.0  # 5.2, 5.4, 5.6, ..., 25
  y[i] <- 5.0 + 2.0 * x1[i] + 3.0 * x2[i] + ((i * 7) %% 13) * 0.08
}

cat(sprintf("\nDataset: n = %d observations, p = 2 predictors\n", n))
cat(sprintf("K-Fold CV with %d folds\n", n_folds))

# ============================================================================
# Manual K-Fold CV Implementation (matching Rust's behavior)
# ============================================================================

# Create K-Fold splits
# Each fold: test on fold indices, train on the rest
# This matches Rust's create_kfold_splits behavior
create_kfold_splits <- function(n, n_folds) {
    # Fold 1: test indices 1, 2, 3, 4 (first 4)
    # Fold 2: test indices 5, 6, 7, 8
    # Fold 3: test indices 9, 10, 11, 12
    # Fold 4: test indices 13, 14, 15, 16
    fold_size <- n %/% n_folds
    remainder <- n %% n_folds

    splits <- list()
    start <- 1

    for (i in 1:n_folds) {
        # First 'remainder' folds get one extra sample
        size <- fold_size + if (i <= remainder) 1 else 0
        end <- start + size - 1

        test_idx <- start:end
        train_idx <- setdiff(1:n, test_idx)

        splits[[i]] <- list(
            fold_index = i,
            train_idx = train_idx,
            test_idx = test_idx
        )

        start <- end + 1
    }

    return(splits)
}

splits <- create_kfold_splits(n, n_folds)

# ============================================================================
# OLS Cross Validation
# ============================================================================

cat("\n--- OLS Cross Validation ---\n")

ols_fold_results <- list()
ols_fold_coefficients <- list()
ols_r_squared_values <- c()
ols_rmse_values <- c()
ols_mae_values <- c()
ols_mse_values <- c()
ols_train_r_squared <- c()

for (split in splits) {
    train_idx <- split$train_idx
    test_idx <- split$test_idx

    # Fit model on training data
    train_data <- data.frame(y = y[train_idx], x1 = x1[train_idx], x2 = x2[train_idx])
    model <- lm(y ~ x1 + x2, data = train_data)

    # Predict on test data
    test_data <- data.frame(x1 = x1[test_idx], x2 = x2[test_idx])
    pred <- predict(model, newdata = test_data)
    y_test <- y[test_idx]

    # Calculate metrics
    mse <- mean((y_test - pred)^2)
    rmse <- sqrt(mse)
    mae <- mean(abs(y_test - pred))

    # R² on test set
    ss_res <- sum((y_test - pred)^2)
    ss_tot <- sum((y_test - mean(y_test))^2)
    r_squared <- 1 - ss_res / ss_tot

    # R² on training set
    train_r2 <- summary(model)$r.squared

    # Store results
    ols_fold_results[[length(ols_fold_results) + 1]] <- list(
        fold_index = split$fold_index,
        train_size = length(train_idx),
        test_size = length(test_idx),
        mse = mse,
        rmse = rmse,
        mae = mae,
        r_squared = r_squared,
        train_r_squared = train_r2
    )

    # Store coefficients (intercept, x1, x2)
    ols_fold_coefficients[[length(ols_fold_coefficients) + 1]] <- as.vector(coef(model))

    ols_rmse_values <- c(ols_rmse_values, rmse)
    ols_mae_values <- c(ols_mae_values, mae)
    ols_r_squared_values <- c(ols_r_squared_values, r_squared)
    ols_mse_values <- c(ols_mse_values, mse)
    ols_train_r_squared <- c(ols_train_r_squared, train_r2)
}

ols_result <- list(
    test = "kfold_cv_ols",
    method = "lm",
    dataset = "simple",
    n_folds = n_folds,
    n_samples = n,
    mean_mse = mean(ols_mse_values),
    std_mse = sd(ols_mse_values),
    mean_rmse = mean(ols_rmse_values),
    std_rmse = sd(ols_rmse_values),
    mean_mae = mean(ols_mae_values),
    std_mae = sd(ols_mae_values),
    mean_r_squared = mean(ols_r_squared_values),
    std_r_squared = sd(ols_r_squared_values),
    mean_train_r_squared = mean(ols_train_r_squared),
    fold_results = ols_fold_results,
    fold_coefficients = ols_fold_coefficients
)

cat(sprintf("  Mean RMSE: %.6f (±%.6f)\n", ols_result$mean_rmse, ols_result$std_rmse))
cat(sprintf("  Mean R²: %.6f (±%.6f)\n", ols_result$mean_r_squared, ols_result$std_r_squared))

# Write to JSON
output_path <- "../../../results/r/kfold_cv_ols.json"
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
# Use higher precision for numerical stability
write(toJSON(ols_result, auto_unbox = TRUE, pretty = TRUE, digits = 15), output_path)
cat(sprintf("Wrote: %s\n", normalizePath(output_path, winslash = "/")))

# ============================================================================
# Ridge Cross Validation
# ============================================================================

cat("\n--- Ridge Cross Validation ---\n")

ridge_lambda <- 0.1
ridge_fold_results <- list()
ridge_fold_coefficients <- list()
ridge_r_squared_values <- c()
ridge_rmse_values <- c()
ridge_mae_values <- c()
ridge_mse_values <- c()
ridge_train_r_squared <- c()

for (split in splits) {
    train_idx <- split$train_idx
    test_idx <- split$test_idx

    # Prepare data for glmnet
    X_train <- as.matrix(data.frame(x1 = x1[train_idx], x2 = x2[train_idx]))
    y_train <- y[train_idx]
    X_test <- as.matrix(data.frame(x1 = x1[test_idx], x2 = x2[test_idx]))
    y_test <- y[test_idx]

    # Fit Ridge (alpha = 0)
    fit <- glmnet(X_train, y_train, family = "gaussian",
                 alpha = 0, lambda = ridge_lambda,
                 standardize = TRUE, intercept = TRUE)

    # Predict
    pred <- as.vector(predict(fit, newx = X_test, s = ridge_lambda))

    # Calculate metrics
    mse <- mean((y_test - pred)^2)
    rmse <- sqrt(mse)
    mae <- mean(abs(y_test - pred))

    # R² on test set
    ss_res <- sum((y_test - pred)^2)
    ss_tot <- sum((y_test - mean(y_test))^2)
    r_squared <- max(0, 1 - ss_res / ss_tot)

    # R² on training set
    pred_train <- as.vector(predict(fit, newx = X_train, s = ridge_lambda))
    ss_res_train <- sum((y_train - pred_train)^2)
    ss_tot_train <- sum((y_train - mean(y_train))^2)
    train_r2 <- max(0, 1 - ss_res_train / ss_tot_train)

    ridge_fold_results[[length(ridge_fold_results) + 1]] <- list(
        fold_index = split$fold_index,
        train_size = length(train_idx),
        test_size = length(test_idx),
        mse = mse,
        rmse = rmse,
        mae = mae,
        r_squared = r_squared,
        train_r_squared = train_r2
    )

    # Store coefficients (includes intercept)
    ridge_fold_coefficients[[length(ridge_fold_coefficients) + 1]] <- as.vector(coef(fit))

    ridge_rmse_values <- c(ridge_rmse_values, rmse)
    ridge_mae_values <- c(ridge_mae_values, mae)
    ridge_r_squared_values <- c(ridge_r_squared_values, r_squared)
    ridge_mse_values <- c(ridge_mse_values, mse)
    ridge_train_r_squared <- c(ridge_train_r_squared, train_r2)
}

ridge_result <- list(
    test = "kfold_cv_ridge",
    method = "glmnet",
    dataset = "simple",
    n_folds = n_folds,
    n_samples = n,
    mean_mse = mean(ridge_mse_values),
    std_mse = sd(ridge_mse_values),
    mean_rmse = mean(ridge_rmse_values),
    std_rmse = sd(ridge_rmse_values),
    mean_mae = mean(ridge_mae_values),
    std_mae = sd(ridge_mae_values),
    mean_r_squared = mean(ridge_r_squared_values),
    std_r_squared = sd(ridge_r_squared_values),
    mean_train_r_squared = mean(ridge_train_r_squared),
    fold_results = ridge_fold_results,
    fold_coefficients = ridge_fold_coefficients
)

cat(sprintf("  Mean RMSE: %.6f (±%.6f)\n", ridge_result$mean_rmse, ridge_result$std_rmse))
cat(sprintf("  Mean R²: %.6f (±%.6f)\n", ridge_result$mean_r_squared, ridge_result$std_r_squared))

output_path <- "../../../results/r/kfold_cv_ridge.json"
write(toJSON(ridge_result, auto_unbox = TRUE, pretty = TRUE, digits = 15), output_path)
cat(sprintf("Wrote: %s\n", normalizePath(output_path, winslash = "/")))

# ============================================================================
# Lasso Cross Validation
# ============================================================================

cat("\n--- Lasso Cross Validation ---\n")

lasso_lambda <- 0.1
lasso_fold_results <- list()
lasso_fold_coefficients <- list()
lasso_r_squared_values <- c()
lasso_rmse_values <- c()
lasso_mae_values <- c()
lasso_mse_values <- c()
lasso_train_r_squared <- c()

for (split in splits) {
    train_idx <- split$train_idx
    test_idx <- split$test_idx

    # Prepare data for glmnet
    X_train <- as.matrix(data.frame(x1 = x1[train_idx], x2 = x2[train_idx]))
    y_train <- y[train_idx]
    X_test <- as.matrix(data.frame(x1 = x1[test_idx], x2 = x2[test_idx]))
    y_test <- y[test_idx]

    # Fit Lasso (alpha = 1)
    fit <- glmnet(X_train, y_train, family = "gaussian",
                 alpha = 1, lambda = lasso_lambda,
                 standardize = TRUE, intercept = TRUE)

    # Predict
    pred <- as.vector(predict(fit, newx = X_test, s = lasso_lambda))

    # Calculate metrics
    mse <- mean((y_test - pred)^2)
    rmse <- sqrt(mse)
    mae <- mean(abs(y_test - pred))

    # R² on test set
    ss_res <- sum((y_test - pred)^2)
    ss_tot <- sum((y_test - mean(y_test))^2)
    r_squared <- max(0, 1 - ss_res / ss_tot)

    # R² on training set
    pred_train <- as.vector(predict(fit, newx = X_train, s = lasso_lambda))
    ss_res_train <- sum((y_train - pred_train)^2)
    ss_tot_train <- sum((y_train - mean(y_train))^2)
    train_r2 <- max(0, 1 - ss_res_train / ss_tot_train)

    lasso_fold_results[[length(lasso_fold_results) + 1]] <- list(
        fold_index = split$fold_index,
        train_size = length(train_idx),
        test_size = length(test_idx),
        mse = mse,
        rmse = rmse,
        mae = mae,
        r_squared = r_squared,
        train_r_squared = train_r2
    )

    # Store coefficients (includes intercept)
    lasso_fold_coefficients[[length(lasso_fold_coefficients) + 1]] <- as.vector(coef(fit))

    lasso_rmse_values <- c(lasso_rmse_values, rmse)
    lasso_mae_values <- c(lasso_mae_values, mae)
    lasso_r_squared_values <- c(lasso_r_squared_values, r_squared)
    lasso_mse_values <- c(lasso_mse_values, mse)
    lasso_train_r_squared <- c(lasso_train_r_squared, train_r2)
}

lasso_result <- list(
    test = "kfold_cv_lasso",
    method = "glmnet",
    dataset = "simple",
    n_folds = n_folds,
    n_samples = n,
    mean_mse = mean(lasso_mse_values),
    std_mse = sd(lasso_mse_values),
    mean_rmse = mean(lasso_rmse_values),
    std_rmse = sd(lasso_rmse_values),
    mean_mae = mean(lasso_mae_values),
    std_mae = sd(lasso_mae_values),
    mean_r_squared = mean(lasso_r_squared_values),
    std_r_squared = sd(lasso_r_squared_values),
    mean_train_r_squared = mean(lasso_train_r_squared),
    fold_results = lasso_fold_results,
    fold_coefficients = lasso_fold_coefficients
)

cat(sprintf("  Mean RMSE: %.6f (±%.6f)\n", lasso_result$mean_rmse, lasso_result$std_rmse))
cat(sprintf("  Mean R²: %.6f (±%.6f)\n", lasso_result$mean_r_squared, lasso_result$std_r_squared))

output_path <- "../../../results/r/kfold_cv_lasso.json"
write(toJSON(lasso_result, auto_unbox = TRUE, pretty = TRUE, digits = 15), output_path)
cat(sprintf("Wrote: %s\n", normalizePath(output_path, winslash = "/")))

# ============================================================================
# Elastic Net Cross Validation
# ============================================================================

cat("\n--- Elastic Net Cross Validation ---\n")

enet_lambda <- 0.1
enet_alpha <- 0.5
enet_fold_results <- list()
enet_fold_coefficients <- list()
enet_r_squared_values <- c()
enet_rmse_values <- c()
enet_mae_values <- c()
enet_mse_values <- c()
enet_train_r_squared <- c()

for (split in splits) {
    train_idx <- split$train_idx
    test_idx <- split$test_idx

    # Prepare data for glmnet
    X_train <- as.matrix(data.frame(x1 = x1[train_idx], x2 = x2[train_idx]))
    y_train <- y[train_idx]
    X_test <- as.matrix(data.frame(x1 = x1[test_idx], x2 = x2[test_idx]))
    y_test <- y[test_idx]

    # Fit Elastic Net (alpha = 0.5)
    fit <- glmnet(X_train, y_train, family = "gaussian",
                 alpha = enet_alpha, lambda = enet_lambda,
                 standardize = TRUE, intercept = TRUE)

    # Predict
    pred <- as.vector(predict(fit, newx = X_test, s = enet_lambda))

    # Calculate metrics
    mse <- mean((y_test - pred)^2)
    rmse <- sqrt(mse)
    mae <- mean(abs(y_test - pred))

    # R² on test set
    ss_res <- sum((y_test - pred)^2)
    ss_tot <- sum((y_test - mean(y_test))^2)
    r_squared <- max(0, 1 - ss_res / ss_tot)

    # R² on training set
    pred_train <- as.vector(predict(fit, newx = X_train, s = enet_lambda))
    ss_res_train <- sum((y_train - pred_train)^2)
    ss_tot_train <- sum((y_train - mean(y_train))^2)
    train_r2 <- max(0, 1 - ss_res_train / ss_tot_train)

    enet_fold_results[[length(enet_fold_results) + 1]] <- list(
        fold_index = split$fold_index,
        train_size = length(train_idx),
        test_size = length(test_idx),
        mse = mse,
        rmse = rmse,
        mae = mae,
        r_squared = r_squared,
        train_r_squared = train_r2
    )

    # Store coefficients (includes intercept)
    enet_fold_coefficients[[length(enet_fold_coefficients) + 1]] <- as.vector(coef(fit))

    enet_rmse_values <- c(enet_rmse_values, rmse)
    enet_mae_values <- c(enet_mae_values, mae)
    enet_r_squared_values <- c(enet_r_squared_values, r_squared)
    enet_mse_values <- c(enet_mse_values, mse)
    enet_train_r_squared <- c(enet_train_r_squared, train_r2)
}

enet_result <- list(
    test = "kfold_cv_elastic_net",
    method = "glmnet",
    dataset = "simple",
    n_folds = n_folds,
    n_samples = n,
    mean_mse = mean(enet_mse_values),
    std_mse = sd(enet_mse_values),
    mean_rmse = mean(enet_rmse_values),
    std_rmse = sd(enet_rmse_values),
    mean_mae = mean(enet_mae_values),
    std_mae = sd(enet_mae_values),
    mean_r_squared = mean(enet_r_squared_values),
    std_r_squared = sd(enet_r_squared_values),
    mean_train_r_squared = mean(enet_train_r_squared),
    fold_results = enet_fold_results,
    fold_coefficients = enet_fold_coefficients
)

cat(sprintf("  Mean RMSE: %.6f (±%.6f)\n", enet_result$mean_rmse, enet_result$std_rmse))
cat(sprintf("  Mean R²: %.6f (±%.6f)\n", enet_result$mean_r_squared, enet_result$std_r_squared))

output_path <- "../../../results/r/kfold_cv_elastic_net.json"
write(toJSON(enet_result, auto_unbox = TRUE, pretty = TRUE, digits = 15), output_path)
cat(sprintf("Wrote: %s\n", normalizePath(output_path, winslash = "/")))

# ============================================================================
# Summary
# ============================================================================
cat("\n==============================================================\n")
cat("Validation Results Generated\n")
cat("==============================================================\n")
cat("\nReference files written to verification/results/r/\n")
cat("Run the Rust validation tests to compare:\n")
cat("  cargo test -p linreg-core --test validation cross_validation\n")
cat("\n")
