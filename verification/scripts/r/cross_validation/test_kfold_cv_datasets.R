# ============================================================================
# K-Fold Cross Validation - CSV Dataset Reference Implementation (R)
# ============================================================================
# Generates reference values for K-Fold CV using real CSV datasets,
# matching the pattern used by other validation tests (regularized, OLS, etc.)
#
# Runs 5-fold CV for OLS, Ridge, Lasso, and Elastic Net on each dataset.
# Source: Base R lm() and glmnet package
#
# Usage:
#   Rscript test_kfold_cv_datasets.R
# ============================================================================

cat("==============================================================\n")
cat("K-Fold Cross Validation - CSV Dataset Reference\n")
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
# Configuration
# ============================================================================

n_folds <- 10
ridge_lambda <- 0.1
lasso_lambda <- 0.1
enet_lambda <- 0.1
enet_alpha <- 0.5

datasets <- c(
    "bodyfat",
    "cars_stopping",
    "faithful",
    "lh",
    "longley",
    "mtcars",
    "prostate",
    "synthetic_autocorrelated",
    "synthetic_collinear",
    "synthetic_heteroscedastic",
    "synthetic_high_vif",
    "synthetic_interaction",
    "synthetic_multiple",
    "synthetic_nonlinear",
    "synthetic_nonnormal",
    "synthetic_outliers",
    "synthetic_simple_linear",
    "synthetic_small",
    "ToothGrowth"
)

# Resolve paths relative to working directory (run from this script's directory)
datasets_dir <- "../../../datasets/csv"
output_dir <- "../../../results/r"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Helper function to convert categorical columns to numeric (R 1-based encoding)
convert_categorical_to_numeric <- function(data, dataset_name) {
    non_numeric_cols <- names(data)[sapply(data, function(x) !is.numeric(x))]
    if (length(non_numeric_cols) > 0) {
        cat(sprintf("  INFO: Converting categorical columns: %s\n",
                    paste(non_numeric_cols, collapse = ", ")))
        for (col in non_numeric_cols) {
            if (is.factor(data[[col]])) {
                data[[col]] <- as.numeric(data[[col]])
            } else if (is.character(data[[col]])) {
                temp_numeric <- as.numeric(data[[col]])
                if (any(is.na(temp_numeric))) {
                    data[[col]] <- as.numeric(as.factor(data[[col]]))
                } else {
                    data[[col]] <- temp_numeric
                }
            }
        }
    }
    return(data)
}

# ============================================================================
# K-Fold Split Function (matches Rust's create_kfold_splits)
# ============================================================================

create_kfold_splits <- function(n, n_folds) {
    fold_size <- n %/% n_folds
    remainder <- n %% n_folds

    splits <- list()
    start <- 1

    for (i in 1:n_folds) {
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

# ============================================================================
# CV Helper Functions
# ============================================================================

compute_cv_metrics <- function(y_test, pred) {
    mse <- mean((y_test - pred)^2)
    rmse <- sqrt(mse)
    mae <- mean(abs(y_test - pred))

    ss_res <- sum((y_test - pred)^2)
    ss_tot <- sum((y_test - mean(y_test))^2)
    r_squared <- if (ss_tot == 0) 0 else 1 - ss_res / ss_tot

    list(mse = mse, rmse = rmse, mae = mae, r_squared = r_squared)
}

run_ols_cv <- function(y, X, splits, dataset_name) {
    fold_results <- list()
    mse_values <- c()
    rmse_values <- c()
    mae_values <- c()
    r_squared_values <- c()
    train_r_squared_values <- c()

    df <- data.frame(y = y, X)

    for (split in splits) {
        train_data <- df[split$train_idx, ]
        test_data <- df[split$test_idx, ]

        model <- lm(y ~ ., data = train_data)
        pred <- predict(model, newdata = test_data)
        y_test <- test_data$y

        metrics <- compute_cv_metrics(y_test, pred)
        train_r2 <- summary(model)$r.squared

        fold_results[[length(fold_results) + 1]] <- list(
            fold_index = split$fold_index,
            train_size = length(split$train_idx),
            test_size = length(split$test_idx),
            mse = metrics$mse,
            rmse = metrics$rmse,
            mae = metrics$mae,
            r_squared = metrics$r_squared,
            train_r_squared = train_r2
        )

        mse_values <- c(mse_values, metrics$mse)
        rmse_values <- c(rmse_values, metrics$rmse)
        mae_values <- c(mae_values, metrics$mae)
        r_squared_values <- c(r_squared_values, metrics$r_squared)
        train_r_squared_values <- c(train_r_squared_values, train_r2)
    }

    list(
        test = "kfold_cv_ols",
        method = "lm",
        dataset = dataset_name,
        n_folds = length(splits),
        n_samples = length(y),
        mean_mse = mean(mse_values),
        std_mse = sd(mse_values),
        mean_rmse = mean(rmse_values),
        std_rmse = sd(rmse_values),
        mean_mae = mean(mae_values),
        std_mae = sd(mae_values),
        mean_r_squared = mean(r_squared_values),
        std_r_squared = sd(r_squared_values),
        mean_train_r_squared = mean(train_r_squared_values),
        fold_results = fold_results
    )
}

run_ridge_cv <- function(y, X, splits, lambda, dataset_name) {
    fold_results <- list()
    mse_values <- c()
    rmse_values <- c()
    mae_values <- c()
    r_squared_values <- c()
    train_r_squared_values <- c()

    X_matrix <- as.matrix(X)

    for (split in splits) {
        X_train <- X_matrix[split$train_idx, , drop = FALSE]
        y_train <- y[split$train_idx]
        X_test <- X_matrix[split$test_idx, , drop = FALSE]
        y_test <- y[split$test_idx]

        fit <- glmnet(X_train, y_train, family = "gaussian",
                     alpha = 0, lambda = lambda,
                     standardize = TRUE, intercept = TRUE)

        pred <- as.vector(predict(fit, newx = X_test, s = lambda))
        metrics <- compute_cv_metrics(y_test, pred)

        # Train R-squared
        pred_train <- as.vector(predict(fit, newx = X_train, s = lambda))
        ss_res_train <- sum((y_train - pred_train)^2)
        ss_tot_train <- sum((y_train - mean(y_train))^2)
        train_r2 <- max(0, 1 - ss_res_train / ss_tot_train)

        fold_results[[length(fold_results) + 1]] <- list(
            fold_index = split$fold_index,
            train_size = length(split$train_idx),
            test_size = length(split$test_idx),
            mse = metrics$mse,
            rmse = metrics$rmse,
            mae = metrics$mae,
            r_squared = metrics$r_squared,
            train_r_squared = train_r2
        )

        mse_values <- c(mse_values, metrics$mse)
        rmse_values <- c(rmse_values, metrics$rmse)
        mae_values <- c(mae_values, metrics$mae)
        r_squared_values <- c(r_squared_values, metrics$r_squared)
        train_r_squared_values <- c(train_r_squared_values, train_r2)
    }

    list(
        test = "kfold_cv_ridge",
        method = "glmnet",
        dataset = dataset_name,
        n_folds = length(splits),
        n_samples = length(y),
        mean_mse = mean(mse_values),
        std_mse = sd(mse_values),
        mean_rmse = mean(rmse_values),
        std_rmse = sd(rmse_values),
        mean_mae = mean(mae_values),
        std_mae = sd(mae_values),
        mean_r_squared = mean(r_squared_values),
        std_r_squared = sd(r_squared_values),
        mean_train_r_squared = mean(train_r_squared_values),
        fold_results = fold_results
    )
}

run_lasso_cv <- function(y, X, splits, lambda, dataset_name) {
    fold_results <- list()
    mse_values <- c()
    rmse_values <- c()
    mae_values <- c()
    r_squared_values <- c()
    train_r_squared_values <- c()

    X_matrix <- as.matrix(X)

    for (split in splits) {
        X_train <- X_matrix[split$train_idx, , drop = FALSE]
        y_train <- y[split$train_idx]
        X_test <- X_matrix[split$test_idx, , drop = FALSE]
        y_test <- y[split$test_idx]

        fit <- glmnet(X_train, y_train, family = "gaussian",
                     alpha = 1, lambda = lambda,
                     standardize = TRUE, intercept = TRUE)

        pred <- as.vector(predict(fit, newx = X_test, s = lambda))
        metrics <- compute_cv_metrics(y_test, pred)

        # Train R-squared
        pred_train <- as.vector(predict(fit, newx = X_train, s = lambda))
        ss_res_train <- sum((y_train - pred_train)^2)
        ss_tot_train <- sum((y_train - mean(y_train))^2)
        train_r2 <- max(0, 1 - ss_res_train / ss_tot_train)

        fold_results[[length(fold_results) + 1]] <- list(
            fold_index = split$fold_index,
            train_size = length(split$train_idx),
            test_size = length(split$test_idx),
            mse = metrics$mse,
            rmse = metrics$rmse,
            mae = metrics$mae,
            r_squared = metrics$r_squared,
            train_r_squared = train_r2
        )

        mse_values <- c(mse_values, metrics$mse)
        rmse_values <- c(rmse_values, metrics$rmse)
        mae_values <- c(mae_values, metrics$mae)
        r_squared_values <- c(r_squared_values, metrics$r_squared)
        train_r_squared_values <- c(train_r_squared_values, train_r2)
    }

    list(
        test = "kfold_cv_lasso",
        method = "glmnet",
        dataset = dataset_name,
        n_folds = length(splits),
        n_samples = length(y),
        mean_mse = mean(mse_values),
        std_mse = sd(mse_values),
        mean_rmse = mean(rmse_values),
        std_rmse = sd(rmse_values),
        mean_mae = mean(mae_values),
        std_mae = sd(mae_values),
        mean_r_squared = mean(r_squared_values),
        std_r_squared = sd(r_squared_values),
        mean_train_r_squared = mean(train_r_squared_values),
        fold_results = fold_results
    )
}

run_elastic_net_cv <- function(y, X, splits, lambda, alpha, dataset_name) {
    fold_results <- list()
    mse_values <- c()
    rmse_values <- c()
    mae_values <- c()
    r_squared_values <- c()
    train_r_squared_values <- c()

    X_matrix <- as.matrix(X)

    for (split in splits) {
        X_train <- X_matrix[split$train_idx, , drop = FALSE]
        y_train <- y[split$train_idx]
        X_test <- X_matrix[split$test_idx, , drop = FALSE]
        y_test <- y[split$test_idx]

        fit <- glmnet(X_train, y_train, family = "gaussian",
                     alpha = alpha, lambda = lambda,
                     standardize = TRUE, intercept = TRUE)

        pred <- as.vector(predict(fit, newx = X_test, s = lambda))
        metrics <- compute_cv_metrics(y_test, pred)

        # Train R-squared
        pred_train <- as.vector(predict(fit, newx = X_train, s = lambda))
        ss_res_train <- sum((y_train - pred_train)^2)
        ss_tot_train <- sum((y_train - mean(y_train))^2)
        train_r2 <- max(0, 1 - ss_res_train / ss_tot_train)

        fold_results[[length(fold_results) + 1]] <- list(
            fold_index = split$fold_index,
            train_size = length(split$train_idx),
            test_size = length(split$test_idx),
            mse = metrics$mse,
            rmse = metrics$rmse,
            mae = metrics$mae,
            r_squared = metrics$r_squared,
            train_r_squared = train_r2
        )

        mse_values <- c(mse_values, metrics$mse)
        rmse_values <- c(rmse_values, metrics$rmse)
        mae_values <- c(mae_values, metrics$mae)
        r_squared_values <- c(r_squared_values, metrics$r_squared)
        train_r_squared_values <- c(train_r_squared_values, train_r2)
    }

    list(
        test = "kfold_cv_elastic_net",
        method = "glmnet",
        dataset = dataset_name,
        n_folds = length(splits),
        n_samples = length(y),
        mean_mse = mean(mse_values),
        std_mse = sd(mse_values),
        mean_rmse = mean(rmse_values),
        std_rmse = sd(rmse_values),
        mean_mae = mean(mae_values),
        std_mae = sd(mae_values),
        mean_r_squared = mean(r_squared_values),
        std_r_squared = sd(r_squared_values),
        mean_train_r_squared = mean(train_r_squared_values),
        fold_results = fold_results
    )
}

# ============================================================================
# Main: Process All Datasets
# ============================================================================

for (dataset_name in datasets) {
    cat(sprintf("\n--- Dataset: %s ---\n", dataset_name))

    csv_path <- file.path(datasets_dir, paste0(dataset_name, ".csv"))
    if (!file.exists(csv_path)) {
        cat(sprintf("  SKIP: CSV not found: %s\n", csv_path))
        next
    }

    # Load data: first column = y, rest = predictors
    data <- read.csv(csv_path)
    data <- convert_categorical_to_numeric(data, dataset_name)
    y <- data[, 1]
    X <- data[, -1, drop = FALSE]
    n <- nrow(data)
    p <- ncol(X)

    cat(sprintf("  n = %d, p = %d predictors, %d folds\n", n, p, n_folds))

    # Create deterministic folds
    splits <- create_kfold_splits(n, n_folds)

    # Add dummy predictor if only 1 predictor (glmnet requires 2+ columns for ridge/lasso/enet)
    # Use a column of zeros as dummy (doesn't affect regularization)
    if (p < 2) {
      cat("  INFO: Adding dummy predictor (zeros) for regularized regression\n")
      X <- cbind(X, rep(0, n))
      colnames(X) <- c(colnames(X)[1], "dummy")
      p <- ncol(X)
    }

    # --- OLS ---
    ols_result <- run_ols_cv(y, X, splits, dataset_name)
    ols_path <- file.path(output_dir, paste0(dataset_name, "_kfold_cv_ols.json"))
    write(toJSON(ols_result, auto_unbox = TRUE, pretty = TRUE, digits = 15), ols_path)
    cat(sprintf("  OLS  - RMSE: %.6f, R2: %.6f -> %s\n",
                ols_result$mean_rmse, ols_result$mean_r_squared, basename(ols_path)))

    # --- Ridge ---
    ridge_result <- run_ridge_cv(y, X, splits, ridge_lambda, dataset_name)
    ridge_path <- file.path(output_dir, paste0(dataset_name, "_kfold_cv_ridge.json"))
    write(toJSON(ridge_result, auto_unbox = TRUE, pretty = TRUE, digits = 15), ridge_path)
    cat(sprintf("  Ridge - RMSE: %.6f, R2: %.6f -> %s\n",
                ridge_result$mean_rmse, ridge_result$mean_r_squared, basename(ridge_path)))

    # --- Lasso ---
    lasso_result <- run_lasso_cv(y, X, splits, lasso_lambda, dataset_name)
    lasso_path <- file.path(output_dir, paste0(dataset_name, "_kfold_cv_lasso.json"))
    write(toJSON(lasso_result, auto_unbox = TRUE, pretty = TRUE, digits = 15), lasso_path)
    cat(sprintf("  Lasso - RMSE: %.6f, R2: %.6f -> %s\n",
                lasso_result$mean_rmse, lasso_result$mean_r_squared, basename(lasso_path)))

    # --- Elastic Net ---
    enet_result <- run_elastic_net_cv(y, X, splits, enet_lambda, enet_alpha, dataset_name)
    enet_path <- file.path(output_dir, paste0(dataset_name, "_kfold_cv_elastic_net.json"))
    write(toJSON(enet_result, auto_unbox = TRUE, pretty = TRUE, digits = 15), enet_path)
    cat(sprintf("  ENet  - RMSE: %.6f, R2: %.6f -> %s\n",
                enet_result$mean_rmse, enet_result$mean_r_squared, basename(enet_path)))
}

# ============================================================================
# Summary
# ============================================================================
cat("\n==============================================================\n")
cat("Done! Reference files written to verification/results/r/\n")
cat("Run the Rust validation tests to compare:\n")
cat("  cargo test -p linreg-core --test validation cross_validation\n")
cat("==============================================================\n")
