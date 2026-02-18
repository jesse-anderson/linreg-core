# ============================================================================
# Prediction Intervals Reference Implementation (R with predict.lm)
# ============================================================================
# This script generates reference values for OLS prediction intervals using
# R's predict() function with interval="prediction".
#
# Source: stats::predict.lm (R base)
#
# Usage:
#   Rscript test_prediction_intervals.R [csv_path] [output_dir]
#   Args:
#     csv_path    - Path to CSV file (first col = response, rest = predictors)
#                   Default: ../../../datasets/csv/mtcars.csv
#     output_dir  - Path to output directory
#                   Default: ../../../results/r
# ============================================================================

# Get script directory and resolve relative paths
args_all <- commandArgs(trailingOnly = FALSE)
script_arg <- args_all[grep("^--file=", args_all)]
if (length(script_arg) > 0) {
  script_path <- dirname(sub("--file=", "", script_arg))
  original_wd <- getwd()
  setwd(script_path)
} else {
  original_wd <- getwd()
}

# Add user library to path
user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "4.4")
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

# Load required packages
if (!require("jsonlite", quietly = TRUE)) {
  stop("Package 'jsonlite' is required. Install with: install.packages('jsonlite')")
}

# Helper function to convert categorical columns to numeric
convert_categorical_to_numeric <- function(data, dataset_name) {
  non_numeric_cols <- names(data)[sapply(data, function(x) !is.numeric(x))]

  if (length(non_numeric_cols) > 0) {
    cat(paste0("INFO: Dataset '", dataset_name, "' contains non-numeric columns: ",
               paste(non_numeric_cols, collapse = ", "), "\n"))
    cat("Converting categorical variables to numeric representations...\n")

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

# Resolve paths relative to original working directory
resolve_path <- function(path) {
  is_absolute <- grepl("^(/|[A-Za-z]:)", path)
  if (!is_absolute) {
    file.path(original_wd, path)
  } else {
    path
  }
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  csv_path <- resolve_path(args[1])
} else {
  csv_path <- "../../../datasets/csv/mtcars.csv"
}

if (length(args) >= 2) {
  output_dir <- resolve_path(args[2])
} else {
  output_dir <- "../../../results/r"
}

# Verify paths exist
if (!file.exists(csv_path)) {
  stop(paste("CSV file not found:", csv_path))
}

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Load and prepare data
dataset_name <- tools::file_path_sans_ext(basename(csv_path))
cat(paste0("Processing dataset: ", dataset_name, "\n"))
cat(paste0("CSV path: ", csv_path, "\n"))

data <- read.csv(csv_path)
data <- convert_categorical_to_numeric(data, dataset_name)

# First column is response, rest are predictors
y_name <- names(data)[1]
x_names <- names(data)[-1]

cat(paste0("Response: ", y_name, "\n"))
cat(paste0("Predictors: ", paste(x_names, collapse = ", "), "\n"))
cat(paste0("n = ", nrow(data), ", p = ", length(x_names), "\n"))

# Fit OLS model
formula_str <- paste(y_name, "~", paste(x_names, collapse = " + "))
model <- lm(as.formula(formula_str), data = data)

cat("\nModel summary:\n")
print(summary(model))

# ============================================================================
# Generate prediction intervals for training data (in-sample)
# ============================================================================
pi_train <- predict(model, newdata = data, interval = "prediction", level = 0.95)
se_train <- predict(model, newdata = data, se.fit = TRUE)

# Compute prediction SE: SE_pred = sqrt(residual_var * (1 + h_i))
# where residual_var = MSE and h_i is the leverage
mse <- sum(model$residuals^2) / model$df.residual
hat_values <- hatvalues(model)
se_pred_train <- sqrt(mse * (1 + hat_values))

# ============================================================================
# Generate prediction intervals for extrapolation points
# ============================================================================
# Create 3 extrapolation points: slightly beyond, moderately beyond, far beyond
n_pred <- ncol(data) - 1  # number of predictors

# Compute ranges for each predictor
x_ranges <- sapply(x_names, function(nm) {
  x <- data[[nm]]
  c(min = min(x), max = max(x), mean = mean(x), sd = sd(x))
})

# Create extrapolation points
new_data <- data.frame(matrix(nrow = 3, ncol = length(x_names)))
names(new_data) <- x_names

for (j in seq_along(x_names)) {
  nm <- x_names[j]
  x_min <- x_ranges["min", j]
  x_max <- x_ranges["max", j]
  x_range <- x_max - x_min

  # Point 1: slightly beyond max (10% extrapolation)
  new_data[1, nm] <- x_max + 0.1 * x_range
  # Point 2: moderately beyond max (50% extrapolation)
  new_data[2, nm] <- x_max + 0.5 * x_range
  # Point 3: far beyond max (100% extrapolation)
  new_data[3, nm] <- x_max + 1.0 * x_range
}

pi_new <- predict(model, newdata = new_data, interval = "prediction", level = 0.95)
se_new <- predict(model, newdata = new_data, se.fit = TRUE)

# Compute leverage for new points
# Build design matrix for new points
x_new_mat <- model.matrix(as.formula(paste("~", paste(x_names, collapse = " + "))),
                          data = new_data)
# Leverage h = diag(X_new (X'X)^{-1} X_new')
x_train_mat <- model.matrix(model)
xtx_inv <- solve(t(x_train_mat) %*% x_train_mat)
hat_new <- diag(x_new_mat %*% xtx_inv %*% t(x_new_mat))
se_pred_new <- sqrt(mse * (1 + hat_new))

cat("\n=== Training data prediction intervals (first 5) ===\n")
print(head(pi_train, 5))

cat("\n=== Extrapolation prediction intervals ===\n")
print(pi_new)
cat("Leverage (new):", hat_new, "\n")
cat("SE_pred (new):", se_pred_new, "\n")

# ============================================================================
# Build output JSON
# ============================================================================
result <- list(
  dataset = dataset_name,
  alpha = 0.05,
  n = nrow(data),
  p = length(x_names),
  df_residuals = model$df.residual,
  mse = mse,

  # Training data predictions
  train = list(
    predicted = as.numeric(pi_train[, "fit"]),
    lower = as.numeric(pi_train[, "lwr"]),
    upper = as.numeric(pi_train[, "upr"]),
    se_pred = as.numeric(se_pred_train),
    leverage = as.numeric(hat_values)
  ),

  # Extrapolation predictions
  extrapolation = list(
    new_x = as.list(new_data),
    predicted = as.numeric(pi_new[, "fit"]),
    lower = as.numeric(pi_new[, "lwr"]),
    upper = as.numeric(pi_new[, "upr"]),
    se_pred = as.numeric(se_pred_new),
    leverage = as.numeric(hat_new)
  )
)

# Write JSON output
output_file <- file.path(output_dir, paste0(dataset_name, "_prediction_intervals.json"))
json_output <- toJSON(result, auto_unbox = TRUE, pretty = TRUE, digits = 15)
writeLines(json_output, output_file)

cat(paste0("\nResults written to: ", output_file, "\n"))
cat("Done.\n")
