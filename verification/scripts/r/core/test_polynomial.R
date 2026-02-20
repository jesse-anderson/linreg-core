# ============================================================================
# Polynomial Regression Test Reference Implementation (R with lm)
# ============================================================================
# This script generates reference values for polynomial regression using R's
# lm() function with poly(x, degree, raw=TRUE). The test validates that the
# Rust implementation matches R's behavior for coefficient estimates and
# statistics.
#
# Source: stats::lm with poly(x, degree, raw=TRUE)
# Reference: "Applied Linear Statistical Models" (Kutner et al.)
#
# Usage:
#   Rscript test_polynomial.R [csv_path] [output_dir] [degree]
#   Args:
#     csv_path    - Path to CSV file (first col = response, second col = predictor)
#                   Default: ../../../datasets/csv/mtcars.csv
#     output_dir  - Path to output directory
#                   Default: ../../../results/r
#     degree      - Polynomial degree (default: 2)
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

# Set defaults
default_csv    <- "../../../datasets/csv/mtcars.csv"
default_output <- "../../../results/r"
default_degree <- 2L

# Parse arguments
csv_path_raw   <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir_raw <- ifelse(length(args) >= 2, args[2], default_output)
degree         <- as.integer(ifelse(length(args) >= 3, args[3], default_degree))

# Resolve paths
csv_path   <- resolve_path(csv_path_raw)
output_dir <- resolve_path(output_dir_raw)

# Validate CSV path
if (!file.exists(csv_path)) {
  stop(paste("CSV file not found:", csv_path))
}

if (degree < 1L) {
  stop("Polynomial degree must be at least 1")
}

# Extract dataset name from filename
dataset_name <- tools::file_path_sans_ext(basename(csv_path))

cat(sprintf("Running polynomial regression (degree=%d) test on dataset: %s\n",
            degree, dataset_name))

# Load data
data <- read.csv(csv_path)
data <- convert_categorical_to_numeric(data, dataset_name)

# Extract response (first column) and FIRST predictor (second column)
# Polynomial regression uses a single predictor
response_col  <- names(data)[1]
predictor_col <- names(data)[2]

y <- data[[response_col]]
x <- data[[predictor_col]]

n <- length(y)
k <- degree  # number of slope terms (no intercept counted)

# Require sufficient data
if (n <= degree + 1) {
  stop(sprintf("Insufficient data: n=%d, degree=%d (need n > degree+1)", n, degree))
}

# Build formula: y ~ poly(x, degree, raw=TRUE)
# raw=TRUE uses monomial basis (1, x, x^2, ...) matching our implementation
formula_str <- sprintf("%s ~ poly(%s, %d, raw=TRUE)", response_col, predictor_col, degree)
formula      <- as.formula(formula_str)

# Fit OLS model with polynomial terms
fit         <- lm(formula, data = data)
summary_fit <- summary(fit)

# Extract coefficients
coefs      <- summary_fit$coefficients
coefficients <- as.vector(coefs[, 1])   # Estimates
std_errors   <- as.vector(coefs[, 2])   # Standard errors
t_stats      <- as.vector(coefs[, 3])   # t-statistics
p_values     <- as.vector(coefs[, 4])   # p-values

# Variable names: rename poly(...) terms to match our output
variable_names <- c("Intercept", paste0("x^", seq_len(degree)))

# Model fit statistics
df_residual <- fit$df.residual
df_model    <- degree

r_squared     <- summary_fit$r.squared
adj_r_squared <- summary_fit$adj.r.squared
f_statistic   <- summary_fit$fstatistic[1]
f_p_value     <- pf(f_statistic, df_model, df_residual, lower.tail = FALSE)

# Residuals and error metrics
residuals_vec  <- as.vector(residuals(fit))
fitted_vals    <- as.vector(fitted(fit))
mse            <- sum(residuals_vec^2) / df_residual
rmse           <- sqrt(mse)
mae            <- mean(abs(residuals_vec))
residual_std_error <- summary_fit$sigma

# Model selection criteria
log_likelihood <- as.numeric(logLik(fit))
aic_val        <- AIC(fit)
bic_val        <- BIC(fit)

# Confidence intervals (95%)
ci             <- confint(fit, level = 0.95)
conf_int_lower <- as.vector(ci[, 1])
conf_int_upper <- as.vector(ci[, 2])

# Build result object
result <- list(
  test       = "polynomial",
  method     = "lm",
  dataset    = dataset_name,
  formula    = formula_str,
  degree     = degree,
  n          = n,
  k          = k,
  df_residual    = as.integer(df_residual),
  df_model       = as.integer(df_model),
  variable_names = variable_names,
  coefficients   = coefficients,
  std_errors     = std_errors,
  t_stats        = t_stats,
  p_values       = p_values,
  r_squared      = r_squared,
  adj_r_squared  = adj_r_squared,
  f_statistic    = f_statistic,
  f_p_value      = f_p_value,
  mse            = mse,
  rmse           = rmse,
  mae            = mae,
  residual_std_error = residual_std_error,
  log_likelihood = log_likelihood,
  aic            = aic_val,
  bic            = bic_val,
  conf_int_lower = conf_int_lower,
  conf_int_upper = conf_int_upper,
  fitted_values  = fitted_vals,
  residuals      = residuals_vec
)

# Ensure output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Write output
output_file <- file.path(output_dir, paste0(dataset_name, "_polynomial_degree", degree, ".json"))
write_json(result, output_file, pretty = TRUE, auto_unbox = TRUE, digits = 22)

cat(sprintf("Wrote: %s\n", normalizePath(output_file)))
