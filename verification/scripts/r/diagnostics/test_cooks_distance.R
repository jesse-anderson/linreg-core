# ============================================================================
# Cook's Distance Reference Implementation (R)
# ============================================================================
# This script generates reference values for Cook's distance using R's
# stats::cooks.distance function. Cook's distance measures how much each
# observation influences the regression model.
#
# Source: stats package, cooks.distance function
# Reference: Cook, R. D. (1977), "Detection of Influential Observations in
#            Linear Regression", Technometrics, 19(1), 15-18
#
# Usage:
#   Rscript test_cooks_distance.R [csv_path] [output_dir]
#   Args:
#     csv_path  - Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     output_dir- Path to output directory
#                 Default: ../../results/r
# ============================================================================

library(stats)
library(jsonlite)

# Helper function to convert categorical columns to numeric
convert_categorical_to_numeric <- function(data, dataset_name) {
  # Check for non-numeric columns
  non_numeric_cols <- names(data)[sapply(data, function(x) !is.numeric(x))]

  if (length(non_numeric_cols) > 0) {
    cat(paste0("INFO: Dataset '", dataset_name, "' contains non-numeric columns: ",
               paste(non_numeric_cols, collapse = ", "), "\n"))
    cat("Converting categorical variables to numeric representations...\n")

    for (col in non_numeric_cols) {
      if (is.factor(data[[col]])) {
        # For factors, use integer level encoding
        unique_vals <- length(unique(data[[col]]))
        data[[col]] <- as.numeric(data[[col]])
        cat(paste0("  ", col, ": ", unique_vals, " unique values -> integer level encoding\n"))
      } else if (is.character(data[[col]])) {
        # For character columns, first try to convert to numeric directly
        # If that produces NAs, convert to factor and use integer encoding
        unique_vals <- length(unique(data[[col]]))
        temp_numeric <- as.numeric(data[[col]])
        if (any(is.na(temp_numeric))) {
          # Contains non-numeric strings, use factor encoding
          data[[col]] <- as.numeric(as.factor(data[[col]]))
          cat(paste0("  ", col, ": ", unique_vals, " unique values -> integer level encoding\n"))
        } else {
          # Successfully converted to numeric
          data[[col]] <- temp_numeric
          cat(paste0("  ", col, ": ", unique_vals, " unique values -> numeric encoding\n"))
        }
      }
    }
  }

  return(data)
}

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set defaults
default_csv <- "../../datasets/csv/mtcars.csv"
default_output <- "../../results/r"

# Parse arguments
csv_path <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir <- ifelse(length(args) >= 2, args[2], default_output)

# Validate CSV path
if (!file.exists(csv_path)) {
  stop(paste("CSV file not found:", csv_path))
}

# Extract dataset name from filename
dataset_name <- tools::file_path_sans_ext(basename(csv_path))

# Read CSV data
data <- read.csv(csv_path)

# Convert categorical columns to numeric
data <- convert_categorical_to_numeric(data, dataset_name)

# Assume first column is response variable, rest are predictors
response_col <- names(data)[1]
predictor_cols <- names(data)[-1]

# Build formula: response ~ predictor1 + predictor2 + ...
formula_str <- paste(response_col, "~", paste(predictor_cols, collapse = " + "))
formula <- as.formula(formula_str)

# Fit the model
model <- lm(formula, data = data)

# Compute Cook's distance
cooks_d <- cooks.distance(model)

# Compute model info
n <- nrow(data)
p <- length(coef(model))  # number of parameters including intercept
df_residual <- model$df.residual
mse <- sum(residuals(model)^2) / df_residual

# Compute thresholds
threshold_4_over_n <- 4.0 / n
threshold_4_over_df <- 4.0 / df_residual
threshold_1 <- 1.0

# Identify influential observations (1-based indexing)
influential_4_over_n <- which(cooks_d > threshold_4_over_n)
influential_4_over_df <- which(cooks_d > threshold_4_over_df)
influential_1 <- which(cooks_d > threshold_1)

# Find max Cook's distance
max_d <- max(cooks_d)
max_idx <- which.max(cooks_d)

# Print results
cat("Cook's Distance (R - stats::cooks.distance)\n")
cat("===========================================\n")
cat("Dataset:", dataset_name, "\n")
cat("Formula:", formula_str, "\n")
cat("n:", n, "\n")
cat("p:", p, "\n")
cat("MSE:", mse, "\n")
cat("Max Cook's D:", max_d, "(observation", max_idx, ")\n")
cat("Threshold 4/n:", threshold_4_over_n, "\n")
cat("Threshold 4/(n-p):", threshold_4_over_df, "\n")
cat("Threshold 1:", threshold_1, "\n")
cat("Influential (4/n):", length(influential_4_over_n), "observations\n")
cat("Influential (4/(n-p)):", length(influential_4_over_df), "observations\n")
cat("Influential (>1):", length(influential_1), "observations\n")
if (length(influential_1) > 0) {
  cat("Highly influential indices:", influential_1, "\n")
}
cat("\n")

# Prepare output
output <- list(
  test_name = "Cook's Distance (R - stats::cooks.distance)",
  dataset = dataset_name,
  formula = formula_str,
  distances = as.numeric(cooks_d),
  p = p,
  mse = mse,
  threshold_4_over_n = threshold_4_over_n,
  threshold_4_over_df = threshold_4_over_df,
  threshold_1 = threshold_1,
  influential_4_over_n = as.integer(influential_4_over_n),
  influential_4_over_df = as.integer(influential_4_over_df),
  influential_1 = as.integer(influential_1),
  max_distance = max_d,
  max_index = as.integer(max_idx),
  description = "Measures influence of each observation on regression coefficients. D_i > 1 indicates high influence."
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save to JSON with naming convention: {dataset}_cooks_distance.json
output_file <- file.path(output_dir, paste0(dataset_name, "_cooks_distance.json"))
write_json(output, output_file, pretty = TRUE, digits = 22)

cat("Results saved to:", output_file, "\n")
