# ============================================================================
# Durbin-Watson Test Reference Implementation (R)
# ============================================================================
# This script generates reference values for the Durbin-Watson test using R's
# lmtest::dwtest function. The Durbin-Watson test checks for autocorrelation
# in the residuals.
#
# Source: lmtest package, dwtest function
# Reference: Durbin & Watson (1950), "Testing for Serial Correlation in
#            Least Squares Regression: I"
#            Biometrika, Vol. 37, pp. 409-428
#            (1951), "Testing for Serial Correlation in Least Squares
#            Regression: II", Biometrika, Vol. 38, pp. 159-178
#
# Usage:
#   Rscript test_durbin_watson.R [csv_path] [output_dir]
#   Args:
#     csv_path  - Path to CSV file (first col = response, rest = predictors)
#                 Default: ../../datasets/csv/mtcars.csv
#     output_dir- Path to output directory
#                 Default: ../../results/r
# ============================================================================

library(lmtest)
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

# Run Durbin-Watson test
dw_result <- dwtest(model)

# Determine interpretation
if (dw_result$statistic > 2.5) {
  interpretation <- "Possible negative autocorrelation"
} else if (dw_result$statistic < 1.5) {
  interpretation <- "Possible positive autocorrelation"
} else {
  interpretation <- "No significant autocorrelation"
}

# Print results
cat("Durbin-Watson Test (R - lmtest::dwtest)\n")
cat("======================================\n")
cat("Dataset:", dataset_name, "\n")
cat("Formula:", formula_str, "\n")
cat("DW-statistic:", dw_result$statistic, "\n")
cat("p-value:", dw_result$p.value, "\n")
cat("Interpretation:", interpretation, "\n")
cat("Passed:", dw_result$p.value > 0.05, "\n\n")

# Prepare output
output <- list(
  test_name = "Durbin-Watson Test (R - lmtest::dwtest)",
  dataset = dataset_name,
  formula = formula_str,
  statistic = as.numeric(dw_result$statistic),
  p_value = as.numeric(dw_result$p.value),
  passed = dw_result$p.value > 0.05,
  interpretation = interpretation,
  description = "Tests for autocorrelation in residuals. Values near 2 indicate no autocorrelation, values near 0 suggest positive autocorrelation, and values near 4 suggest negative autocorrelation. Uses lmtest::dwtest."
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save to JSON with naming convention: {dataset}_durbin_watson.json
output_file <- file.path(output_dir, paste0(dataset_name, "_durbin_watson.json"))
write_json(output, output_file, pretty = TRUE, digits = 22)

cat("Results saved to:", output_file, "\n")
