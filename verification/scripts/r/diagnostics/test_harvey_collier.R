# ============================================================================
# Harvey-Collier Test Reference Implementation (R)
# ============================================================================
# This script generates reference values for the Harvey-Collier test using R's
# lmtest::harvtest function. The Harvey-Collier test checks for functional form
# misspecification by examining whether recursive residuals exhibit a linear trend.
#
# Source: lmtest package, harvtest function
# Reference: Harvey & Collier (1977), "Testing for functional misspecification
#            using the work of a higher order" (unpublished)
#
# Usage:
#   Rscript test_harvey_collier.R [csv_path] [output_dir]
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

# Run Harvey-Collier test
# harvtest parameters:
# - formula: the model formula
# - order.by: NULL uses fitted values (default for lmtest::harvtest)
hc_result <- harvtest(model, order.by = NULL)

# Print results
cat("Harvey-Collier Test (R - lmtest::harvtest)\n")
cat("=========================================\n")
cat("Dataset:", dataset_name, "\n")
cat("Formula:", formula_str, "\n")
cat("Statistic (t):", hc_result$statistic, "\n")
cat("p-value:", hc_result$p.value, "\n")
cat("Passed:", hc_result$p.value > 0.05, "\n\n")

# Prepare output
output <- list(
  test_name = "Harvey-Collier Test (R - lmtest::harvtest)",
  dataset = dataset_name,
  formula = formula_str,
  statistic = as.numeric(hc_result$statistic),
  p_value = as.numeric(hc_result$p.value),
  passed = hc_result$p.value > 0.05,
  description = "Tests for functional form misspecification by examining whether recursive residuals exhibit a linear trend. R variant uses lmtest::harvtest with order.by = fitted values."
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save to JSON with naming convention: {dataset}_harvey_collier.json
output_file <- file.path(output_dir, paste0(dataset_name, "_harvey_collier.json"))
write_json(output, output_file, pretty = TRUE, digits = 22)

cat("Results saved to:", output_file, "\n")
