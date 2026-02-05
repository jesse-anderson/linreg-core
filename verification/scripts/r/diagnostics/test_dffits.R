# ============================================================================
# DFFITS Reference Implementation (R)
# ============================================================================
# This script generates reference values for DFFITS using R's
# stats::dffits function. DFFITS measures the influence of each observation
# on its own fitted value.
#
# Source: stats package, dffits function
# Reference: Belsley, D. A., Kuh, E., & Welsch, R. E. (1980),
#            "Regression Diagnostics", Wiley
#
# Usage:
#   Rscript test_dffits.R [csv_path] [output_dir]
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

# Compute DFFITS
dff <- dffits(model)

# Compute model info
n <- nrow(data)
p <- length(coef(model))  # number of parameters including intercept

# Compute threshold: 2*sqrt(p/n)
threshold <- 2.0 * sqrt(p / n)

# Identify influential observations (|DFFITS| > threshold)
influential_indices <- which(abs(dff) > threshold)

# Find max absolute DFFITS value and its index
max_abs_val <- max(abs(dff))
max_idx <- which.max(abs(dff))

# Print results
cat("DFFITS (R - stats::dffits)\n")
cat("============================\n")
cat("Dataset:", dataset_name, "\n")
cat("Formula:", formula_str, "\n")
cat("n:", n, "\n")
cat("p:", p, "\n")
cat("Threshold (2*sqrt(p/n)):", threshold, "\n")
cat("Max |DFFITS|:", max_abs_val, "\n")
cat("Max index: observation", max_idx, "\n")
cat("Influential observations:", length(influential_indices), "\n")
if (length(influential_indices) > 0) {
  cat("Influential indices:", influential_indices, "\n")
} else {
  cat("Influential indices: none\n")
}
cat("\n")

# Prepare output
output <- list(
  test_name = "DFFITS (R - stats::dffits)",
  dataset = dataset_name,
  formula = formula_str,
  dffits = as.numeric(dff),
  n = n,
  p = p,
  threshold = threshold,
  influential_observations = as.integer(influential_indices),
  description = "Measures influence of each observation on its fitted value."
)

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save to JSON with naming convention: {dataset}_dffits.json
output_file <- file.path(output_dir, paste0(dataset_name, "_dffits.json"))
write_json(output, output_file, pretty = TRUE, digits = 22)

cat("Results saved to:", output_file, "\n")
