# ============================================================================
# LOESS Test Reference Implementation (R with loess)
# ============================================================================
# This script generates reference values for LOESS regression using R's
# loess() function from the stats package. The test validates that the
# Rust implementation matches R's behavior for fitted values.
#
# Source: stats::loess (R base)
# Reference: Cleveland, W. S. (1979). "Robust Locally Weighted Regression
#           and Smoothing Scatterplots". JASA, 74(368), 829-836.
#
# Usage:
#   Rscript test_loess.R [csv_path] [output_dir]
#   Args:
#     csv_path    - Path to CSV file (first col = response, rest = predictors)
#                   Default: ../../../datasets/csv/faithful.csv
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
default_csv <- "../../../datasets/csv/faithful.csv"
default_output <- "../../../results/r"

# Parse arguments
csv_path_raw <- ifelse(length(args) >= 1, args[1], default_csv)
output_dir_raw <- ifelse(length(args) >= 2, args[2], default_output)

# Resolve paths
csv_path <- resolve_path(csv_path_raw)
output_dir <- resolve_path(output_dir_raw)

# Validate CSV path
if (!file.exists(csv_path)) {
  stop(paste("CSV file not found:", csv_path))
}

# Extract dataset name from filename
dataset_name <- tools::file_path_sans_ext(basename(csv_path))

cat(sprintf("Running LOESS regression on dataset: %s\n", dataset_name))

# Load data
data <- read.csv(csv_path)
data <- convert_categorical_to_numeric(data, dataset_name)

# Extract response (first column) and predictors (remaining columns)
response_col <- names(data)[1]
predictor_cols <- names(data)[-1]
n_predictors <- length(predictor_cols)

# Use first predictor only (LOESS is primarily for single predictor smoothing)
if (n_predictors > 1) {
  cat(sprintf("Note: Using first predictor only (LOESS single-predictor focus)\n"))
}
x_col <- predictor_cols[1]
formula_str <- paste(response_col, "~", x_col)
formula <- as.formula(formula_str)

# Get x and y values
y_values <- as.vector(data[[response_col]])
x_values <- as.vector(data[[x_col]])
n <- length(y_values)

# Test configurations: 3 spans × 2 degrees × 2 surface types = 12 tests per dataset
spans <- c(0.25, 0.50, 0.75)
degrees <- c(1, 2)
surface_types <- c("interpolate", "direct")

# Ensure output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

for (surface_val in surface_types) {
  cat(sprintf("\nSurface type: %s\n", surface_val))
  cat(sprintf("─────────────────────────────────────\n", surface_val))

  for (span_val in spans) {
    for (degree_val in degrees) {
      cat(sprintf("  Testing span=%.2f degree=%d surface=%s...\n", span_val, degree_val, surface_val))

      # Fit LOESS model
      fit <- loess(formula,
                   data = data,
                   span = span_val,
                   degree = degree_val,
                   family = "gaussian",
                   surface = surface_val)

      # Extract fitted values
      fitted_values <- as.vector(fitted(fit))

      # Build result object
      result <- list(
        test = "loess",
        method = "loess",
        dataset = dataset_name,
        formula = formula_str,
        n = as.integer(n),
        n_predictors = 1L,
        span = span_val,
        degree = as.integer(degree_val),
        surface = surface_val,
        fitted = fitted_values,
        y = y_values,
        x = list(x_values)
      )

      # Write output (format span with 2 decimal places for consistency)
      span_formatted <- sprintf("%.2f", span_val)
      output_file <- file.path(output_dir, paste0(dataset_name, "_loess_", span_formatted, "_d", degree_val, "_", surface_val, ".json"))
      write_json(result, output_file, pretty = TRUE, auto_unbox = TRUE, digits = 15)

      cat(sprintf("    Wrote: %s\n", basename(output_file)))
    }
  }
}

cat(sprintf("\nDone: %s (12 outputs: 6 interpolate, 6 direct)\n", dataset_name))
