# ============================================================================
# Run All Diagnostic Tests (R)
# ============================================================================
# This runner script executes all diagnostic tests against all datasets in the
# datasets/csv/ directory. Results are saved to results/r/ with naming
# convention {dataset}_{test}.json
#
# Usage:
#   Rscript run_all_diagnostics_r.R [csv_dir] [output_dir] [script_dir]
#
#   Args:
#     csv_dir    Path to directory containing CSV files
#                 Default: verification/datasets/csv
#     output_dir Path to output directory for results
#                 Default: verification/results/r
#     script_dir Path to directory containing diagnostic scripts
#                 Default: verification/scripts/r/diagnostics
#
# Example:
#   Rscript run_all_diagnostics_r.R
#   Rscript run_all_diagnostics_r.R verification/datasets/csv verification/results/r
# ============================================================================

library(jsonlite)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set defaults (relative to repo root)
default_csv_dir <- "verification/datasets/csv"
default_output_dir <- "verification/results/r"
default_script_dir <- "verification/scripts/r/diagnostics"
default_regularized_dir <- "verification/scripts/r/regularized"
default_core_dir <- "verification/scripts/r/core"
default_lambda_count <- 20

csv_dir <- ifelse(length(args) >= 1, args[1], default_csv_dir)
output_dir <- ifelse(length(args) >= 2, args[2], default_output_dir)
script_dir <- ifelse(length(args) >= 3, args[3], default_script_dir)
regularized_dir <- ifelse(length(args) >= 4, args[4], default_regularized_dir)
lambda_count <- ifelse(length(args) >= 5, args[5], default_lambda_count)
core_dir <- ifelse(length(args) >= 6, args[6], default_core_dir)

# Validate CSV directory
if (!dir.exists(csv_dir)) {
  stop(paste("CSV directory not found:", csv_dir))
}

# Get list of CSV files
csv_files <- list.files(csv_dir, pattern = "\\.csv$", full.names = TRUE)

if (length(csv_files) == 0) {
  stop(paste("No CSV files found in:", csv_dir))
}

# Diagnostic scripts to run
diagnostic_scripts <- list(
  list(name = "OLS", script = "test_ols_by_dataset.R", dir = core_dir, suffix = "ols", args = ""),
  list(name = "Rainbow", script = "test_rainbow.R", dir = script_dir, suffix = "rainbow", args = ""),
  list(name = "Harvey-Collier", script = "test_harvey_collier.R", dir = script_dir, suffix = "harvey_collier", args = ""),
  list(name = "Breusch-Pagan", script = "test_breusch_pagan.R", dir = script_dir, suffix = "breusch_pagan", args = ""),
  list(name = "White", script = "test_white.R", dir = script_dir, suffix = "white", args = ""),
  list(name = "Jarque-Bera", script = "test_jarque_bera.R", dir = script_dir, suffix = "jarque_bera", args = ""),
  list(name = "Durbin-Watson", script = "test_durbin_watson.R", dir = script_dir, suffix = "durbin_watson", args = ""),
  list(name = "Breusch-Godfrey", script = "test_breusch_godfrey.R", dir = script_dir, suffix = "breusch_godfrey", args = ""),
  list(name = "Shapiro-Wilk", script = "test_shapiro_wilk.R", dir = script_dir, suffix = "shapiro_wilk", args = ""),
  list(name = "Anderson-Darling", script = "test_anderson_darling.R", dir = script_dir, suffix = "anderson_darling", args = ""),
  list(name = "Cooks-Distance", script = "test_cooks_distance.R", dir = script_dir, suffix = "cooks_distance", args = ""),
  list(name = "RESET", script = "test_reset.R", dir = script_dir, suffix = "reset", args = ""),
  list(name = "Ridge GLMNet", script = "test_ridge.R", dir = regularized_dir, suffix = "ridge_glmnet", args = lambda_count),
  list(name = "Lasso GLMNet", script = "test_lasso.R", dir = regularized_dir, suffix = "lasso_glmnet", args = lambda_count)
)

# Counter for results
total_tests <- length(csv_files) * length(diagnostic_scripts)
completed_tests <- 0
failed_tests <- 0

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("========================================================\n")
cat("Running All Diagnostic Tests (R)\n")
cat("========================================================\n")
cat("CSV Directory:", csv_dir, "\n")
cat("Output Directory:", output_dir, "\n")
cat("Script Directory:", script_dir, "\n")
cat("Regularized Directory:", regularized_dir, "\n")
cat("Core Directory:", core_dir, "\n")
cat("Datasets:", length(csv_files), "\n")
cat("Diagnostic Tests:", length(diagnostic_scripts), "\n")
cat("Total Tests to Run:", total_tests, "\n")
cat("========================================================\n\n")

# Loop through each CSV file
for (csv_file in csv_files) {
  dataset_name <- tools::file_path_sans_ext(basename(csv_file))

  cat("--- Dataset:", dataset_name, "---\n")

  # Loop through each diagnostic test
  for (diag in diagnostic_scripts) {
    test_name <- diag$name
    script_path <- file.path(diag$dir, diag$script)

    # Check if script exists
    if (!file.exists(script_path)) {
      cat("  [SKIP]", test_name, "- Script not found:", script_path, "\n")
      next
    }

    # Build command
    cmd <- paste(
      "Rscript",
      shQuote(script_path),
      shQuote(csv_file),
      shQuote(output_dir),
      diag$args
    )

    # Run test
    result <- tryCatch({
      system(cmd, intern = TRUE, ignore.stderr = TRUE)
    }, error = function(e) {
      NULL
    })

    # Check if test ran successfully
    expected_output <- file.path(output_dir, paste0(dataset_name, "_", diag$suffix, ".json"))

    if (file.exists(expected_output)) {
      cat("  [PASS]", test_name, "\n")
      completed_tests <- completed_tests + 1
    } else {
      cat("  [FAIL]", test_name, "- Output file not created\n")
      failed_tests <- failed_tests + 1
    }
  }

  cat("\n")
}

# Summary
cat("========================================================\n")
cat("Summary\n")
cat("========================================================\n")
cat("Total Tests:", total_tests, "\n")
cat("Completed:", completed_tests, "\n")
cat("Failed:", failed_tests, "\n")
cat("========================================================\n")
