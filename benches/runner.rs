//! Custom benchmark runner that aggregates results and outputs to JSON.
//!
//! This script runs all benchmark suites and produces a single JSON file
//! with versioned benchmark results.
//!
//! # Usage
//!
//! Run from the project root:
//! ```bash
//! cargo test --bench runner --release
//! ```
//!
//! Or directly:
//! ```bash
//! cargo run --benches --bench runner
//! ```
//!
//! Results will be saved to `benches/results/benches-{version}.json`

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

const CARGO_TOML: &str = include_str!("../Cargo.toml");

/// Extracts the version from Cargo.toml
fn extract_version() -> String {
    for line in CARGO_TOML.lines() {
        if line.starts_with("version = \"") {
            return line
                .strip_prefix("version = \"")
                .and_then(|s| s.strip_suffix("\""))
                .unwrap_or("0.0.0")
                .to_string();
        }
    }
    "0.0.0".to_string()
}

/// Represents the complete benchmark output structure
#[derive(serde::Serialize)]
struct BenchmarkResults {
    version: String,
    timestamp: String,
    rustc_version: String,
    target: String,
    opt_level: String,
    os: String,
    arch: String,
    benches: HashMap<String, CriterionBenchmark>,
}

/// Represents a single Criterion benchmark result
#[derive(serde::Serialize)]
struct CriterionBenchmark {
    #[serde(skip_serializing_if = "Option::is_none")]
    mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stddev: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    median: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mad: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    throughput: Option<ThroughputData>,
    unit_name: String,
}

#[derive(serde::Serialize)]
struct ThroughputData {
    #[serde(skip_serializing_if = "Option::is_none")]
    elements: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<u64>,
}

impl BenchmarkResults {
    fn new() -> Self {
        BenchmarkResults {
            version: extract_version(),
            timestamp: humantime::format_rfc3339_seconds(SystemTime::now()).to_string(),
            rustc_version: rustc_version::version()
                .map(|v| v.to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            target: std::env::var("TARGET")
                .unwrap_or_else(|_| format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)),
            opt_level: if cfg!(debug_assertions) {
                "0".to_string()
            } else {
                "3".to_string()
            },
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            benches: HashMap::new(),
        }
    }
}

/// Finds all Criterion benchmark.json files and aggregates them
fn find_benchmark_json_files(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                // Check for benchmark.json in this directory
                let benchmark_json = path.join("benchmark.json");
                if benchmark_json.exists() {
                    results.push(benchmark_json);
                }

                // Recursively search subdirectories
                results.extend(find_benchmark_json_files(&path));
            }
        }
    }

    results
}

/// Parses a Criterion benchmark.json file
fn parse_benchmark_json(
    path: &Path,
) -> Result<(String, CriterionBenchmark), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    // Extract the benchmark name from the path or JSON
    let name = json["group_name"]
        .as_str()
        .or_else(|| json["title"].as_str())
        .or_else(|| json["id"].as_str())
        .unwrap_or("unknown")
        .to_string();

    // Extract timing data
    let mean = json["mean"]["point_estimate"].as_f64();
    let stddev = json["std_dev"]["point_estimate"].as_f64();
    let median = json["median"]["point_estimate"].as_f64();
    let mad = json["mad"]["point_estimate"].as_f64();
    let unit = json["unit"]["value"].as_str().map(|s| s.to_string());
    let unit_name = json["unit"]["unit"].as_str().unwrap_or("ns").to_string();

    // Extract throughput data if present
    let throughput = json["throughput"].as_object().map(|t| ThroughputData {
        elements: t["Elements"].as_u64(),
        bytes: t["Bytes"].as_u64(),
    });

    Ok((
        name,
        CriterionBenchmark {
            mean,
            stddev,
            median,
            mad,
            unit,
            throughput,
            unit_name,
        },
    ))
}

/// Aggregates all Criterion benchmark results
fn aggregate_criterion_results(project_dir: &Path) -> HashMap<String, CriterionBenchmark> {
    let mut results = HashMap::new();
    let criterion_dir = project_dir.join("target").join("criterion");

    if !criterion_dir.exists() {
        println!("No criterion results found at {:?}", criterion_dir);
        return results;
    }

    let benchmark_files = find_benchmark_json_files(&criterion_dir);
    println!("Found {} benchmark result files", benchmark_files.len());

    for path in benchmark_files {
        match parse_benchmark_json(&path) {
            Ok((name, data)) => {
                results.insert(name, data);
            },
            Err(e) => {
                eprintln!("Failed to parse {:?}: {}", path, e);
            },
        }
    }

    results
}

fn main() {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║         linreg-core Benchmark Runner                  ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!();
    println!("Version: {}", extract_version());
    println!(
        "Target: {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!(
        "Rustc:   {}",
        rustc_version::version()
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    );
    println!();

    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let results_dir = project_dir.join("benches").join("results");
    fs::create_dir_all(&results_dir).expect("Failed to create results directory");

    let benchmark_suites = vec!["core", "linalg", "distributions", "diagnostics", "pressure"];

    println!("Running benchmark suites:");
    println!("─────────────────────────────────────────────────────────");

    for suite in &benchmark_suites {
        print!("  {:15} ... ", suite);

        let status = Command::new("cargo")
            .args(["bench", "--bench", suite])
            .output();

        match status {
            Ok(output) if output.status.success() => {
                println!("✓");
            },
            Ok(output) => {
                println!("✗ (code: {:?})", output.status.code());
                if !output.stderr.is_empty() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    for line in stderr.lines().take(5) {
                        println!("    {}", line);
                    }
                }
            },
            Err(e) => {
                println!("✗ ({})", e);
            },
        }
    }

    println!();
    println!("Aggregating results:");
    println!("─────────────────────────────────────────────────────────");

    let mut results = BenchmarkResults::new();
    results.benches = aggregate_criterion_results(project_dir);

    println!("  Parsed {} benchmark results", results.benches.len());

    // Save the aggregated results
    let version = results.version.clone();
    let output_path = results_dir.join(format!("benches-{}.json", version));

    match serde_json::to_string_pretty(&results) {
        Ok(json) => {
            fs::write(&output_path, json).expect("Failed to write results");
            println!();
            println!("Results saved to:");
            println!("  {}", output_path.display());
        },
        Err(e) => {
            println!();
            println!("Failed to serialize results: {}", e);
        },
    }

    // Print summary
    println!();
    println!("Summary:");
    println!("─────────────────────────────────────────────────────────");
    println!("  Version:          {}", results.version);
    println!("  Timestamp:        {}", results.timestamp);
    println!("  Total benchmarks: {}", results.benches.len());
    println!(
        "  Output file:      benches/results/benches-{}.json",
        results.version
    );
    println!();
}
