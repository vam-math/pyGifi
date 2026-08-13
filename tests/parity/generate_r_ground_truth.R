## tests/parity/generate_r_ground_truth.R
##
## Runs the real CRAN `Gifi` package (homals / princals / morals) on every
## dataset + method listed in tests/parity/manifest.json, and writes one JSON
## result file per (dataset, method) pair to results/r_ground_truth/.
##
## R's gifiEngine() always seeds its internal starting matrix with
## set.seed(123) (hardcoded in the package, not configurable): this is the
## same seed pygifi's r_seed=123 (pygifi_rng extension) reproduces bit-for-bit
## via the ported RNG algorithm, so no manual init_x hand-off between R and
## Python is needed for exact-parity runs.
##
## Usage (from repo root):
##   Rscript tests/parity/generate_r_ground_truth.R

.libPaths("C:/Users/Vam/R/win-library/4.6")
suppressPackageStartupMessages({
  library(Gifi)
  library(jsonlite)
  library(readxl)
})

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args_full[grep("^--file=", args_full)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[1], winslash = "/")) else getwd()
ROOT <- normalizePath(file.path(script_dir, "..", ".."), winslash = "/")
MANIFEST <- fromJSON(file.path(ROOT, "tests", "parity", "manifest.json"))
OUT_DIR <- file.path(ROOT, "results", "r_ground_truth")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

write_result <- function(name, method, payload) {
  payload$dataset <- name
  payload$method <- method
  path <- file.path(OUT_DIR, paste0(name, "_", method, ".json"))
  write_json(payload, path, digits = 10, null = "null", auto_unbox = TRUE)
  cat(sprintf("  [OK] %s\n", basename(path)))
}

write_failure <- function(name, method, err) {
  path <- file.path(OUT_DIR, paste0(name, "_", method, ".json"))
  write_json(list(dataset = name, method = method, error = conditionMessage(err)),
             path, auto_unbox = TRUE)
  cat(sprintf("  [FAIL] %s: %s\n", basename(path), conditionMessage(err)))
}

run_homals <- function(name, df) {
  tryCatch({
    fit <- homals(df, ndim = 2, eps = 1e-8, itmax = 1000)
    write_result(name, "homals", list(
      evals = fit$evals, f = fit$f, ntel = fit$ntel,
      quantifications = fit$quantifications,
      scoremat = fit$scoremat, dmeasures = fit$dmeasures
    ))
  }, error = function(e) write_failure(name, "homals", e))
}

run_princals <- function(name, df) {
  tryCatch({
    fit <- princals(df, ndim = 2, eps = 1e-8, itmax = 1000)
    write_result(name, "princals", list(
      evals = fit$evals, f = fit$f, ntel = fit$ntel,
      loadings = fit$loadings, lambda = fit$lambda,
      quantifications = fit$quantifications,
      scoremat = fit$scoremat, dmeasures = fit$dmeasures
    ))
  }, error = function(e) write_failure(name, "princals", e))
}

run_morals <- function(name, x, y) {
  tryCatch({
    fit <- morals(x, y, eps = 1e-8, itmax = 1000)
    write_result(name, "morals", list(
      beta = as.vector(fit$beta), smc = fit$smc, evals = fit$evals,
      yhat = as.vector(fit$yhat), xhat = fit$xhat,
      ntel = fit$ntel, f = fit$f
    ))
  }, error = function(e) write_failure(name, "morals", e))
}

## ── Classic datasets: homals + princals ──────────────────────────────────
cat("== Classic datasets (homals, princals) ==\n")
DATA_DIR <- file.path(ROOT, "pygifi", "data")
for (name in MANIFEST$classic_datasets) {
  cat(sprintf("-- %s --\n", name))
  data(list = name)
  df <- get(name)
  # Re-export the authoritative R data as CSV so pygifi.get_dataset() and the
  # Python side of this pipeline read the exact same values R just used.
  write.csv(df, file.path(DATA_DIR, paste0(name, ".csv")), row.names = FALSE)
  if ("homals" %in% MANIFEST$classic_methods) run_homals(name, df)
  if ("princals" %in% MANIFEST$classic_methods) run_princals(name, df)
}

## ── Morals runs: explicit x/y datasets ───────────────────────────────────
cat("\n== Morals runs ==\n")
read_any <- function(path) {
  full <- file.path(ROOT, path)
  if (grepl("\\.xlsx$", path)) as.data.frame(read_excel(full)) else read.csv(full)
}

for (i in seq_len(nrow(MANIFEST$morals_runs))) {
  run <- MANIFEST$morals_runs[i, ]
  cat(sprintf("-- %s --\n", run$name))
  df <- read_any(run$file)
  x_idx <- unlist(run$x_cols) + 1L   # R is 1-indexed
  y_idx <- run$y_col + 1L
  run_morals(run$name, df[, x_idx, drop = FALSE], df[, y_idx])
}

cat("\nDone.\n")
