#!/usr/bin/env Rscript
# ==============================================================
# get_r_transforms.R
# Runs R Gifi Princals on all processed datasets and saves
# the optimally-scaled transformed DataFrames for analysis.
#
# Usage (called by analyze.py):
#   Rscript analysis/get_r_transforms.R <data_dir> <output_dir>
# ==============================================================

library(Gifi)

args     <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else "validation/datasets/processed"
out_dir  <- if (length(args) >= 2) args[2] else "analysis/r_transforms"

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

all_csv  <- list.files(data_dir, pattern = "\\.csv$", full.names = FALSE)
datasets <- sort(all_csv[!grepl("transformed", all_csv)])

cat("Found", length(datasets), "dataset(s):", paste(datasets, collapse = ", "), "\n")

for (ds_file in datasets) {
    cat("\nProcessing:", ds_file, "\n")

    df <- read.csv(file.path(data_dir, ds_file),
                   stringsAsFactors = TRUE, na.strings = c("", "NA"))

    # Drop index columns
    df <- df[, !grepl("^X$|^Unnamed", colnames(df)), drop = FALSE]
    df[] <- lapply(df, factor)

    cat("  Rows:", nrow(df), "| Columns:", ncol(df), "\n")

    fit <- tryCatch(
        princals(df, ndim = 2),
        error = function(e) {
            cat("  [ERROR]", conditionMessage(e), "\n")
            NULL
        }
    )
    if (is.null(fit)) next

    df_transform <- as.data.frame(as.matrix(fit$transform))
    colnames(df_transform) <- names(df)

    out_path <- file.path(out_dir, paste0("r_transform_", ds_file))
    write.csv(df_transform, out_path, row.names = FALSE)
    cat("  Saved ->", out_path, "\n")
}

cat("\nDone.\n")
