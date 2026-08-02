library(Gifi)

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args_full[grep("^--file=", args_full)])
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg[1], winslash = "/")) else getwd()
root_dir <- normalizePath(file.path(script_dir, "..", ".."), winslash = "/")
validation_dir <- file.path(root_dir, "validation")
data_dir <- file.path(validation_dir, "datasets", "processed")
results_dir <- file.path(validation_dir, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

sink(file.path(results_dir, "r_master_report.txt"), split = TRUE)

csv_files <- sort(list.files(data_dir, pattern = "\\.csv$"))
csv_files <- csv_files[!grepl("transformed", csv_files)]

for (ds_file in csv_files) {
    cat("\n============================================================\n")
    cat("PROCESSING DATASET:", ds_file, "\n")
    cat("============================================================\n")

    data_path <- file.path(data_dir, ds_file)
    df <- read.csv(data_path, stringsAsFactors = TRUE, na.strings = c("", "NA"))
    df <- df[, !grepl("^X$|^Unnamed", names(df)), drop = FALSE]

    cat("\nRows:", nrow(df), "\n")
    cat("Columns:", ncol(df), "\n")
    cat("\nFirst 5 rows:\n")
    print(head(df, 5))

    cat("\n============================================\n")
    cat("Running PRINCALS\n")
    cat("============================================\n")

    ndim <- 2
    set.seed(123)
    fit <- princals(df, ndim = ndim)

    cat("\nEigenvalues:\n")
    print(fit$evals)

    cat("\nLoadings:\n")
    print(fit$loadings)

    cat("\n============================================\n")
    cat("Category Quantifications (Dimension 1)\n")
    cat("============================================\n")

    for (vname in names(df)) {
        cat("\n----------------------------------\n")
        cat("Variable:", vname, "\n")
        cat("----------------------------------\n")

        q <- fit$quantifications[[vname]]
        if (is.null(q)) {
            next
        }

        levels_v <- levels(df[[vname]])
        if (is.null(levels_v)) {
            levels_v <- as.character(sort(unique(df[[vname]])))
        }

        vals <- q[, 1]
        for (i in seq_along(vals)) {
            cat(sprintf("%-20s -> %.9f\n", as.character(levels_v[i]), vals[i]))
        }
    }

    cat("\n============================================\n")
    cat("Building Transformed Dataset\n")
    cat("============================================\n")

    df_transformed <- as.data.frame(as.matrix(fit$transform))
    colnames(df_transformed) <- names(df)

    cat("\nFirst 10 rows of transformed dataset:\n")
    print(head(df_transformed, 10))

    out_file <- file.path(data_dir, paste0("gifi_transformed_master_", ds_file))
    write.csv(df_transformed, out_file, row.names = FALSE)
    cat("\nSaved file:", out_file, "\n")
}

sink()
