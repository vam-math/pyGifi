# ==============================================================
# R GIFI TRANSFORMATION SCRIPT
# ==============================================================

library(Gifi)

# --------------------------------------------------------------
# SETUP LOGGING
# --------------------------------------------------------------
dir.create("validation/results", showWarnings=FALSE, recursive=TRUE)
sink("validation/results/r_master_report.txt", split=TRUE)

cat("\n============================================\n")
cat("Finding Datasets\n")
cat("============================================\n")

DATA_DIR <- "validation/datasets/processed/"
all_csvs <- list.files(DATA_DIR, pattern="\\.csv$")
datasets <- sort(all_csvs[!grepl("transformed", all_csvs)])

if (length(datasets) == 0) {
    cat("No datasets found in", DATA_DIR, "\n")
    quit(save="no", status=0)
}

cat("Found", length(datasets), "datasets:", paste(datasets, collapse=", "), "\n")

for (ds_file in datasets) {
    cat("\n", paste(rep("=", 60), collapse=""), "\n", sep="")
    cat("PROCESSING DATASET:", ds_file, "\n")
    cat(paste(rep("=", 60), collapse=""), "\n", sep="")

    DATA_PATH <- paste0(DATA_DIR, ds_file)

    df <- read.csv(DATA_PATH, stringsAsFactors = TRUE, na.strings = c("", "NA"))

    df <- df[, !grepl("^X$|^Unnamed", colnames(df)), drop=FALSE]
    df[] <- lapply(df, factor)

    cat("\nRows:", nrow(df), "\n")
    cat("Columns:", ncol(df), "\n")

    cat("\nFirst 5 rows:\n")
    print(head(df, 5))

    # --------------------------------------------------------------
    # INITIALIZATION EXPORT (for Python Parity)
    # --------------------------------------------------------------
    # R's Gifi uses set.seed(123) and rnorm internally. 
    # We export it for Python to use the exact same start point.
    set.seed(123)
    nobs <- nrow(df)
    ndim <- 2
    init_x <- matrix(rnorm(nobs * ndim), nobs, ndim)
    write.csv(init_x, "init_x_r.csv", row.names = FALSE)

    # --------------------------------------------------------------
    # RUN PRINCALS
    # --------------------------------------------------------------

    cat("\n============================================\n")
    cat("Running PRINCALS\n")
    cat("============================================\n")

    fit <- princals(df, ndim = ndim)

    cat("\nEigenvalues:\n")
    print(fit$evals)

    cat("\nLoadings:\n")
    print(fit$loadings)

    # --------------------------------------------------------------
    # CATEGORY QUANTIFICATIONS
    # --------------------------------------------------------------

    cat("\n============================================\n")
    cat("Category Quantifications (Dimension 1)\n")
    cat("============================================\n")

    quant_list <- fit$quantifications

    for (col in names(quant_list)) {
        cat("\n----------------------------------\n")
        cat("Variable:", col, "\n")
        cat("----------------------------------\n")

        q <- quant_list[[col]]
        values <- q[, 1]
        
        # Gifi drops rownames sometimes
        cat_levels <- levels(df[[col]])
        if (is.null(cat_levels)) cat_levels <- as.character(unique(df[[col]]))
        
        for (i in seq_along(values)) {
            level_name <- if (!is.null(rownames(q))) rownames(q)[i] else cat_levels[i]
            cat(sprintf(
                "%-20s -> %.9f\n",
                level_name,
                values[i]
            ))
        }
    }

    # --------------------------------------------------------------
    # BUILD TRANSFORMED DATASET
    # --------------------------------------------------------------

    cat("\n============================================\n")
    cat("Building Transformed Dataset\n")
    cat("============================================\n")

    df_transformed <- as.data.frame(as.matrix(fit$transform))
    colnames(df_transformed) <- names(df)

    cat("\nFirst 10 rows of transformed dataset:\n")
    print(head(df_transformed, 10))

    out_file <- paste0(DATA_DIR, "gifi_transformed_master_", ds_file)
    write.csv(
        df_transformed,
        out_file,
        row.names = FALSE
    )

    cat("\nSaved file:", out_file, "\n")
}
