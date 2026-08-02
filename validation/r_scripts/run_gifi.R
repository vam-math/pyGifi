# run_gifi.R — Generate per-parameter result CSVs from R's Gifi package.
#
# Exports Princals category quantifications per variable, and the
# full Gifi Transformed Dataset.

library(Gifi)

DATASETS_DIR <- "datasets/processed/"
RESULTS_DIR  <- "results/"
dir.create(RESULTS_DIR, showWarnings=FALSE, recursive=TRUE)

# Avoid previously transformed datasets from this logic
all_csvs <- list.files(DATASETS_DIR, pattern="\\.csv$")
datasets <- sort(all_csvs[!grepl("transformed", all_csvs)])

safe_name <- function(s) gsub("[/ ]", ".", s)

for (ds in datasets) {
    cat("\n[", gsub("\\.csv", "", ds), "]\n", sep="")
    # Explicitly treat empty strings as NA to match pandas default behavior
    data <- read.csv(paste0(DATASETS_DIR, ds), stringsAsFactors=TRUE, na.strings=c("", "NA"))

    # Drop unnamed index columns
    data <- data[, !grepl("^X$|^Unnamed", colnames(data)), drop=FALSE]

    prefix <- gsub("\\.csv", "", ds)
    varnames <- colnames(data)

    # ─────────────────────────────────────────────────────────────────
    # PRINCALS
    # ─────────────────────────────────────────────────────────────────
    cat("  Fitting R Gifi PRINCALS ...\n")
    # Using explicit seed for exact parity with PyGifi's r_seed=1
    set.seed(1)
    pr <- princals(data, ndim=2, itmax=1000, levels="ordinal")
    base <- paste0(RESULTS_DIR, "r_princals_", prefix)

    # 1. Export Category Quantifications
    for (vname in varnames) {
        q <- pr$quantifications[[vname]]
        if (!is.null(q)) {
            # Gifi drops rownames; explicitly get them from the original factor levels
            cat_levels <- levels(data[[vname]])
            if (is.null(cat_levels)) cat_levels <- as.character(unique(data[[vname]]))
            
            q_df <- as.data.frame(q)
            # Ensure number of categories matches quantifications rows
            if (length(cat_levels) == nrow(q_df)) {
                q_df <- cbind(Category = cat_levels, q_df)
            } else {
                q_df <- cbind(Category = paste0("Cat", 1:nrow(q_df)), q_df)
            }

            write.csv(q_df,
                      paste0(base, "_quant_", safe_name(vname), ".csv"),
                      row.names=FALSE)
        }
    }

    # 2. Build and export the Transformed Dataset
    df_transformed <- as.data.frame(as.matrix(pr$transform))
    colnames(df_transformed) <- varnames

    transformed_path = paste0(DATASETS_DIR, "gifi_transformed_", prefix, ".csv")
    write.csv(df_transformed, transformed_path, row.names=FALSE)
    cat(sprintf("    -> gifi_transformed_%s.csv  [%d, %d] [Transformed Dataset]\n", prefix, nrow(df_transformed), ncol(df_transformed)))

    # 3. Export Eigenvalues
    ev_df <- data.frame(eigenvalue = pr$evals)
    write.csv(ev_df, paste0(base, "_evals.csv"), row.names=FALSE)
}

cat("\nDone — all R Gifi Princals results exported.\n")
