library(Gifi)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript tests/parity/diag_gifi.R <csv-path>")
}

data_path <- normalizePath(args[1], winslash = "/")
df <- read.csv(data_path, stringsAsFactors = TRUE, na.strings = "")
df[] <- lapply(df, factor)

fit <- princals(df, ndim = 2)

cat("\n--- Structure of fit$transform ---\n")
str(fit$transform)

cat("\n--- Length of each element in fit$transform ---\n")
print(sapply(fit$transform, length))

cat("\n--- Dim of each element in fit$transform ---\n")
print(lapply(fit$transform, dim))

df_transformed <- as.data.frame(lapply(fit$transform, as.vector))
cat("\n--- Dim of df_transformed ---\n")
print(dim(df_transformed))

cat("\n--- First 5 rows of df_transformed ---\n")
print(head(df_transformed, 5))
