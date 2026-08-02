library(Gifi)
DATA_PATH <- "/media/bhaavesh/New Volume/studies/antigravity-projects/pyGifi/PyGifi2/car_pure_categorical_1000.csv"
df <- read.csv(DATA_PATH, stringsAsFactors = TRUE, na.strings = "")
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
