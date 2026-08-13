# tests/fixtures/generate_fixtures.R
# R script to generate ground-truth output from cran/Gifi

library(Gifi)
library(jsonlite)

# Helper to extract the exact starting random matrix gifiEngine uses
get_init_x <- function(nobs, ndim) {
  set.seed(123)
  matrix(rnorm(nobs * ndim), nobs, ndim)
}

# 1. Homals Default (Hartigan)
data(hartigan)
h_fit <- homals(hartigan, ndim = 2, eps = 1e-8)

write_json(list(
  objectscores = h_fit$objectscores,
  evals = h_fit$evals,
  quantifications = h_fit$quantifications,
  scoremat = h_fit$scoremat,
  dmeasures = h_fit$dmeasures,
  f = h_fit$f,
  ntel = h_fit$ntel,
  init_x = get_init_x(nrow(hartigan), 2)
), "tests/fixtures/homals_hartigan.json", digits = 10, null="null", auto_unbox=TRUE)

# 2. Princals Default (ABC)
data(ABC)
write.csv(ABC, "tests/fixtures/ABC.csv", row.names=FALSE)
# Convert ABC numeric to factors for princals ordinal
abc_ord <- as.data.frame(lapply(ABC, as.factor))
p_fit <- princals(abc_ord, ndim = 2, eps = 1e-8)

write_json(list(
  objectscores = p_fit$objectscores,
  evals = p_fit$evals,
  quantifications = p_fit$quantifications,
  loadings = p_fit$loadings,
  scoremat = p_fit$scoremat,
  dmeasures = p_fit$dmeasures,
  lambda = p_fit$lambda,
  f = p_fit$f,
  init_x = get_init_x(nrow(abc_ord), 2)
), "tests/fixtures/princals_abc.json", digits = 10, null="null", auto_unbox=TRUE)

p_copies_fit <- princals(abc_ord[,1:3], ndim = 2, copies=c(1,2,1), eps=1e-8)
write_json(list(
  objectscores = p_copies_fit$objectscores,
  evals = p_copies_fit$evals,
  quantifications = p_copies_fit$quantifications,
  loadings = p_copies_fit$loadings,
  f = p_copies_fit$f,
  init_x = get_init_x(nrow(abc_ord), 2)
), "tests/fixtures/princals_copies.json", digits = 10, null="null", auto_unbox=TRUE)

p_passive_fit <- princals(abc_ord[,1:3], ndim = 2, active=c(TRUE, TRUE, FALSE), eps=1e-8)
write_json(list(
  objectscores = p_passive_fit$objectscores,
  evals = p_passive_fit$evals,
  quantifications = p_passive_fit$quantifications,
  loadings = p_passive_fit$loadings,
  f = p_passive_fit$f,
  init_x = get_init_x(nrow(abc_ord), 2)
), "tests/fixtures/princals_passive.json", digits = 10, null="null", auto_unbox=TRUE)

# 5. Morals Default (Neumann)
m_fit <- morals(neumann[,1:2], neumann[,3], eps=1e-8)

write_json(list(
  evals = m_fit$evals,
  smc = m_fit$smc,
  yhat = m_fit$yhat,
  ypred = m_fit$ypred,
  xhat = m_fit$xhat,
  beta = m_fit$beta,
  init_x = get_init_x(nrow(neumann), 1)
), "tests/fixtures/morals_neumann.json", digits = 10, null="null", auto_unbox=TRUE)

# 6. Morals Polynomial Spline
m_spline_fit <- morals(neumann[,1:2], neumann[,3], xknots=knotsGifi(neumann[,1:2], "E"), xdegrees=2, eps=1e-8)
write_json(list(
  smc = m_spline_fit$smc,
  beta = m_spline_fit$beta,
  init_x = get_init_x(nrow(neumann), 1)
), "tests/fixtures/morals_spline.json", digits = 10, null="null", auto_unbox=TRUE)

# 7. Morals Monotone
m_mono_fit <- morals(neumann[,1:2], neumann[,3], ydegrees=1, yordinal=TRUE, eps=1e-8)
write_json(list(
  yhat = m_mono_fit$yhat,
  smc = m_mono_fit$smc,
  beta = m_mono_fit$beta,
  init_x = get_init_x(nrow(neumann), 1)
), "tests/fixtures/morals_monotone.json", digits = 10, null="null", auto_unbox=TRUE)

# 8/9. Ties and Missing Modes (Galo)
data(galo)
write.csv(galo, "tests/fixtures/galo.csv", row.names=FALSE)

# Ties modes
for (t in c("s", "p", "t")) {
  # We use princals to test ties and missing on ordinal data
  fit_t <- princals(galo[,1:3], ndim=2, ties=t, eps=1e-8)
  write_json(list(
    evals = fit_t$evals,
    quantifications = fit_t$quantifications,
    f = fit_t$f,
    init_x = get_init_x(nrow(galo), 2)
  ), paste0("tests/fixtures/ties_", t, ".json"), digits = 10, null="null", auto_unbox=TRUE)
}

# Missing modes
# Introduce NAs deterministically
galo_na <- galo[,1:3]
galo_na[1:10, 1] <- NA
galo_na[11:20, 2] <- NA
write.csv(galo_na, "tests/fixtures/galo_na.csv", row.names=FALSE)

for (m in c("m", "s", "a")) {
  fit_m <- princals(galo_na, ndim=2, missing=m, eps=1e-8)
  write_json(list(
    evals = fit_m$evals,
    quantifications = fit_m$quantifications,
    f = fit_m$f,
    init_x = get_init_x(nrow(galo_na), 2)
  ), paste0("tests/fixtures/missing_", m, ".json"), digits = 10, null="null", auto_unbox=TRUE)
}

cat("Successfully generated R Gifi fixtures and datasets in tests/fixtures/\n")
