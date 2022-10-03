library(glmnet)
library(SLOPE)

dual_norm <- function(x, theta, lambda) {
  x_theta <- sort(abs(t(x) %*% theta), decreasing = TRUE)
  taus <- 1 / cumsum(lambda)

  max(cumsum(x_theta) * taus)
}

lambda_sequence <- function(x, y, reg = 0.1, q = 0.1) {
  lambda <- qnorm(1 - 1:p * q / (2 * p))
  lambda_max <- dual_norm(x, (y - mean(y)) / n, lambda)

  lambda_max * lambda * reg
}

dual_gap <- function(beta, intercept, x, y, lambda) {
  n <- nrow(x)
  p <- ncol(x)

  residual <- y - x %*% beta - intercept
  theta <- residual / n
  theta <- theta / max(1, dual_norm(x, theta, lambda))

  primal <- norm(residual, "2")^2 / (2 * n) +
    sum(lambda * sort(abs(beta), decreasing = TRUE))
  dual <- ((norm(y, "2")^2 - norm(y - theta * n, "2")^2) / (2 * n))
  gap <- primal - dual

  gap
}

uri <- "https://s3.amazonaws.com/pbreheny-data-sets/bcTCGA.rds"

xy <- readRDS(url(uri))

x <- as.matrix(xy$X)

x <- scale(x)
y <- xy$y - mean(xy$y)

n <- nrow(x)
p <- ncol(x)

tol_gap <- 1e-5

tol <- 1e-5

reg <- 0.1

nonzero_target <- 100

glmnet.control(mnlam = 1)

lambda_max <- max(abs(t(x) %*% (y - mean(y)))) / n

while (TRUE) {
  # thresh <- thresh * 0.1
  lambda <- lambda_max * reg

  glmnet_time <- system.time({
    glmnet_fit <- glmnet(
      x,
      y,
      lambda = lambda,
      thresh = tol,
      standardize = FALSE
    )
  })[3]

  lambda <- glmnet_fit$lambda
  beta <- as.matrix(glmnet_fit$beta)
  intercept <- glmnet_fit$a0

  gap <- dual_gap(beta, intercept, x, y, rep(lambda, p))
  n_nonzero <- sum(beta != 0)

  cat("  reg:", reg, "\tgap:", gap, "\tn_nonzero:", n_nonzero, "\n")

  if (n_nonzero < nonzero_target) {
    reg <- reg * 0.99
  } else if (n_nonzero > nonzero_target){
    reg <- reg / 0.9
  }

  if (gap < tol_gap) {
    tol <- tol * 1.1
  } else {
    tol <- tol * 0.1
  }

  if (gap < tol_gap && n_nonzero == 100) {
    break
  }
}

tol <- 1.442099e-05
reg <- 0.1582866

cat("slope\n")

while (TRUE) {
  lambda_seq <- lambda_sequence(x, y, reg)
  slope_time <- system.time({
    slope_fit <- SLOPE(
      x,
      y,
      lambda = lambda_seq,
      alpha = 1,
      scale = "none",
      center = FALSE,
      tol_rel_gap = tol,
      tol_infeas = tol,
      tol_dev_ratio = 0.999
    )
  })[3]

  lambda <- slope_fit$lambda
  alphas <- slope_fit$alpha
  coefs <- coef(slope_fit)
  intercept <- coefs[1]
  beta <- coefs[-1]

  gap <- dual_gap(beta, intercept, x, y, lambda)

  n_nonzero <- sum(unique(abs(beta)) != 0)

  cat("  reg:", reg, "\tgap:", gap, "\tn_nonzero:", n_nonzero, "\n")

  if (n_nonzero < nonzero_target) {
    reg <- reg * 0.99
  } else if (n_nonzero > nonzero_target){
    reg <- reg / 0.9
  }

  if (gap < tol_gap) {
    tol <- tol * 1.1
  } else {
    tol <- tol * 0.1
  }

  if (gap < tol_gap && n_nonzero == 100) {
    break
  }
}

cat("timings\n")
cat("  glmnet: ", glmnet_time, "\n")
cat("  slope: ", slope_time, "\n")
