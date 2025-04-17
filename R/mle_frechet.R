# ------------------------------------------------------------------------------
# MLE for Frechet distribution in R
# ------------------------------------------------------------------------------

rm(list = ls())
set.seed(123)

# CDF: F(x) = exp(-T*x^{-theta})
# PDF: f(x) = T*theta*x^(-theta-1)*exp(-T*x^{-theta})

# T > 0 is the technology/location parameter
# theta is the shape/dispersion parameter

# ------------------------------------------------------------------------------

# ---- 1. Simulate Frechet data ----
n <- 500
T_true <- 2
theta_true <- 5

# Inverse CDF: x = (-log(U)/T)^(-1/theta)
u <- runif(n)
x <- (-log(u)/T_true )^(-1/theta_true)

# Note: If U ~ Uniform(0,1), then X = (-log(U)/T)^(-1/theta) ~ Frechet(T, theta)
# This uses the inverse of the Frechet CDF: F(x) = exp(-T x^{-theta})

# ---- 2. Plot the true Frechet PDF against histogram of simulated data ----

# Create grid of x-values for the theoretical PDF
x_grid <- seq(min(x), max(x), length.out = 500)

# True Frechet PDF: f(x) = T * theta * x^{-theta - 1} * exp(-T * x^{-theta})
frechet_pdf <- function(x, T, theta) {
  T * theta * x^(-theta - 1) * exp(-T * x^(-theta))
}

y_pdf <- frechet_pdf(x_grid, T_true, theta_true)

# Plot histogram with density scaling
hist(
  x,
  breaks = 100,
  col = "lightblue",
  border = "white",
  probability = TRUE,  # scales histogram to a density
  main = "Frechet Simulation: Data vs. True PDF",
  xlab = "x"
)

# Add PDF curve
lines(x_grid, y_pdf, col = "red", lwd = 2)
legend("topright", legend = c("True PDF"), col = "red", lwd = 2)

# ---- 3. Log-likelihood function for Frechet(T, theta) ----

# Log-likelihood for Frechet(T, theta) distribution:
# f(x; T, theta) = T * theta * x^{-theta - 1} * exp(-T * x^{-theta})
# log f(x) = log(T) + log(theta) - (theta + 1) * log(x) - T * x^{-theta}

# The total log-likelihood is the sum over observations:
# L(T, theta) = sum_{i=1}^n [log(T) + log(theta) - (theta + 1) * log(x_i) - T * x_i^{-theta}]
# This function returns the NEGATIVE log-likelihood for use with optim()

frechet_llk <- function(par, data) {
  T     <- par[1]
  theta <- par[2]
  
  # Penalize invalid parameter values
  if (T <= 0 || theta <= 0) return(1e10)
  
  # Frechet log-likelihood: log f(x) = log(T) + log(theta) - (theta + 1) * log(x) - T * x^{-theta}
  ll <- log(T) + log(theta) - (theta + 1) * log(data) - T * data^(-theta)
  
  # Return negative log-likelihood for minimization
  return(-sum(ll))
}

# ---- 4. Estimate parameters via MLE ----

# Starting values (should be positive)
start_vals <- c(T = 1, theta = 1)

# Call optim to minimize negative log-likelihood
mle_result <- optim(
  par = start_vals,
  fn = frechet_llk,
  data = x,
  method = "L-BFGS-B",               # bounded optimizer
  lower = c(1e-6, 1e-6),             # ensure T > 0, theta > 0,
  hessian = TRUE                     # numerical approximation
)

# ---- 5. Point Estimates ----
cat("True parameters:\n")
cat("T:", T_true, " | Theta:", theta_true, "\n\n")

cat("MLE estimates:\n")
cat("T:", round(mle_result$par[1], 4), "\n")
cat("Theta:", round(mle_result$par[2], 4), "\n")

# ---- 6. Standard Errors ----

# Invert the Hessian to get covariance matrix
vcov_matrix <- solve(mle_result$hessian)

# Standard errors are sqrt of diagonal
se <- sqrt(diag(vcov_matrix))

cat("\nStandard Errors:\n")
cat("SE(T):", round(se[1], 4), "\n")
cat("SE(Theta):", round(se[2], 4), "\n")

# ------------------------------------------------------------------------------

