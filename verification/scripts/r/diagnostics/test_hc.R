library(lmtest)

# Test 1: Linear relationship with small noise (matching Rust test)
set.seed(42)
y <- 1.0 + 2.0*(1:29) + 0.01 * ((1:29) %% 7 - 3)
x <- 1:29

fit <- lm(y ~ x)
harv <- harvtest(fit)
cat("Test 1: Linear with 0.01 noise\n")
cat("  p-value:", harv$p.value, "\n")

# Test 2: Negative values
set.seed(42)
y2 <- c(-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5)
x2 <- -5:5

fit2 <- lm(y2 ~ x2)
harv2 <- harvtest(fit2)
cat("\nTest 2: Negative values\n")
cat("  p-value:", harv2$p.value, "\n")

# Test 3: Multiple predictors (linear relationship)
set.seed(42)
n <- 30
y3 <- 1.0 + 2.0*(1:n) + 0.5*(1:n)/2 + ((1:n)%%3)*1e-10
x3a <- 1:n
x3b <- (1:n)/2

fit3 <- lm(y3 ~ x3a + x3b)
harv3 <- harvtest(fit3)
cat("\nTest 3: Multiple predictors linear\n")
cat("  p-value:", harv3$p.value, "\n")
