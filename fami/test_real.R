library(grf)

centering <- function(X, Y, e1, e2, e3, f1, f2, f3){

  forest.Y <- regression_forest(X, Y)
  Y.hat = predict(forest.Y)$predictions

  forest.e1 <- regression_forest(X, e1)
  e1.hat = predict(forest.e1)$predictions

  forest.e2 <- regression_forest(X, e2)
  e2.hat = predict(forest.e2)$predictions

  forest.e3 <- regression_forest(X, e3)
  e3.hat = predict(forest.e3)$predictions

  forest.f1 <- regression_forest(X, f1)
  f1.hat = predict(forest.f1)$predictions

  forest.f2 <- regression_forest(X, f2)
  f2.hat = predict(forest.f2)$predictions

  forest.f3 <- regression_forest(X, f3)
  f3.hat = predict(forest.f3)$predictions


  data <- cbind(X, Y = Y - Y.hat,
                e1 = e1 - e1.hat, e2 = e2 - e2.hat, e3 = e3 - e3.hat,
                f1 = f1 - f1.hat, f2 = f2 - f2.hat, f3 = f3 - f3.hat)

  return(data)
}

n <- 3000
x_number <- 6
dt <- rbinom(n * x_number, 1, 0.5)
dim(dt) <- c(n, x_number)
dt <- data.frame(dt)
names(dt) <- c(paste0('X', 1:x_number))

dt$e1 <- runif(n, min = 1, max = 1000)
dt$e2 <- runif(n, min = 1, max = 1000)
dt$e3 <- runif(n, min = 1, max = 1000)
dt$f1 <- runif(n, min = 1, max = 1000)
dt$f2 <- runif(n, min = 1, max = 1000)
dt$f3 <- runif(n, min = 1, max = 1000)

dt$Y <- dt$X1 + dt$X2 + dt$X3 + dt$X4 + dt$X5 + dt$X6 +
  dt$e1 + dt$e2 + dt$e3 +
  dt$f1 +
  (dt$X1) * dt$f2 +
  (dt$X2 + dt$X3) *dt$f3 +
  rnorm(n, mean = 0, sd = .1)

dt <- centering(dt[,1:x_number], dt$Y, dt$e1, dt$e2, dt$e3, dt$f1, dt$f2, dt$f3)



#X <- dt[,1:441]
X <- dt[,1:6]
Y <- dt$Y
e1 <- dt$e1
e2 <- dt$e2
e3 <- dt$e3
f1 <- dt$f1
f2 <- dt$f2
f3 <- dt$f3

#W <- rbinom(n, 1, 0.5)
#IV <- rbinom(n, 1, 0.5)

#D <- cbind(1, e1, e2, e3, f1, f2, f3)
#overall_beta <- solve(t(D) %*% D) %*% t(D) %*% Y


#forest <- instrumental_forest(X, Y, W, IV, num.trees = 200)
forest <- custom_forest(X, Y, e1, e2, e3, f1, f2, f3, ll.split.cutoff = 100, num.trees = 200)
predictions <- predict(forest, X)
