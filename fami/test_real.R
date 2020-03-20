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
