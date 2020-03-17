
X <- dt[,1:441]
Y <- dt$Y
e1 <- dt$e1
e2 <- dt$e2
e3 <- dt$e3
f1 <- dt$f1
f2 <- dt$f2
f3 <- dt$f3

forest <- custom_forest(X, Y, e1, e2, e3, f1, f2, f3, ll.split.cutoff = 100, num.trees = 200)
predictions <- predict(forest, X)
