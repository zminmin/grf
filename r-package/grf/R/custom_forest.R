#' Custom forest
#'
#' Trains a custom forest model.
#'
#' @param X The covariates used in the regression.
#' @param Y The outcome.
#' @param sample.fraction Fraction of the data used to build each tree.
#'                        Note: If honesty = TRUE, these subsamples will
#'                        further be cut by a factor of honesty.fraction. Default is 0.5.
#' @param mtry Number of variables tried for each split. Default is
#'             \eqn{\sqrt p + 20} where p is the number of variables.
#' @param num.trees Number of trees grown in the forest. Note: Getting accurate
#'                  confidence intervals generally requires more trees than
#'                  getting accurate predictions. Default is 2000.
#' @param min.node.size A target for the minimum number of observations in each tree leaf. Note that nodes
#'                      with size smaller than min.node.size can occur, as in the original randomForest package.
#'                      Default is 5.
#' @param honesty Whether to use honest splitting (i.e., sub-sample splitting). Default is TRUE.
#'  For a detailed description of honesty, honesty.fraction, honesty.prune.leaves, and recommendations for
#'  parameter tuning, see the grf
#'  \href{https://grf-labs.github.io/grf/REFERENCE.html#honesty-honesty-fraction-honesty-prune-leaves}{algorithm reference}.
#' @param honesty.fraction The fraction of data that will be used for determining splits if honesty = TRUE. Corresponds
#'                         to set J1 in the notation of the paper. Default is 0.5 (i.e. half of the data is used for
#'                         determining splits).
#' @param honesty.prune.leaves If TRUE, prunes the estimation sample tree such that no leaves
#'  are empty. If FALSE, keep the same tree as determined in the splits sample (if an empty leave is encountered, that
#'  tree is skipped and does not contribute to the estimate). Setting this to FALSE may improve performance on
#'  small/marginally powered data, but requires more trees (note: tuning does not adjust the number of trees).
#'  Only applies if honesty is enabled. Default is TRUE.
#' @param alpha A tuning parameter that controls the maximum imbalance of a split. Default is 0.05.
#' @param ci.group.size The forest will grow ci.group.size trees on each subsample.
#'                      In order to provide confidence intervals, ci.group.size must
#'                      be at least 2. Default is 2.
#' @param imbalance.penalty A tuning parameter that controls how harshly imbalanced splits are penalized. Default is 0.
#' @param clusters Vector of integers or factors specifying which cluster each observation corresponds to.
#'  Default is NULL (ignored).
#' @param equalize.cluster.weights If FALSE, each unit is given the same weight (so that bigger
#'  clusters get more weight). If TRUE, each cluster is given equal weight in the forest. In this case,
#'  during training, each tree uses the same number of observations from each drawn cluster: If the
#'  smallest cluster has K units, then when we sample a cluster during training, we only give a random
#'  K elements of the cluster to the tree-growing procedure. When estimating average treatment effects,
#'  each observation is given weight 1/cluster size, so that the total weight of each cluster is the
#'  same.
#' @param compute.oob.predictions Whether OOB predictions on training set should be precomputed. Default is TRUE.
#' @param num.threads Number of threads used in training. By default, the number of threads is set
#'                    to the maximum hardware concurrency
#' @param seed The seed of the C++ random number generator.
#'
#' @return A trained regression forest object.
#'
#' @examples
#' \dontrun{
#' # Train a custom forest.
#' n <- 50
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- X[, 1] * rnorm(n)
#' c.forest <- custom_forest(X, Y)
#'
#' # Predict using the forest.
#' X.test <- matrix(0, 101, p)
#' X.test[, 1] <- seq(-2, 2, length.out = 101)
#' c.pred <- predict(c.forest, X.test)
#' }
#'
#' @export
custom_forest <- function(X, Y, 
                          expe_1, expe_2, expe_3, 
                          fami_1, fami_2, fami_3,
                          sample.fraction = 0.5,
                          ll.split.cutoff = NULL,
                          mtry = min(ceiling(sqrt(ncol(X)) + 20), ncol(X)),
                          num.trees = 2000,
                          sample.weights = NULL,
                          min.node.size = 5,
                          honesty = TRUE,
                          honesty.fraction = 0.5,
                          honesty.prune.leaves = TRUE,
                          alpha = 0.05,
                          ci.group.size = 2,
                          imbalance.penalty = 0.0,
                          clusters = NULL,
                          equalize.cluster.weights = FALSE,
                          compute.oob.predictions = TRUE,
                          num.threads = NULL,
                          seed = runif(1, 0, .Machine$integer.max)) {
  validate_X(X)
  validate_sample_weights(sample.weights, X)
  Y <- validate_observations(Y, X)
  expe_1 <- validate_observations(expe_1, X)
  expe_2 <- validate_observations(expe_2, X)
  expe_3 <- validate_observations(expe_3, X)
  fami_1 <- validate_observations(fami_1, X)
  fami_2 <- validate_observations(fami_2, X)
  fami_3 <- validate_observations(fami_3, X)
  clusters <- validate_clusters(clusters, X)
  samples.per.cluster <- validate_equalize_cluster_weights(equalize.cluster.weights, clusters, NULL)
  num.threads <- validate_num_threads(num.threads)

  # no.split.variables <- numeric(0)

  data <- create_data_matrices(X, outcome = Y, 
                              expe_1 = expe_1, expe_2 = expe_2, expe_3 = expe_3,
                              fami_1 = fami_1, fami_2 = fami_2, fami_3 = fami_3,
                              sample.weights = sample.weights)
  
  D <- cbind(1, expe_1, expe_2, expe_3, fami_1, fami_2, fami_3)
  overall.beta <- solve(t(D) %*% D) %*% t(D) %*% Y


  if (is.null(ll.split.cutoff)) {
    ll.split.cutoff <- 30
  } else if (!is.numeric(ll.split.cutoff) || length(ll.split.cutoff) > 1) {
    stop("LL split cutoff must be NULL or a scalar")
  } else if (ll.split.cutoff < 0) {
    stop("Invalid range for LL split cutoff")
  }

  forest <- custom_train(
    data$train.matrix, data$sparse.train.matrix, data$outcome.index,
    data$expe_1.index, data$expe_2.index, data$expe_3.index,
    data$fami_1.index, data$fami_2.index, data$fami_3.index,
    ll.split.cutoff, overall.beta, mtry, num.trees, min.node.size,
    sample.fraction, honesty, honesty.fraction, honesty.prune.leaves, ci.group.size, alpha,
    imbalance.penalty, clusters, samples.per.cluster, compute.oob.predictions, num.threads, seed
  )

  class(forest) <- c("custom_forest", "grf")
  forest[["X.orig"]] <- X
  forest[["Y.orig"]] <- Y
  forest[["e1.orig"]] <- expe_1
  forest[["e2.orig"]] <- expe_2
  forest[["e3.orig"]] <- expe_3
  forest[["f1.orig"]] <- fami_1
  forest[["f2.orig"]] <- fami_2
  forest[["f3.orig"]] <- fami_3
  forest[["sample.weights"]] <- sample.weights


  forest
}

#' Predict with a custom forest.
#'
#' @param object The trained forest.
#' @param newdata Points at which predictions should be made. If NULL, makes out-of-bag
#'                predictions on the training set instead (i.e., provides predictions at
#'                Xi using only trees that did not use the i-th training example). Note
#'                that this matrix should have the number of columns as the training
#'                matrix, and that the columns must appear in the same order.
#' @param num.threads Number of threads used in training. If set to NULL, the software
#'                    automatically selects an appropriate amount.
#' @param ... Additional arguments (currently ignored).
#'
#' @return Vector of predictions.
#'
#' @examples
#' \dontrun{
#' # Train a custom forest.
#' n <- 50
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' Y <- X[, 1] * rnorm(n)
#' c.forest <- custom_forest(X, Y)
#'
#' # Predict using the forest.
#' X.test <- matrix(0, 101, p)
#' X.test[, 1] <- seq(-2, 2, length.out = 101)
#' c.pred <- predict(c.forest, X.test)
#' }
#'
#' @method predict custom_forest
#' @export
predict.custom_forest <- function(object, newdata = NULL, num.threads = NULL, estimate.variance = TRUE, ...) {
  forest.short <- object[-which(names(object) == "X.orig")]

  X <- object[["X.orig"]]
  train.data <- create_data_matrices(X, outcome = object[["Y.orig"]], 
                                    expe_1 = object[["e1.orig"]],  expe_2 = object[["e2.orig"]], expe_3 = object[["e3.orig"]],
                                    fami_1 = object[["f1.orig"]],  fami_2 = object[["f2.orig"]], fami_3 = object[["f3.orig"]])

  num.threads <- validate_num_threads(num.threads)

  if (!is.null(newdata)) {
    validate_newdata(newdata, X)
    data <- create_data_matrices(newdata)
    ret <- custom_predict(
      forest.short, train.data$train.matrix, train.data$sparse.train.matrix, train.data$outcome.index,
      train.data$expe_1.index, train.data$expe_2.index, train.data$expe_3.index,
      train.data$fami_1.index, train.data$fami_2.index, train.data$fami_3.index,
      data$train.matrix, data$sparse.train.matrix, num.threads, estimate.variance
    )
  } else {
    ret <- custom_predict_oob(
      forest.short, train.data$train.matrix, train.data$sparse.train.matrix, train.data$outcome.index, 
      train.data$expe_1.index, train.data$expe_2.index, train.data$expe_3.index,
      train.data$fami_1.index, train.data$fami_2.index, train.data$fami_3.index,
      num.threads, estimate.variance
    )
  }
  
  # Convert list to data frame.
  empty <- sapply(ret, function(elem) length(elem) == 0)
  do.call(cbind.data.frame, ret[!empty])
}
