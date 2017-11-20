context("extractSelectedFeatures")

trainLearnExtract = function(learner.name, task, ...) {
  lrn = makeLearner(learner.name, ...)
  mod = train(lrn, task)
  feats = extractSelectedFeatures(mod)
  return(feats)
}

classifTaskWithFeatures = function(learner.name, ...) {
  feats = trainLearnExtract(learner.name, sonar.task, ...)
  all.feats = getTaskFeatureNames(sonar.task)
  expect_character(feats, min.len = 1, max.len = length(all.feats),
    any.missing = FALSE, null.ok = FALSE, info = learner.name)
  expect_subset(feats, all.feats, info = learner.name)
  invisible(NULL)
}

classifTaskWithoutFeatures = function(learner.name, ...) {
  noFeatsTask = makeClassifTask(data = data.frame(x = factor(1:2)), target = "x")
  feats = trainLearnExtract(learner.name, noFeatsTask, ...)
  expect_set_equal(feats, character(0L))
  invisible(NULL)
}

classifSelectNone = function(learner.name, ...) {
  feats = trainLearnExtract(learner.name, sonar.task, ...)
  expect_set_equal(feats, character(0L))
  invisible(NULL)
}

classifErrors = function(learner.name, message, ...) {
  res = tryCatch(
    trainLearnExtract(learner.name, sonar.task, ...),
    error = function(e) e$message
  )
  expect_set_equal(res, message)
  invisible(NULL)
}


classif.learners = c(
  "classif.glmboost",
  "classif.ksvm",
  "classif.LiblineaRL1L2SVC",
  "classif.LiblineaRL1LogReg",
  "classif.LiblineaRL2L1SVC",
  "classif.LiblineaRL2LogReg",
  "classif.LiblineaRL2SVC",
  "classif.LiblineaRMultiClassSVC",
  "classif.rda",
  "classif.rpart",
  "classif.xgboost"
)


test_that("task with features", {
  lapply(classif.learners, classifTaskWithFeatures)
  classifTaskWithFeatures("classif.ranger", importance = "impurity")
  classifTaskWithFeatures("classif.xgboost", booster = "gblinear")
})


test_that("task without features", {
  lapply(classif.learners, classifTaskWithoutFeatures)
  classifTaskWithoutFeatures("classif.ranger", importance = "impurity")
})


# select no features from a task with features
# only learners with embedded feature selection
# for which selecting zero features can happen
test_that("embedded selection selects no features", {
  classifSelectNone("classif.LiblineaRL1L2SVC", cost = 1e-10)
  classifSelectNone("classif.LiblineaRL1LogReg", cost = 1e-10)
  classifSelectNone("classif.ranger", importance = "impurity",
    min.node.size = getTaskSize(sonar.task))
  classifSelectNone("classif.rpart", cp = 1)
})


test_that("filter wrapper", {
  lrn = makeFilterWrapper(makeLearner("classif.ksvm"), fw.method = "variance", fw.perc = 0.5)
  feats = extractSelectedFeatures(train(lrn, sonar.task))
  all.feats = getTaskFeatureNames(sonar.task)
  expect_character(feats, min.len = 1, max.len = length(all.feats),
    any.missing = FALSE, null.ok = FALSE)
  expect_subset(feats, all.feats)
})


test_that("errors for required parameters", {
  classifErrors("classif.ranger",
    message = "For extracting the selected features of classif.ranger set 'importance' to 'impurity' or 'permutation'!")
})
