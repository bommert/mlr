context("stability")

test_that("stability performance", {
  lrn = makeLearner("classif.LiblineaRL1LogReg", cost = 1, epsilon = 1e-4)
  mod = train(lrn, sonar.task)
  pred = predict(mod, sonar.task)
  perf = performance(pred, measures = list(stability), model = mod)
  expect_scalar_na(perf, info = "stability performance")
})


test_that("stability resampling successful", {
  rdescs = list(
    makeResampleDesc(method = "CV", iters = 5L),
    makeResampleDesc(method = "LOO"),
    makeResampleDesc("RepCV", reps = 2L, folds = 3L),
    makeResampleDesc("Bootstrap", iters = 5L),
    makeResampleDesc("Subsample", iters = 5L)
  )

  lrn = makeLearner("classif.LiblineaRL1LogReg", cost = 1, epsilon = 1e-4)
  lapply(rdescs, function(rdesc) {
    r = resample(lrn, sonar.task, rdesc, measures = list(stability), models = TRUE)
    expect_number(r$aggr, lower = -1L, upper = 1L, null.ok = FALSE, na.ok = FALSE,
      info = "stability aggregation")
  })
})


test_that("stability resampling errors", {
  rdescs_error = list(
    makeResampleDesc("Bootstrap", iters = 1L),
    makeResampleDesc("Subsample", iters = 1L),
    makeResampleDesc("Holdout")
  )

  lrn = makeLearner("classif.LiblineaRL1LogReg", cost = 1, epsilon = 1e-4)
  lapply(rdescs_error, function(rdesc) {
    res = tryCatch(
      resample(lrn, sonar.task, rdesc, measures = list(stability), models = TRUE),
      error = function(e) e$message
    )
    expect_set_equal(res, "For stability evaluation, at least two models are needed!")
    invisible(NULL)
  })
})


test_that("resampling models TRUE", {
  lrn = makeLearner("classif.LiblineaRL1LogReg", cost = 1, epsilon = 1e-4)
  rdesc = makeResampleDesc(method = "CV", iters = 5L)
  expect_error(resample(lrn, sonar.task, rdesc, measures = list(stability)))
})


