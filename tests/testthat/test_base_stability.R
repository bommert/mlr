context("stability")

dat = data.frame(
  y = factor(c(0, 1, 0, 0, 0, 1, 0, 1, 1, 1)),
  x1 = 10:1,
  x2 = rep(1:5, each = 2),
  x3 = rep(1:5, each = 2),
  x4 = rep(1:5, 2),
  x5 = rep(1:5, each = 2) + rep(1:5, 2)
)
easy.task = makeClassifTask(data = dat, target = "y")

test_that("stability performance", {
  lrn = makeLearner("classif.LiblineaRL1LogReg", cost = 1, epsilon = 1e-4)
  mod = train(lrn, easy.task)
  pred = predict(mod, easy.task)
  perf = performance(pred, measures = list(stability), model = mod)
  expect_scalar_na(perf, info = "stability performance")
})


test_that("stability resampling", {
  rdescs = list(
    makeResampleDesc(method = "CV", iters = 3L),
    makeResampleDesc(method = "LOO"),
    makeResampleDesc("RepCV", reps = 2L, folds = 2L),
    makeResampleDesc("Bootstrap", iters = 3L),
    makeResampleDesc("Subsample", iters = 3L)
  )

  lrn = makeLearner("classif.LiblineaRL1LogReg", cost = 1, epsilon = 1e-4)
  lapply(rdescs, function(rdesc) {
    r = resample(lrn, easy.task, rdesc, measures = list(stability), models = TRUE)
    expect_number(r$aggr, lower = -1L, upper = 1L, null.ok = FALSE, na.ok = FALSE,
      info = "stability aggregation")
  })
})


test_that("stability rdesc errors", {
  rdescs_error = list(
    makeResampleDesc("Bootstrap", iters = 1L),
    makeResampleDesc("Subsample", iters = 1L),
    makeResampleDesc("Holdout")
  )

  lrn = makeLearner("classif.LiblineaRL1LogReg", cost = 1, epsilon = 1e-4)
  lapply(rdescs_error, function(rdesc) {
    expect_error(resample(lrn, easy.task, rdesc, measures = list(stability), models = TRUE),
      "For stability evaluation, at least two models are needed!")
  })
})


test_that("resampling models TRUE", {
  lrn = makeLearner("classif.LiblineaRL1LogReg", cost = 1, epsilon = 1e-4)
  rdesc = makeResampleDesc(method = "CV", iters = 3L)
  expect_error(resample(lrn, easy.task, rdesc, measures = list(stability)),
    "Assertion on 'models' failed: Must be TRUE.")
})



