#' @title Extract the selected features from a model.
#' @description Given a `WrappedModel`, selects the features which
#' are used in this model. This is useful, if feature selection is performed in the model
#' fitting process (e.g. by a filter wrapper or by embedded feature selection).
#' @param model `WrappedModel`\cr
#'   Wrapped model, result of `train`.
#' @return `character(n)`\cr
#'   Names of selected features.
#' @export
#' @examples
#' lrn = makeLearner("classif.rpart")
#' mod = train(lrn, sonar.task)
#' extractSelectedFeatures(mod)
#' @section Note: Currently, this function only works for some classification models.
#' To write a feature extraction method for models fitted using any other learner see
#' `extractSelectedFeaturesInternal`.
extractSelectedFeatures = function(model) {
  extractSelectedFeaturesInternal(model$learner, model)
}
