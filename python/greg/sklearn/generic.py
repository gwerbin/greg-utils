from sklearn.base import BaseEstimator, TransformerMixin


class EstimatorTransformer(BaseEstimator, TransformerMixin):
    """ Make a transformer out of a Scikit-learn "Estimator"
    
    Useful for imputing values from an existing trained model.
    
    linear_regression_transformer = EstimatorTransformer('LinearRegressionTransformer', LinearRegression)
    """
    def __init__(self, estimator, predict_proba=False):
        self.estimator = estimator
        self.predict_proba = predict_proba

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def transform(self, X, y=None):
        if self.predict_proba:
            return self.estimator.predict_proba(X)
        else:
            return self.estimator.predict(X)
