from sklearn.base import BaseEstimator, TransformerMixin


def transformerize_function(class_name, function):
    """ Make a stateless transformer out of a function, i.e. one that doesn't need to be fit
    
    Subtract10Transformer = transformerize_function('Subtract10Transformer', lambda x: x - 10)
    """
    class Transformer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return function(X)
    
    Transformer.__name__ = class_name
    return Transformer


def transformerize_tokenizer(tokenizer_class):
    """ Turn an NLTK tokenizer into a transformer
    
    NISTTokenizerTransformer = transformerize_tokenizer(NISTTokenizer)
    """
    class WrappedTokenizer(BaseEstimator, TransformerMixin):
        def __init__(self, *args, **kwargs):
            self._tokenizer = tokenizer_class(*args, **kwargs)

        def transform(self, X, y=None, **pandas_map_kwargs):
            if isinstance(X, pd.Series):
                return X.map(self._tokenizer.tokenize, **pandas_map_kwargs)
            else:
                X_lst = list(X)
                # hack to guarantee an array of lists
                out = np.array([None for _ in range(len(X_lst))])
                out[:] = self._tokenizer.tokenize_sents(X)
                return out

        def fit_transform(self, X, y=None):
            return self.transform(X)

        def fit(self, X, y=None):
            return self

    WrappedTokenizer.__name__ = tokenizer_class.__name__ + 'Transformer'
    return WrappedTokenizer


def transformerize_estimator(estimator_class, predict_proba=False):
    """ Turn an estimator into a transformer

    LogisticRegressionProbaTransformer = transformerize_estimator(LogisticRegression, predict_proba=True)
    """
    class WrappedEstimator(BaseEstimator, TransformerMixin):
        def __init__(self, *args, **kwargs):
            self._estimator = estimator_class(*args, **kwargs)

        def fit(self, X, y=None, *args, **kwargs):
            return self._estimator.fit(X, y, *args, **kwargs)

    if predict_proba:
        def transform(self, X, y=None, *args, **kwargs):
            return self._estimator.predict_proba(X, y, *args, **kwargs)
    else:
        def transform(self, X, y=None, *args, **kwargs):
            return self._estimator.predict(X, y, *args, **kwargs)

    WrappedEstimator.transform = transform

    WrappedEstimator.__name__ = estimator_class.__name__ + 'Proba' if predict_proba else '' + 'Transformer'
    return WrappedEstimator
