import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class NltkTokenizer(BaseEstimator, TransformerMixin):
    """ Turn an NLTK tokenizer into a transformer
    
    nist_tokenizer_transformer = NltkTokenizerTransformer(NISTTokenizer)
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def transform(self, X, y=None, **pandas_map_kwargs):
        if isinstance(X, pd.Series):
            return X.map(self.tokenizer.tokenize, **pandas_map_kwargs)
        else:
            X_lst = list(X)
            # hack to guarantee an array of lists
            out = np.array([None for _ in range(len(X_lst))])
            out[:] = self.tokenizer.tokenize_sents(X)
            return out

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def fit(self, X, y=None):
        return self
