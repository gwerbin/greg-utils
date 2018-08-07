from gensim.corpora import Dictionary, HashDictionary


class DictionaryTransformer(BaseEstimator, TransformerMixin):
    """ Minimal Scikit-learn wrapper for gensim.corpora.Dictionary

    Does not include .filter_*() methods and other functionality.
    """
    def __init__(self, binary=True, prune_at=2000000, dictionary=None, sparse=True):
        self.dictionary = dictionary if dictionary is not None else None  # don't init the Dictionary right away to accomodate param changes
        self.prune_at = prune_at
        self.sparse = sparse
        self.binary = binary
    
    def fit(self, X, y=None):
        if self.dictionary is None:
            self.dictionary = Dictionary(prune_at=self.prune_at)
        self.dictionary.add_documents(X)
        return self
    
    def doc2dict(self, doc, use_id=True):
        if use_id:
            return dict(self.dictionary.doc2bow(doc))
        else:
            return {self.dictionary.id2token[i]: count for i, count in self.dictionary.doc2bow(doc)}
    
    def transform(self, X, y=None):
        i, j, values = unzip((i, j, count) for i, bow in enumerate(map(self.doc2dict, X)) for j, count in bow.items())
        if self.binary:
            values = [bool(val) for val in values]
        out = sps.coo_matrix((values, (i, j)), (len(X), len(self.dictionary)))
        if self.sparse:
            return out
        return out.todense()


class HashDictionaryTransformer(BaseEstimator, TransformerMixin):
    """ Minimal Scikit-learn wrapper for gensim.corpora.HashDictionary
    """
    def __init__(self, binary=True, id_range=32000, myhash=zlib.adler32, debug=True, dictionary=None, sparse=True):
        self.dictionary = dictionary if dictionary is not None else None  # don't init the Dictionary right away to accomodate param changes
        self.id_range = id_range
        self.binary = binary
        self.sparse = sparse
        self.debug = debug
        self.myhash = myhash

    def fit(self, X, y=None):
        self.dictionary = HashDictionary(id_range=self.id_range, myhash=self.myhash, debug=self.debug)
        if self.debug:
            self.dictionary.add_documents(X)
        return self

    def doc2dict(self, doc, use_id=True):
        if use_id:
            return dict(self.dictionary.doc2bow(doc))
        else:
            if not self.debug:
                raise ValueError('use_id=False is not valid when self.debug=False')
            return {frozenset(self.dictionary.id2token[i]): count for i, count in self.dictionary.doc2bow(doc)}
    
    def transform(self, X, y=None):
        i, j, values = unzip((i, j, count) for i, bow in enumerate(map(self.doc2dict, X)) for j, count in bow.items())
        if self.binary:
            values = [bool(val) for val in values]
        out = sps.coo_matrix((values, (i, j)), (len(X), len(self.dictionary)))
        if self.sparse:
            return out
        return out.todense()
