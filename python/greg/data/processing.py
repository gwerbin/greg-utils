import cytoolz as tz


def make_ngrams(s, n, join=None):
    """ Make n-grams
    
    For character ngrams, s should be a string
    For token/word ngrams, s should be a sequence of tokens

    sep='' is recommended for characters, and sep='_' for words.
    """
    ngrams = tz.sliding_window(n, s)
    if join is not None:
        ngrams = (sep.join(grams) for grams in ngrams)
    return ngrams


def jaccard_sim(a, b):
    """ Jaccard similarity score """
    a = set(a)
    b = set(b)
    return len(a&b) / len(a|b)


def ngram_jaccard(s1, s2, n=2):
    """ Jaccard similarity on n-grams """
    return jaccard_sim(set(ngrams(s1, n)), set(ngrams(s2, n)))
