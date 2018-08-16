import itertools as it


def identity(x):
    return x

def unzip(iterable):
    return zip(*iterable)


def star(func):
    def _func(*x):
        return func(x)
    return _func

def unstar(func):
    def _func(x):
        return func(*x)
    return _func

def star2(func):
    def _func(x):
        return func(**x)
    return _func

def unstar2(func):
    def _func(**x):
        return func(x)
    return _func


def group_by(key, iterable, include_label=True, collect_group=False):
    gb = it.groupby(iterable, key=key)
    if collect_group and not callable(collect_group):
        collect_group = tuple
    elif not collect_group:
        collect_group = identity

    if include_label:
        for lab, grp in gb:
            yield lab, collect_group(grp)
    else:
        for lab, grp in gb:
            yield collect_group(grp)
