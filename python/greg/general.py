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
