import scipy.stats as spt

class log_uniform():        
    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=None, random_state=None):
        uniform = spt.uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.power(self.base, uniform.rvs(random_state=random_state))
        else:
            return np.power(self.base, uniform.rvs(size=size, random_state=random_state))


class log_randint():        
    def __init__(self, a=0, b=1, base=10):
        a = round(a)
        b = round(b)
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=None, random_state=None):
        uniform = spt.uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.round(np.power(self.base, uniform.rvs(random_state=random_state))).astype(int)
        else:
            return np.round(np.power(self.base, uniform.rvs(size=size, random_state=random_state))).astype(int)
