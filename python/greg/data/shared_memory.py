import ctypes
import numpy as np
from bidict import bidict


TYPE_MAPPING = bidict({
    np.dtype('float64'): ctypes.c_double,
    np.dtype('float32'): ctypes.c_float,
    np.dtype('int64'): ctypes.c_int64,
    np.dtype('int32'): ctypes.c_int32,
    np.dtype('bool'): ctypes.c_bool
})


class SharedNdarray():
    def __init__(self, data, ctype, shape):
        self.data = data
        self.ctype = ctype
        self.shape = shape

    @classmethod
    def from_numpy(cls, x):
        shape = x.shape
        k = x.size

        dtype = x.dtype
        ctype = TYPE_MAPPING[dtype]

        array_raw = multiprocessing.RawArray(ctype, k)
        array_np = np.frombuffer(array_raw, dtype=dtype).reshape(shape)
        np.copyto(array_np, x)

        return cls(array_raw, ctype, shape)

    def to_numpy(self, x):
        ctype = self.ctype
        dtype = TYPE_MAPPING.inv[ctype]
        return np.frombuffer(self.data, dtype=dtype).reshape(self.shape)


class SharedSparseArray():
    def __init__(self, data, row, col, shape):
        self.data = data
        self.row = row
        self.col = col
        self.shape = shape

    @classmethod
    def from_scipy(cls, x):
        x = sps.coo_matrix(x)
        return cls(x.data, x.row, x.col, x.shape)

    def to_scipy(self, x):
        return sps.coo_matrix(x.data, (x.row, x.col), shape=x.shape)
