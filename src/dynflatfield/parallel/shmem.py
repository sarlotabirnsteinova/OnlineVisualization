import mmap

import numpy as np


class SharedMemory:
    def __init__(self):
        self.names = {}
        self.desc = {}
        self.nbytes = 0

    def declare(self, name, shape, dtype):
        nb = np.dtype(dtype).itemsize * np.prod(shape)
        self.desc[name] = (shape, dtype, nb)
        self.nbytes += nb

    def alloc(self):
        self.npages = (
            self.nbytes // mmap.PAGESIZE +
            int(self.nbytes % mmap.PAGESIZE > 0)
        )
        self.buf = mmap.mmap(
            -1, self.npages * mmap.PAGESIZE,
            flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE
        )

        b0 = 0
        for name, (shape, dtype, nb) in self.desc.items():
            bN = b0 + nb
            arr = np.frombuffer(memoryview(self.buf)[b0:bN],
                                dtype=dtype).reshape(shape)
            self.names[name] = arr
            b0 = bN

        self.r = type('Rec', (), self.names)

    def map_arrays(self, instance, names):
        for name in names:
            arr = self.names[name]
            setattr(instance, name, arr)

    def reshape(self, name, shape):
        n = np.prod(shape, dtype=int)
        arr = self.names[name]
        r = np.frombuffer(arr, arr.dtype, n)
        return r.reshape(*shape)

    def put(self, name, data):
        r = self.reshape(name, data.shape)
        r[:] = data
        return r
