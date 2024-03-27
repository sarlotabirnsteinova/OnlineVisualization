import numpy as np

from .parallel import ReaderBase, NumberCruncherBase, ParallelProcessor


class FlatFieldCorrectionFileReader(ReaderBase):
    def __init__(self, shm, buf_size, alg_cls, args, source, image_key):
        super().__init__(shm, buf_size, alg_cls, args)

        self._shm.map_arrays(self, ['images', 'images_corr'])
        self.source = source
        self.image_key = image_key
        _, self.ny, self.nx = self.images.shape

    def is_eof(self, nimg):
        return nimg == 0 and self.nimg_wr[0] >= self.nimg_rd[0]

    def read(self):
        count = 0
        while count == 0:
            try:
                tid, data = next(self.data_iter)
            except StopIteration:
                break

            self.data_chunk = data[self.source].get(self.image_key)
            if self.data_chunk is None:
                continue

            self.data_tid = tid
            count = len(self.data_chunk)

        return count

    def push(self, first, count):
        self.tid_queue.append(self.data_tid)

        i0 = first % self.buf_size
        iN = (i0 + count) % self.buf_size
        if iN < i0:
            k = self.buf_size - i0
            self.images[i0:] = self.data_chunk[:k]
            self.images[:iN] = self.data_chunk[k:]
        else:
            self.images[i0:iN] = self.data_chunk

        self.data_chunk = None

    def pop(self, first, count):
        tid = self.tid_queue.pop(0)
        images_corr = np.zeros([count, self.ny, self.nx], float)

        i0 = first % self.buf_size
        iN = (i0 + count) % self.buf_size

        if iN < i0:
            k = self.buf_size - i0
            images_corr[:k] = self.images_corr[i0:]
            images_corr[k:] = self.images_corr[:iN]
        else:
            images_corr[:] = self.images_corr[i0:iN]

        # store
        self.results.append(images_corr)
        self.trains.append(tid)

    def run(self, dc):
        cam_data = dc.select([(self.source, self.image_key)])
        self.data_iter = iter(cam_data.trains())

        self.data_chunk = None
        self.data_tid = 0

        self.tid_queue = []

        self.results = []
        self.trains = []

        super().run()


class FlatFieldCorrectionNumberCruncher(NumberCruncherBase):
    def __init__(self, nwrk, wrk_id, shm, buf_size, alg_cls, args):
        super().__init__(nwrk, wrk_id, shm, buf_size, alg_cls, args)
        shm = self._shm
        shm.map_arrays(self, ['images', 'images_corr'])

        self._dffc = alg_cls(**args)
        self._dffc.set_constants(shm.r.dark, shm.r.flat, shm.r.components)
        self.nc = len(self._dffc.components)

        shape = tuple(shm.r.downsample_shape.tolist())
        self._dffc.set_downsampled_constants(
            tuple(shm.r.downsample_factors.tolist()),
            shm.reshape('dark_ds', shape),
            shm.reshape('flat_ds', shape),
            shm.reshape('components_ds', (self.nc,) + shape),
            shm.r.components_mean
        )

    def process(self, j):
        img = self.images[j]
        self.w = np.zeros(self.nc)
        w, warnflag = self._dffc.refine_weigths(img, self.w)
        self.images_corr[j] = self._dffc.correct_dyn(w, img)
        # self.w = w

    def run(self):
        self.w = np.zeros(self.nc)
        super().run()


class FlatFieldCorrectionFileProcessor(ParallelProcessor):

    def __init__(self, alg, nwrk, source, image_key, buf_size=4096):
        super().__init__(alg, nwrk, buf_size)
        self.source = source
        self.image_key = image_key

        shm = self._shm

        shm.r.dark[:] = alg.dark
        shm.r.flat[:] = alg.flat
        shm.r.components[:] = alg.components

        shm.r.downsample_factors[:] = alg.downsample_factors
        shm.r.downsample_shape[:] = alg.dark_ds.shape

        shm.put('dark_ds', alg.dark_ds)
        shm.put('flat_ds', alg.flat_ds)
        shm.put('components_ds', alg.components_ds)
        shm.r.components_mean[:] = alg.components_mean

        self.args = {}

    def number_cruncher_constructor(self, wrk_id):
        def constructor():
            return FlatFieldCorrectionNumberCruncher(
                self.nwrk, wrk_id, self._shm, self.buf_size,
                self.alg_cls, self.args)
        return constructor

    def reader_constructor(self):
        def constructor():
            return FlatFieldCorrectionFileReader(
                self._shm, self.buf_size, self.alg_cls, self.args,
                self.source, self.image_key)
        return constructor

    def shmem_declare(self):
        shm = self._shm

        nc, ny, nx = self.alg.components.shape
        buf_size = self.buf_size

        # parameters
        shm.declare('dark', (ny, nx), float)
        shm.declare('flat', (ny, nx), float)
        shm.declare('components', (nc, ny, nx), float)
        shm.declare('dark_ds', (ny, nx), np.float32)
        shm.declare('flat_ds', (ny, nx), np.float32)
        shm.declare('components_ds', (nc, ny, nx), np.float32)
        shm.declare('components_mean', (nc,), np.float32)
        shm.declare('downsample_factors', (2,), int)
        shm.declare('downsample_shape', (2,), int)

        # buffer
        shm.declare('images', (buf_size, ny, nx), np.uint16)
        shm.declare('images_corr', (buf_size, ny, nx), float)

    def run(self, dc):
        super().run()
        self.rdr.run(dc)
