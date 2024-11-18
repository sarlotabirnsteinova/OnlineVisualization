import queue

import numpy as np

from .parallel import (
    FileImageReader, NumberCruncherBase, ParallelProcessor, QueueImageReader)


class DynamicFlatFieldNumberCruncher(NumberCruncherBase):
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


class DynamicFlatFiledProcessorBase(ParallelProcessor):

    def __init__(self, alg, nwrk, buf_size):
        super().__init__(alg, nwrk, buf_size)
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
            return DynamicFlatFieldNumberCruncher(
                self.nwrk, wrk_id, self._shm, self.buf_size,
                self.alg_cls, self.args)
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


class FileDynamicFlatFieldProcessor(DynamicFlatFiledProcessorBase):

    def __init__(self, alg, nwrk, source, image_key, buf_size=4096):
        super().__init__(alg, nwrk, buf_size)
        self.source = source
        self.image_key = image_key

    def reader_constructor(self):
        def constructor():
            return FileImageReader(
                self._shm, self.buf_size, self.alg_cls, self.args,
                self.source, self.image_key)
        return constructor

    def run(self, dc):
        super().run()
        self.rdr.run(dc)
        self.join_workers()


FlatFieldCorrectionFileProcessor = FileDynamicFlatFieldProcessor


class QueueDynamicFlatFieldProcessor(DynamicFlatFiledProcessorBase):

    def __init__(self, alg, nwrk, buf_size=512):
        super().__init__(alg, nwrk, buf_size)
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

    def reader_constructor(self):
        def constructor():
            return QueueImageReader(
                self._shm, self.buf_size, self.alg_cls, self.args,
                self.input_queue, self.output_queue)
        return constructor

    def run(self):
        super().run()
        self.rdr.run()

    def shutdown(self):
        self.rdr.shutdown()
        self.join_workers()

    def __del__(self):
        self.rdr.shutdown()
        self.join_workers()

    def put(self, tid, image_data):
        self.input_queue.put((tid, image_data))

    def get(self):
        return self.output_queue.get()
