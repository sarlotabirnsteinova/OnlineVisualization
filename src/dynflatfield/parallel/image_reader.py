import queue

import numpy as np

from .mpfarm import ReaderBase


class ImageReaderBase(ReaderBase):
    def __init__(self, shm, buf_size, alg_cls, args):
        super().__init__(shm, buf_size, alg_cls, args)

        self._shm.map_arrays(self, ['images', 'images_corr'])
        _, self.ny, self.nx = self.images.shape

    def push(self, first, count, data):
        raw_images, meta = data
        self.meta_queue.append(meta)

        i0 = first % self.buf_size
        iN = (i0 + count) % self.buf_size
        if iN < i0:
            k = self.buf_size - i0
            self.images[i0:] = raw_images[:k]
            self.images[:iN] = raw_images[k:]
        else:
            self.images[i0:iN] = raw_images

    def pop(self, first, count):
        meta = self.meta_queue.pop(0)
        images_corr = np.zeros(
            [count, self.ny, self.nx], self.images_corr.dtype)

        i0 = first % self.buf_size
        iN = (i0 + count) % self.buf_size

        if iN < i0:
            k = self.buf_size - i0
            images_corr[:k] = self.images_corr[i0:]
            images_corr[k:] = self.images_corr[:iN]
        else:
            images_corr[:] = self.images_corr[i0:iN]

        self.write(images_corr, meta)

    def write(self, data, meta):
        pass

    def run(self):
        self.meta_queue = []
        super().run()


class QueueImageReader(ImageReaderBase):
    def __init__(self, shm, buf_size, alg_cls, args,
                 input_queue, output_queue):
        super().__init__(shm, buf_size, alg_cls, args)

        self.input_queue = input_queue
        self.output_queue = output_queue
        self._eof = True

    def is_eof(self, nimg):
        return self._eof

    def read(self):
        try:
            tid, data = self.input_queue.get_nowait()
        except queue.Empty:
            return 0, None

        if data is None:
            return 0, None

        meta = {"tid": tid}
        return len(data), (data, meta)

    def write(self, data, meta):
        tid = meta["tid"]
        self.output_queue.put((tid, data))

    def run(self):
        self._eof = False
        super().run()

    def shutdown(self):
        self._eof = True


class FileImageReader(ImageReaderBase):
    def __init__(self, shm, buf_size, alg_cls, args, source, image_key):
        super().__init__(shm, buf_size, alg_cls, args)

        self.source = source
        self.image_key = image_key

    def is_eof(self, nimg):
        return nimg == 0 and self.nimg_wr[0] >= self.nimg_rd[0]

    def read(self):
        count = 0
        while count == 0:
            try:
                tid, data_dict = next(self.data_iter)
            except StopIteration:
                return 0, None

            data = data_dict[self.source].get(self.image_key)
            if data is None:
                continue

            count = len(data)

        meta = {"tid": tid}
        return count, (data, meta)

    def write(self, data, meta):
        self.results.append(data)
        self.trains.append(meta["tid"])

    def run(self, dc):
        cam_data = dc.select([(self.source, self.image_key)])
        self.data_iter = iter(cam_data.trains())

        self.results = []
        self.trains = []

        super().run()
