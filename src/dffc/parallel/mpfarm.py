import time
import multiprocessing as mp

from .shmem import SharedMemory


class ReaderBase:
    def __init__(self, shm, buf_size, alg_cls, args):
        self._shm = shm
        self.buf_size = buf_size

        worker_arrays = ['nimg_rd', 'nimg_prc', 'nimg_wr',
                         '_ctrl_neof', '_ctrl_started']
        shm.map_arrays(self, worker_arrays)

    def read(self):
        pass

    def push(self, first, count):
        pass

    def pop(self, first, count):
        pass

    def is_buffer_available(self, nimg):
        nimg_avail = self.buf_size - (self.nimg_rd[0] - self.nimg_wr[0])
        return nimg > 0 and nimg_avail >= nimg

    def is_result_available(self, count):
        nimg_prc = min(self.nimg_prc)
        return (self.nimg_wr[0] + count) <= min(nimg_prc, self.nimg_rd[0])

    def run(self):
        count = list()

        self.nimg_rd[0] = nimg_rd = 0
        self.nimg_wr[0] = nimg_wr = 0
        nimg = 0

        self._ctrl_neof[0] = True
        self._ctrl_started[0] = True
        while True:
            # read data chunk
            if nimg == 0:
                nimg = self.read()

            # push data chunk to queue
            if self.is_buffer_available(nimg):
                count.append(nimg)

                # push data chunk
                self.push(nimg_rd, nimg)

                nimg_rd += nimg
                self.nimg_rd[0] = nimg_rd
                nimg = 0

            # pop processed images from queue
            if len(count) and self.is_result_available(count[0]):
                nimg_corr = count.pop(0)

                # pop result
                self.pop(nimg_wr, nimg_corr)

                nimg_wr += nimg_corr
                self.nimg_wr[0] = nimg_wr

            if nimg == 0 and nimg_wr >= nimg_rd:
                break

            time.sleep(0)

        self._ctrl_neof[0] = False
        self._ctrl_started[0] = False


class NumberCruncherBase:
    def __init__(self, nwrk, wrk_id, shm, buf_size, alg_cls, args):
        self.nwrk = nwrk
        self.wrk_id = wrk_id

        self._shm = shm
        self.buf_size = buf_size

        worker_arrays = ['nimg_rd', 'nimg_prc', 'nimg_wr',
                         '_ctrl_neof', '_ctrl_started']
        shm.map_arrays(self, worker_arrays)

    def process(self, j):
        pass

    def run(self):
        while not self._ctrl_started[0]:
            time.sleep(0)

        i = self.nimg_prc[self.wrk_id] = self.wrk_id
        while self._ctrl_neof[0]:
            if self.nimg_rd[0] > i:
                j = i % self.buf_size
                # process image
                self.process(j)

                # next image
                i += self.nwrk
                self.nimg_prc[self.wrk_id] = i

            time.sleep(0)


class ParallelProcessor:

    NumberCruncher = NumberCruncherBase
    Reader = ReaderBase

    def __init__(self, alg, nwrk, buf_size=4096):
        self.nwrk = nwrk
        self.alg = alg
        self.pool = []
        self.buf_size = buf_size

        self._shm = shm = SharedMemory()

        shm.declare('nimg_rd', (1,), int)
        shm.declare('nimg_prc', (self.nwrk,), int)
        shm.declare('nimg_wr', (1), int)
        shm.declare('_ctrl_neof', (1,), bool)
        shm.declare('_ctrl_started', (1,), bool)

        self.shmem_declare()

        shm.alloc()

        shm.r._ctrl_neof[0] = True
        shm.r._ctrl_started[0] = False

        self.args = {}

    def shmem_declare(self):
        pass

    def start_workers(self):
        if self.pool:
            return

        def number_cruncher(nwrk, wrk_id, shm, buf_size, alg_cls, args):
            nc = self.NumberCruncher(
                nwrk, wrk_id, shm, buf_size, alg_cls, args)
            nc.run()

        alg_cls = type(self.alg)
        self.pool = []
        for wrk_id in range(self.nwrk):
            args = (self.nwrk, wrk_id, self._shm,
                    self.buf_size, alg_cls, self.args)
            nc = mp.Process(target=number_cruncher, args=args)
            nc.start()
            self.pool.append(nc)

        self.rdr = self.Reader(self._shm, self.buf_size, alg_cls, self.args)

    def join_workers(self):
        for w in self.pool:
            w.join()
        self.pool = []

    def run(self):
        pass
