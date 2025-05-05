import numpy as np
cimport numpy as np

cdef class FastLSH:
    cdef public int num_perm
    cdef public int bands
    cdef public int rows
    cdef list tables  # <- FIXED!

    def __init__(self, int num_perm=128, int bands=32):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.tables = [dict() for _ in range(bands)]  # fine now

    def insert(self, idx, signature):
        cdef np.ndarray[np.uint64_t, ndim=1] sig = np.array(signature.hashvalues, dtype=np.uint64)
        cdef int band_idx
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band = tuple(sig[start:end])
            if band not in self.tables[band_idx]:
                self.tables[band_idx][band] = []
            self.tables[band_idx][band].append(idx)

    def query(self, signature):
        cdef np.ndarray[np.uint64_t, ndim=1] sig = np.array(signature.hashvalues, dtype=np.uint64)
        cdef int band_idx
        candidates = set()
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band = tuple(sig[start:end])
            if band in self.tables[band_idx]:
                candidates.update(self.tables[band_idx][band])
        return list(candidates)