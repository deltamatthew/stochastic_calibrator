import numpy as np
cimport numpy as np
from cython.parallel import prange

np.import_array()
DTYPE = np.float64

ctypedef np.float_t DTYPE_t

cdef class EWMPoissonMarketArrivalProcess:
    """
        poisson market arrival process arrival rate using exponential weighted moving average
    """

    def __init__(self, float[:] deltas, float[:] lambdas, float alpha = 0.01, float dT = 1.):
        self.deltas = deltas
        if lambdas is None:
            self.lambdas = np.zeros_like(self.deltas) 
        else:
            self.lambdas = lambdas
        self.alpha = alpha
        self.dT = dT

    cpdef set_lambdas(self, float[:] lambdas, float[:] deltas = None):
        if deltas is not None:
            assert len(lambdas) == len(deltas)
            self.deltas = deltas
        else:
            assert len(lambdas) == len(self.deltas)
        self.lambdas = lambdas

    cdef float[:] _sample_lambdas(self, float[:] sample_empirical_deltas):
        cdef int i, j, n
        n = len(self.deltas)
        cdef float[n] sample_lambdas
        for j in prange(n):
            sample_lambdas[j] = 0
        for i in prange(len(sample_empirical_deltas)):
            for j in range(n):
                if sample_empirical_deltas[i] > self.deltas[j]:
                    sample_lambdas[j] += 1.
        return sample_lambdas/self.dT

    cpdef float[:] sample_lambdas(self, float[:] sample_empirical_deltas):
        return self._sample_lambdas(sample_empirical_deltas)

    cpdef float[:] update_lambdas(self, float[:] sample_lambdas):
        self.lambdas = self.alpha * sample_lambdas + (1 - self.alpha) * self.lambdas
        return self.lambdas

    cpdef insert_zero_lambdas(self, int n_empty):
        self.lambdas = (1 - self.alpha)**n_empty * self.lambdas

    cdef float[2] _estimate_params():

    cpdef estimate_params(self):
        return self._estimate_params(self.deltas, self.lambdas)