import numpy as np
cimport numpy as np

np.import_array()
DTYPE = np.float64

ctypedef np.float_t DTYPE_t

cdef class EWMPoissonMarketArrivalProcess
    cdef float alpha, dT
    cdef np.ndarray lambdas, deltas
    cpdef float[:] sample_lambdas(self, float[:] sample_empirical_deltas)
    