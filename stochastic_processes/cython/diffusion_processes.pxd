cdef class BrownianDiffusionEWM:
    cdef public float dT, sigma2, alpha
    cpdef update_sigma(self, float dS)
    cpdef float get_sigma(self)
    cpdef float compute_ask_delta(self, float gamma, float A, float k, float simga, float mu, float q)
    cpdef float compute_bid_delta(self, float gamma, float A, float k, float simga, float mu, float q)
    cpdef float compute_spread(self, float gamma, float A, float k, float simga)
    cdef float _compute_bid_delta(self, float gamma, float A, float k, float sigma, float q)
    cdef float _compute_bid_delta_base_qty(self, float gamma, float A, float k, float sigma, float q, float base_qty)
    cdef float _compute_ask_delta(self, float gamma, float A, float k, float sigma, float q)
    cdef float _compute_ask_delta_base_qty(self, float gamma, float A, float k, float sigma, float q, float base_qty)
    cdef float _compute_spread(self, float gamma, float A, float k, float sigma)