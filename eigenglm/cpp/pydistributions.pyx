# distutils: language = c++
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11

import numpy as np
cimport numpy as np

cdef class PyIndependentBernoulli:
    cdef IndependentBernoulli *thisptr
    cdef public int D

    def __cinit__(self):
        self.thisptr = new IndependentBernoulli()
        self.D = 1

    def __dealloc__(self):
        del self.thisptr

    def logp(self, double[::1] x):
        assert x.size == self.D
        return self.thisptr.logp(&x[0])

cdef class PyDiagonalGaussian:
    cdef DiagonalGaussian *thisptr
    cdef public int D

    def __cinit__(self):
        self.thisptr = new DiagonalGaussian()
        self.D = 1

    def __dealloc__(self):
        del self.thisptr

    def logp(self, double[::1] x):
        assert x.size == self.D
        return self.thisptr.logp(&x[0])

    def grad(self, double[::1] x):
        assert x.size == self.D
        cdef double[::1] dx = np.zeros(self.D)
        self.thisptr.grad(&x[0], &dx[0])
        return np.asarray(dx).reshape(self.D)


cdef class PyDirichlet:
    cdef public int D
    cdef Dirichlet *thisptr

    def __cinit__(self, int D):
        self.D = D
        self.thisptr = new Dirichlet(D)

    def __dealloc__(self):
        del self.thisptr

    def logp(self, double[::1] x):
        assert x.size == self.D
        return self.thisptr.logp(&x[0])

    def grad(self, double[::1] x):
        assert x.size == self.D
        cdef double[::1] dx = np.zeros(self.D)
        self.thisptr.grad(&x[0], &dx[0])
        return np.asarray(dx).reshape(self.D)

    def as_dirichlet(self, double[::1] x):
        assert x.size == self.D
        cdef double[::1] dx = np.zeros(self.D)
        self.thisptr.as_dirichlet(&x[0], &dx[0])
        return np.asarray(dx).reshape(self.D)

    def dw_dg(self, double[::1] x):
        assert x.size == self.D
        cdef double[:,::1] dx = np.zeros((self.D, self.D))
        self.thisptr.dw_dg(&x[0], &dx[0,0])
        return np.asarray(dx).reshape((self.D, self.D))