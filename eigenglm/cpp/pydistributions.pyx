# distutils: language = c++
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11

import numpy as np
cimport numpy as np


# Python wrapper of Random
cdef class PyRandom:
    #cdef Random *thisptr

    def __cinit__(self, int seed):
        self.thisptr = new Random(seed)

cdef class PyIndependentBernoulli:
    #cdef IndependentBernoulli *thisptr
    #cdef public int D

    def __cinit__(self, double[::1] rho, PyRandom random):
        self.D = rho.size
        self.thisptr = new IndependentBernoulli(&rho[0], self.D, random.thisptr)

    def __dealloc__(self):
        del self.thisptr

    def logp(self, double[::1] x):
        assert x.size == self.D
        return self.thisptr.logp(&x[0])

cdef class PyDiagonalGaussian:
    #cdef DiagonalGaussian *thisptr
    #cdef public int D

    def __cinit__(self, double[::1] mu, double[::1] sigma, PyRandom random):
        self.D = mu.size
        assert sigma.size == self.D, "Mu and sigma must be the same size!"
        self.thisptr = new DiagonalGaussian(&mu[0], &sigma[0], self.D, random.thisptr)

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
    #cdef public int D
    #cdef Dirichlet *thisptr

    def __cinit__(self, double[::1] alpha, PyRandom random):
        self.D = alpha.size
        self.thisptr = new Dirichlet(&alpha[0], self.D, random.thisptr)

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

    def sample(self):
        cdef double[::1] s = np.zeros(self.D)
        self.thisptr.sample(&s[0])
        return np.asarray(s)