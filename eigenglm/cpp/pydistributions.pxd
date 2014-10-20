
# Import C++ classes from distributions.h
cdef extern from "distributions.h":

    # Wrapper for a random number generator shared by C++ objects
    cdef cppclass Random:
        Random(int) except +

    cdef cppclass IndependentBernoulli:
        IndependentBernoulli(double*, int, Random*) except +
        void get_rho(double*)
        void set_rho(double*)
        double logp(double*)

    cdef cppclass DiagonalGaussian:
        DiagonalGaussian(double*, double*, int, Random*) except +
        void get_mu(double *)
        void set_mu(double *)
        void get_sigma(double *)
        void set_sigma(double *)
        double logp(double*)
        void grad(double*, double*)

    cdef cppclass Dirichlet:
        Dirichlet(double*, int, Random*) except +
        void get_alpha(double *)
        void set_alpha(double *)
        double logp(double*)
        void grad(double*, double*)
        void as_dirichlet(double*, double*)
        void dw_dg(double*, double*)
        void sample(double*)

# Python wrappers
cdef class PyRandom:
    cdef Random *thisptr

cdef class PyIndependentBernoulli:
    cdef IndependentBernoulli *thisptr
    cdef public int D

cdef class PyDiagonalGaussian:
    cdef DiagonalGaussian *thisptr
    cdef public int D

cdef class PyDirichlet:
    cdef public int D
    cdef Dirichlet *thisptr