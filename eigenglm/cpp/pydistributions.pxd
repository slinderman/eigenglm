
# Import C++ classes from distributions.h
cdef extern from "distributions.h":

    cdef cppclass IndependentBernoulli:
        IndependentBernoulli() except +
        double logp(double*)

    cdef cppclass DiagonalGaussian:
        DiagonalGaussian() except +
        double logp(double*)
        void grad(double*, double*)

    cdef cppclass Dirichlet:
        Dirichlet(int) except +
        double logp(double*)
        void grad(double*, double*)
        void as_dirichlet(double*, double*)
        void dw_dg(double*, double*)