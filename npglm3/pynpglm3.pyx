# distutils: language = c++
# distutils: libraries = stdc++
# distutils: extra_compile_args = -std=c++11

from libcpp.vector cimport vector
from cython cimport floating

cdef extern from "npglm3.h":
    cdef cppclass NPGlm[Type]:
        NPGlm()
        Type log_likelihood(int T, Type dt, Type* S, Type I_bias, Type* I_stim, Type* I_net)

        void compute_I_net(int T, int D_imp, Type* ir, Type* w_ir, Type* I_net)

# Expose the NPGlm methods
def log_likelihood(int T,
                 floating dt,
                 floating[::1] S,
                 floating I_bias,
                 floating[::1] I_stim,
                 floating[::1] I_net):

    # Create a reference to the dummy class.
    cdef NPGlm[floating] ref
    return ref.log_likelihood(T, dt, &S[0], I_bias, &I_stim[0], &I_net[0]);

def compute_I_net(int T,
                  int D_imp,
                  floating[:,::1] ir,
                  floating[::1] w_ir,
                  floating[::1] I_net):

    # Create a reference to the dummy class.
    cdef NPGlm[floating] ref
    ref.compute_I_net(T, D_imp, &ir[0,0], &w_ir[0], &I_net[0])