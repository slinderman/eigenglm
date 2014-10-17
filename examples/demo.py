import time
import numpy as np
import matplotlib.pyplot as plt

from eigenglm import StandardGLM, StandardGLMParameters
from eigenglm import NormalizedGLM, NormalizedGLMParameters

# Make fake data
def create_test_data(N, T, dt=0.001):
    # Create a fake spike trains
    S = np.random.randint(0,3,(T,N)).astype(np.double)
    return {'N' : N,
            'T' : T,
            'dt' : dt,
            'S' : S}

def run():
    # Specify the number of neurons in the population
    N = 2

    # Make a parameters object that we can modify
    # prms = StandardGLMParameters()
    prms = NormalizedGLMParameters()
    # E.g. change the number of basis elements
    prms.impulse.basis.n_bas = 5

    # Make the GLM object
    # glm = StandardGLM(0, N, prms)
    glm = NormalizedGLM(0, N, prms)

    # Make some fake data
    T = 60000
    M = 1
    for m in range(M):
        glm.add_data(create_test_data(N, T))

    # Run some MCMC
    N_iters = 1000
    print "Running ", N_iters, " iterations of MCMC."
    intvl = 25
    start = time.time()
    for i in range(N_iters):
        glm.resample()
        ll = glm.log_likelihood()

        if i % intvl == 0:
            stop = time.time()
            print "Iteration ", i, ":\tLL: %.3f" % ll, "\tIters/sec: %.3f" % (intvl/(stop-start))
            start = stop

run()