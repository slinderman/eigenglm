import time
import numpy as np

# Set the random seed for reproducability
# np.random.seed(0)

import matplotlib.pyplot as plt

from eigenglm import StandardGLMPopulation, StandardGLM, StandardGLMParameters, StandardGLMPopulationParameters
from eigenglm import NormalizedGLMPopulation, NormalizedGLMParameters, NormalizedGLMPopulationParameters

def collect_sample(population):
    W = population.W
    A = population.A
    bias = population.bias
    ir = population.impulse_response()

    return bias, W, A, ir

def run():
    # Specify the number of neurons in the population
    N = 2

    # Make a parameters object that we can modify
    # prms = StandardGLMParameters()
    # prms = NormalizedGLMParameters()
    # prms = StandardGLMPopulationParameters(N)
    prms = NormalizedGLMPopulationParameters(N)
    # E.g. change the number of basis elements
    prms.glms[0].impulse.basis.n_bas = 5
    prms.glms[1].impulse.basis.n_bas = 5

    # Make the GLM object
    # glm = StandardGLM(0, N, prms)
    # glm = NormalizedGLM(0, N, prms)
    # population = StandardGLMPopulation(N, prms)
    population = NormalizedGLMPopulation(N, prms)

    # Simulate some data
    T = 15
    dt = 0.001
    data = population.simulate(T, dt)
    population.add_data(data)

    # Run some MCMC
    N_iters = 1000
    print "Running ", N_iters, " iterations of MCMC."
    samples = []
    print_interval = 5
    start = time.time()
    for i in range(N_iters):
        population.resample()
        # ll = population.log_likelihood()
        lp = population.log_probability()

        if i % print_interval == 0:
            stop = time.time()
            print "Iteration ", i, ":\tLP: %.3f" % lp, "\tIters/sec: %.3f" % (print_interval/(stop-start))
            start = stop

        samples.append(collect_sample(population))

    # Now analyze the samples
    print "True bias:\t", population.bias
    print "True A*W:\t", population.A * population.W
    # print "True W:\t", population.W

    bias_samples = np.array([s[0] for s in samples])
    print "Inferred bias:\t", bias_samples.mean(axis=0), "+-", bias_samples.std(axis=0)

    W_samples = np.array([s[1] for s in samples])
    A_samples = np.array([s[2] for s in samples])
    AW_samples = A_samples * W_samples
    # print "Inferred W:\n", W_samples.mean(axis=0), "\n+-\n", W_samples.std(axis=0), "\n"
    # print "Inferred A:\n", A_samples.mean(axis=0), "\n+-\n", A_samples.std(axis=0), "\n"
    print "Inferred A*W:\n", AW_samples.mean(axis=0), "\n+-\n", AW_samples.std(axis=0), "\n"

run()