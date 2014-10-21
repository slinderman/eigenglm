"""
GLM classes that use an underlying C++ implementation for inference and
likelihood calculation.
"""
import numpy as np
from scipy.misc import logsumexp

from hips.plotting.layout import *
from hips.inference.log_sum_exp import log_sum_exp_sample
from hips.inference.ars2 import AdaptiveRejectionSampler

from eigenglm.parameters import *
from eigenglm.utils.basis import Basis
import eigenglm.cpp.pyeigenglm as peg

import eigenglm.cpp.pydistributions as pd

class _GLM(object):
    """
    Base class to be subclassed with specific implementations.
    """
    spiketrains = []
    glm = None

    def __init__(self, n, N, population=None):
        """
        Default constructor
        :param n: Index of this neuron in [0, N)
        :param N: Number of neurons in the population
        :return:
        """
        self.n = n
        self.N = N
        self.population = population

    def check_data(self, data):
        """
        Check that the data dictionary is valid.

        :param data:
        :return:
        """
        # Check the existence of the spike train.
        assert 'S' in data, "Data must contain a TxN array of spike counts!"
        T,N = data['S'].shape
        assert N == self.N, "Number of neurons in data (%d) does not " \
                            "match the expected number (%d)!" % (N, self.N)

        # Make sure the sampling frequency is specified.
        assert 'dt' in data and np.isscalar(data['dt']), "Data must contain a scalar " \
                                                         "dt specifying the bin size!"

        # Make sure that T and N are available in data
        data['T'] = T
        data['N'] = N

    def add_data(self, data):
        raise NotImplementedError()

    def log_prior(self):
        raise NotImplementedError()

    def log_likelihood(self, data=None):
        raise NotImplementedError()

    def log_probability(self, data=None):
        raise NotImplementedError()

    def resample(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def plot_firing_rate(self, data=None, color=None, T_lim=None, plot_currents=True, ax=None):
        # TODO: Plot the given spike train

        # HACK: For now just plot the first spike train
        st = self.spiketrains[0]
        tt = st['dt'] * np.arange(0, st['T'])
        fr = self.glm.get_firing_rate(st)

        # Make a figure
        ax_given = ax is not None
        if not ax_given:
            fig = create_figure((4,3))
            ax = fig.add_subplot(1,1,1)
        ax.plot(tt, fr, color=color)

        if not ax_given:
            return ax


##
#  Standard GLM implementation
#
class StandardGLM(_GLM):

    def __init__(self, n, N, params):
        super(StandardGLM, self).__init__(n, N)
        assert isinstance(params, StandardGLMParameters)
        self.params = params

        # Create the random number generator
        self.random = pd.PyRandom(np.random.randint(np.iinfo(np.int32).max))

        # Create the bias
        self.bias_prior = pd.PyDiagonalGaussian(np.array([0.]), np.array([1.]), self.random)
        self.bias_component = peg.PyBiasCurrent(self.random, self.bias_prior)

        # Create the impulse response
        D_imp = self.params.impulse.basis.D
        self.impulse_basis = Basis(params.impulse.basis)
        self.impulse_prior = pd.PyDiagonalGaussian(np.zeros(D_imp), np.ones(D_imp), self.random)
        self.impulse_component = peg.PyLinearImpulseCurrent(N, D_imp, self.random, self.impulse_prior)

        # Create the nonlinearity
        self.nlin = peg.PySmoothRectLinearLink()

        # Create the constant network column
        self.network_component = peg.PyConstantNetworkColumn(N)

        # Create the GLM
        self.glm = peg.PyStandardGlm(n, N, self.random,
                                     self.bias_component,
                                     self.impulse_component,
                                     self.nlin,
                                     self.network_component)

    def check_data(self, data):
        super(StandardGLM, self).check_data(data)

        # TODO: Also check for a stimulus

    def add_data(self, data):
        """
        Add a data sequence.
        :param data:
        :return:
        """
        self.check_data(data)
        N = self.N
        T = data['T']
        dt = data['dt']
        S = data['S']
        D_imp = self.params.impulse.basis.D
        dt_max = self.params.impulse.dt_max

        # Filter the spike train
        filtered_S = self.impulse_basis.convolve_with_basis(S, dt, dt_max)

        Sn = S[:, self.n].copy(order='C')

        # Create a spike train object and add it to the GLM
        st = peg.PySpikeTrain(N, T, dt, Sn, D_imp, filtered_S)
        self.glm.add_spike_train(st)
        self.spiketrains.append(st)

    def log_prior(self):
        return self.glm.log_prior()

    def log_likelihood(self, data=None):
        return self.glm.log_likelihood()

    def log_probability(self, data=None):
        return self.glm.log_probability()

    def resample(self):
        self.glm.resample()

    # Properties for the GLM state
    def impulse_response(self, dt):
        dt_max = self.params.impulse.dt_max
        # Get the L x B basis
        basis = self.impulse_basis.interpolate_basis(dt, dt_max)
        # The N x B weights
        weights = self.impulse_component.get_w_ir()
        return np.dot(weights, basis.T)

    @property
    def bias(self):
        return self.bias_component.get_bias()

    @property
    def w_ir(self):
        return self.impulse_component.get_w_ir()

    @property
    def An(self):
        # Return the n-th column of A
        return self.network_component.get_A()

    @property
    def Wn(self):
        # Return the n-th column of W
        return self.network_component.get_W()



class NormalizedGLM(_GLM):
    def __init__(self, n, N, params, population=None):
        super(NormalizedGLM, self).__init__(n, N, population=population)
        assert isinstance(params, NormalizedGLMParameters)
        self.params = params

        # Create the random number generator
        self.random = pd.PyRandom(np.random.randint(np.iinfo(np.int32).max))

        # Create the bias
        self.bias_prior = pd.PyDiagonalGaussian(np.array([self.population.bias_hyperprior.mu]),
                                                np.sqrt(np.array([self.population.bias_hyperprior.sigmasq])),
                                                self.random)
        self.bias_component = peg.PyBiasCurrent(self.random, self.bias_prior)

        # Create the impulse response
        D_imp = self.params.impulse.basis.D
        self.impulse_basis = Basis(params.impulse.basis)
        self.impulse_prior = pd.PyDirichlet(np.ones(D_imp), self.random)
        self.impulse_component = peg.PyDirichletImpulseCurrent(N, D_imp, self.random, self.impulse_prior)

        # Create the nonlinearity
        self.nlin = peg.PySmoothRectLinearLink()

        # Create the constant network column
        rho = 0.5 * np.ones(N)
        rho[n] = 0.9
        self.A_prior = pd.PyIndependentBernoulli(rho, self.random)

        mu = np.zeros(N)
        mu[n] = -1.0
        sigma = np.ones(N)
        sigma[n] = 0.5
        self.W_prior = pd.PyDiagonalGaussian(mu, sigma, self.random)
        self.network_component = peg.PyGaussianNetworkColumn(N, self.random, self.W_prior, self.A_prior)

        # Create the GLM
        self.glm = peg.PyNormalizedGlm(n, N, self.random,
                                       self.bias_component,
                                       self.impulse_component,
                                       self.nlin,
                                       self.network_component)

        # Create the C++ wrapper object
        # self.glm = peg.PyNormalizedGlm(n, N, self.params.impulse.basis.D)

    def check_data(self, data):
        super(NormalizedGLM, self).check_data(data)

        # TODO: Also check for a stimulus

    def add_data(self, data):
        """
        Add a data sequence.
        :param data:
        :return:
        """
        self.check_data(data)
        N = self.N
        T = data['T']
        dt = data['dt']
        S = data['S']
        D_imp = self.params.impulse.basis.D
        dt_max = self.params.impulse.dt_max

        # Filter the spike train
        filtered_S = self.impulse_basis.convolve_with_basis(S, dt, dt_max)

        Sn = S[:, self.n].copy(order='C')

        # Create a spike train object and add it to the GLM
        st = peg.PySpikeTrain(N, T, dt, Sn, D_imp, filtered_S)
        self.glm.add_spike_train(st)
        self.spiketrains.append(st)

    def log_prior(self):
        return self.glm.log_prior()

    def log_likelihood(self, data=None):
        return self.glm.log_likelihood()

    def log_probability(self, data=None):
        return self.glm.log_probability()

    # Add a sampler for the network
    def collapsed_sample_AW(self, deg_gauss_hermite=10):
        """
        Do collapsed Gibbs sampling for an entry A_{n,n'} and W_{n,n'} where
        n = n_pre and n' = n_post.
        """
        gauss_hermite_abscissae, gauss_hermite_weights = \
            np.polynomial.hermite.hermgauss(deg_gauss_hermite)

        # TODO: Get the weight prior from a network object
        mu_w = self.W_prior.get_mu()
        sigma_w = self.W_prior.get_sigma()

        # rho = 0.5 * np.ones(self.N)
        rho = self.A_prior.get_rho()

        A = self.An
        W = self.Wn

        for n_pre in range(self.N):
            # print "Sampling A and W for ", n_pre, " to ", self.n

            # Compute the prior probabilities
            prior_lp_A = np.log(rho[n_pre])
            prior_lp_noA = np.log(1.0-rho[n_pre])

            # First approximate the marginal likelihood with an edge
            self.network_component.set_A(n_pre, 1.0)

            # Approximate G = \int_0^\infty p({s,c} | A, W) p(W_{n,n'}) dW_{n,n'}
            log_L = np.zeros(deg_gauss_hermite)
            weighted_log_L = np.zeros(deg_gauss_hermite)
            W_nns = np.sqrt(2) * sigma_w[n_pre] * gauss_hermite_abscissae + mu_w[n_pre]
            for i in np.arange(deg_gauss_hermite):
                # Set the weight for the incoming connection
                self.network_component.set_W(n_pre, W_nns[i])

                # Compute the Gauss hermite weight
                w = gauss_hermite_weights[i]


                # Compute the log likelihood
                log_L[i] = self.glm.log_likelihood()
                assert self.network_component.get_A()[n_pre] == 1

                # Handle NaNs in the GLM log likelihood
                if np.isnan(log_L[i]):
                    log_L[i] = -np.Inf

                weighted_log_L[i] = log_L[i] + np.log(w/np.sqrt(np.pi))

                # Handle NaNs in the GLM log likelihood
                if np.isnan(weighted_log_L[i]):
                    weighted_log_L[i] = -np.Inf

            # compute log pr(A_nn) and log pr(\neg A_nn) via log G
            log_G = logsumexp(weighted_log_L)
            if not np.isfinite(log_G):
                print weighted_log_L
                raise Exception("log_G not finite")

            # Compute log Pr(A_nn=1) given prior and estimate of log lkhd after integrating out W
            log_pr_A = prior_lp_A + log_G

            # Compute log Pr(A_nn = 0 | {s,c}) = log Pr({s,c} | A_nn = 0) + log Pr(A_nn = 0)
            self.network_component.set_A(n_pre, 0)
            log_pr_noA = prior_lp_noA + self.glm.log_likelihood()

            if np.isnan(log_pr_noA):
                log_pr_noA = -np.Inf

            # Sample A
            try:
                A[n_pre] = log_sum_exp_sample(np.array([log_pr_noA, log_pr_A]))
                self.network_component.set_A(n_pre, A[n_pre])

                # DEBUG
                if np.allclose(rho[n_pre], 1.0) and not A[n_pre]:
                    print "Sampled no self edge"
                    print log_pr_noA
                    print log_pr_A
                    # raise Exception("Sampled no self edge")
            except Exception as e:
                raise e

            # Sample W from its posterior, i.e. log_L with denominator log_G
            # If A_nn = 0, we don't actually need to resample W since it has no effect
            if A[n_pre] == 1:
                W[n_pre] = self.adaptive_rejection_sample_w(n_pre, mu_w[n_pre], sigma_w[n_pre], W_nns, log_L)
            else:
                # Sample W from the prior
                W[n_pre] = mu_w[n_pre] + sigma_w[n_pre] * np.random.randn()

            self.network_component.set_W(n_pre, W[n_pre])


    def adaptive_rejection_sample_w(self, n_pre, mu_w, sigma_w, ws_init, log_L):
        """
        Sample weights using adaptive rejection sampling.
        This only works for log-concave distributions, which will
        be the case if the nonlinearity is convex and log concave, and
        when the prior on w is log concave (as it is when w~Gaussian).
        """
        log_prior_W = -0.5/sigma_w**2 * (ws_init-mu_w)**2
        log_posterior_W = log_prior_W + log_L

        #  Define a function to evaluate the log posterior
        # For numerical stability, try to normalize
        Z = np.amax(log_posterior_W)
        def _log_posterior(ws_in):
            ws = np.asarray(ws_in)
            shape = ws.shape
            ws = np.atleast_1d(ws)
            lp = np.zeros_like(ws)
            for (i,w) in enumerate(ws):
                self.network_component.set_W(n_pre, w)
                lp[i] = -0.5/sigma_w**2 * (w-mu_w)**2 + \
                        self.glm.log_likelihood() - Z

            if isinstance(ws_in, np.ndarray):
                return lp.reshape(shape)
            elif isinstance(ws_in, float) or isinstance(ws_in, np.float):
                return np.float(lp)

        # Only use the valid ws
        valid_ws = np.bitwise_and(np.isfinite(log_posterior_W),
                                  log_posterior_W > -1e8,
                                  log_posterior_W < 1e8)

        ars = AdaptiveRejectionSampler(_log_posterior, -np.inf, np.inf, ws_init[valid_ws], log_posterior_W[valid_ws] - Z)
        return ars.sample()


    def resample(self):
        self.glm.resample()

        # Now sample the weights
        self.collapsed_sample_AW()

    # Properties for the GLM state
    def impulse_response(self, dt):
        dt_max = self.params.impulse.dt_max
        # Get the L x B basis
        ibasis = self.impulse_basis.interpolate_basis(dt, dt_max)
        # The N x B weights
        weights = self.impulse_component.get_w_ir()
        return np.dot(weights, ibasis.T)

    @property
    def bias(self):
        return self.bias_component.get_bias()

    @property
    def w_ir(self):
        return self.impulse_component.get_w_ir()

    @property
    def An(self):
        # Return the n-th column of A
        return self.network_component.get_A()

    @property
    def Wn(self):
        # Return the n-th column of W
        return self.network_component.get_W()

