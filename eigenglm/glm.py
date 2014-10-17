"""
GLM classes that use an underlying C++ implementation for inference and
likelihood calculation.
"""
import numpy as np

from eigenglm.parameters import *
from eigenglm.utils.basis import Basis
import eigenglm.cpp.pyeigenglm as peg

class _GLM(object):
    """
    Base class to be subclassed with specific implementations.
    """
    def __init__(self, n, N):
        """
        Default constructor
        :param n: Index of this neuron in [0, N)
        :param N: Number of neurons in the population
        :return:
        """
        self.n = n
        self.N = N

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

    def log_likelihood(self, data=None):
        raise NotImplementedError()

    def log_probability(self, data=None):
        raise NotImplementedError()

    def resample(self):
        raise NotImplementedError()


##
#  Standard GLM implementation
#
class StandardGLM(_GLM):
    spiketrains = []
    glm = None

    def __init__(self, n, N, params):
        super(StandardGLM, self).__init__(n, N)

        assert isinstance(params, StandardGLMParameters)
        self.params = params

        # Create the necessary bases
        self.impulse_basis = Basis(params.impulse.basis)

        # Create the C++ wrapper object
        self.glm = peg.PyStandardGlm(N, self.params.impulse.basis.D)

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

    def log_likelihood(self, data=None):
        return self.glm.log_likelihood()

    def log_probability(self, data=None):
        return self.glm.log_probability()

    def resample(self):
        self.glm.resample()

class NormalizedGLM(_GLM):
    spiketrains = []
    glm = None

    def __init__(self, n, N, params):
        super(NormalizedGLM, self).__init__(n, N)

        assert isinstance(params, NormalizedGLMParameters)
        self.params = params

        # Create the necessary bases
        self.impulse_basis = Basis(params.impulse.basis)

        # Create the C++ wrapper object
        self.glm = peg.PyNormalizedGlm(N, self.params.impulse.basis.D)

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

    def log_likelihood(self, data=None):
        return self.glm.log_likelihood()

    def log_probability(self, data=None):
        return self.glm.log_probability()

    def resample(self):
        self.glm.resample()