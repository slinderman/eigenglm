"""
GLM classes that use an underlying C++ implementation for inference and
likelihood calculation.
"""
import numpy as np

from eigenglm.parameters import *
from eigenglm.utils.basis import create_basis, convolve_with_basis
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
        self.impulse_basis = create_basis(params.impulse.basis)

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
        L = self.params.impulse.basis.L
        dt_max = self.params.impulse.dt_max

        # Interpolate basis at the resolution of the data
        Lt_int = dt_max // dt
        t_int = np.linspace(0,1, Lt_int)
        t_bas = np.linspace(0,1,L)
        ibasis = np.zeros((len(t_int), D_imp))
        for b in np.arange(D_imp):
            ibasis[:,b] = np.interp(t_int, t_bas, self.impulse_basis[:,b])

        # Filter the spike train
        filtered_S = []
        for n in range(N):
            Sn = S[:,n].reshape((-1,1))
            fS = convolve_with_basis(Sn, ibasis)

            # Flatten this manually to be safe
            # (there's surely a way to do this with numpy)
            (nT,Nc,Nb) = fS.shape
            assert Nc == 1 and Nb==D_imp, \
                "ERROR: Convolution with spike train " \
                "resulted in incorrect shape: %s" % str(fS.shape)
            filtered_S.append(fS[:,0,:])

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
