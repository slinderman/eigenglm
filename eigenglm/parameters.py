import eigenglm.deps.pybasicbayes.distributions as pbd

class BasisParameters:
    type = 'cosine'
    n_eye = 0
    n_bas = 3
    a = 1.0/120
    b = 0.5
    L = 100
    orth = True
    norm = False

    @property
    def D(self):
        return self.n_eye + self.n_bas

class BiasHyperParameters:
    """
    Parameters of a Gaussian-Inverse-Chi Sq prior on
    the mean and variance of the Gaussian biases.
    """
    cls = pbd.ScalarGaussianNIX
    prms = {"mu_0" : 5.0,
            "sigmasq_0" : 1.0,
            "kappa_0" : 1.0,
            "nu_0" : 1.0}

class BiasParameters:
    pass

class LinearStimulusParameters:
    stimulus_distribution = "diagonal_gaussian"
    dt_max = 0.25
    basis = BasisParameters()

class LinearImpulseParameters:
    impulse_distribution = "diagonal_gaussian"
    dt_max = 0.25
    basis = BasisParameters()

class StandardGLMParameters:
    """
    Placeholder for a real parameters class
    TODO: Move the bias, background
    """
    bias = BiasParameters()
    stimulus = LinearStimulusParameters()
    impulse = LinearImpulseParameters()

class NormalizedGLMParameters:
    """
    Placeholder for a real parameters class
    TODO: Move the bias, background
    """
    bias = BiasParameters()
    stimulus = LinearStimulusParameters()

    # Use a nonnegative, normalized basis for impulse responses
    impulse = LinearImpulseParameters()
    impulse.basis.norm = True
    impulse.basis.orth = False

class StandardGLMPopulationParameters:
    def __init__(self, N):
        # TODO: Create latent variable parameters
        # TODO: Create network parameters

        self.glms = []
        for n in range(N):
            self.glms.append(StandardGLMParameters())

class NormalizedGLMPopulationParameters:
    def __init__(self, N):
        # TODO: Create latent variable parameters
        # TODO: Create network parameters

        # Hyperprior parameters
        self.bias_hyperprior = BiasHyperParameters()

        # GLM parameters
        self.glms = []
        for n in range(N):
            self.glms.append(NormalizedGLMParameters())
