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


class BiasParameters:
    # TODO: Implement distribution classes
    bias_distribution = "gaussian"
    bias_parameters = {"mu" : 5.0, "sigma" : 1.0}

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
            self.glms.append(NormalizedGLMParameters())

class NormalizedGLMPopulationParameters:
    def __init__(self, N):
        # TODO: Create latent variable parameters
        # TODO: Create network parameters

        self.glms = []
        for n in range(N):
            self.glms.append(NormalizedGLMParameters())
