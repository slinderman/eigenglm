import numpy as np
from eigenglm.nbregression import ScalarRegressionFixedCov, SpikeAndSlabNegativeBinomialRegression
from eigenglm.deps.pybasicbayes.distributions import GaussianFixedCov, GaussianFixedMean

def test_ss_nbregression(do_plot=False):
    # Make a model
    N = 20
    xi = 10
    sigma_psi = np.array(0.1)

    b = np.array([-1.0])
    sigma_b = np.array(1.0)

    B = 5
    w = np.random.randn(N,B)
    Sigma_w =  0.1 * np.eye(B)

    rho = min(0.8, 3./N)
    A = np.random.rand(N) < rho

    # Make regression models for each dimension
    true_bias_model = GaussianFixedCov(mu=b, sigma=sigma_b)
    true_regression_models = [ScalarRegressionFixedCov(A=w[d,:], sigma=Sigma_w) for d in range(N)]
    true_noise_model  = GaussianFixedMean(mu=np.zeros(1,), sigma=sigma_psi)
    true_model = SpikeAndSlabNegativeBinomialRegression(true_bias_model, true_regression_models, true_noise_model, As=A, xi=xi)

    # Make synthetic data
    datasets = 1
    Xss = []
    ys = []
    T = 60000
    for i in range(datasets):
        X = np.random.normal(size=(T,N*B))
        Xs = [X[:,(B*d):(B*(d+1))].reshape((T,B)) for d in range(N)]
        y = true_model.rvs(Xs, return_xy=False)
        print "Max y:\t", np.amax(y)

        Xss.append(Xs)
        ys.append(y)

    # Fit with the same model
    inf_noise_model  = GaussianFixedMean(mu=np.zeros(1,), nu_0=1, lmbda_0=np.eye(1))
    inf_bias_model = GaussianFixedCov(mu_0=np.zeros((1,)), lmbda_0=np.ones((1,1)), sigma=inf_noise_model.sigma)
    inf_regression_models = [ScalarRegressionFixedCov(mu_A=np.zeros((B,)),
                                                      Sigma_A=np.eye(B),
                                                      sigma=inf_noise_model.sigma)
                             for _ in range(N)]

    inf_model = SpikeAndSlabNegativeBinomialRegression(inf_bias_model, inf_regression_models, inf_noise_model,
                                                       rho_s=rho*np.ones(N), xi=xi)

    # Add data
    for Xs,y in zip(Xss, ys):
        inf_model.add_data(Xs, y)

    # Prepare samples
    N_samples = 10
    A_samples = []

    if do_plot:
        assert B == 1, "Can only plot 1D regressors"
        # Scatter the data
        import matplotlib.pyplot as plt
        plt.figure()
        plt.gca().set_aspect('equal')
        if T > 100:
            inds = (T * np.random.random(100)).astype(np.int)
        else:
            inds = np.arange(T)
        plt.scatter(Xss[0][0][inds], Xss[0][1][inds], c=ys[0][inds], cmap='hot')
        plt.title('%d / %d Datapoints' % (len(np.unique(inds)), T))
        plt.colorbar(label='Count')

        # Plot A
        l_true = plt.plot([0, A[0] * w[0]], [0, A[1] * w[1]], ':k')

        # Plot the initial sample
        l_inf = plt.plot([0, inf_model.As[0] * inf_regression_models[0].A[0]],
                         [0, inf_model.As[1] * inf_regression_models[1].A[0]], '-k')

        plt.ion()
        plt.show()

        # MCMC
        raw_input("Press any key to continue...\n")
        for i in range(N_samples):
            inf_model.resample()
            print "ll:\t", inf_model.log_likelihood(Xs, y)
            print "A:\t", inf_model.As
            print "bias:\t", inf_model.bias
            print "sigma:\t", inf_model.sigma

            A_samples.append(inf_model.As.copy())

            l_inf[0].set_data([0, inf_model.As[0] * inf_regression_models[0].A[0]],
                              [0, inf_model.As[1] * inf_regression_models[1].A[0]])
            plt.pause(0.1)
    else:
        # Profile instead of plot
        import cProfile, StringIO, pstats
        pr = cProfile.Profile()
        pr.enable()

        # MCMC
        for i in range(N_samples):
            print "Iteration ", i
            inf_model.resample()

            A_samples.append(inf_model.As.copy())

        # END Profiling
        pr.disable()

        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

    # Print posterior mean
    A_mean = np.array(A_samples).mean(axis=0)
    print "A_true: ", A.astype(np.float)
    print "A_inf: ", A_mean


test_ss_nbregression(do_plot=False)