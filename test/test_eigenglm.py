import numpy as np
import matplotlib.pyplot as plt

import eigenglm.cpp.pyeigenglm as pe
from eigenglm.utils.datahelper import *
from eigenglm.utils.basis import create_basis, convolve_with_basis


def create_test_data():
    # Create M spike trains
    M = 1
    N = 2
    D_imp = 5
    # glm = pe.PyStandardGlm(N, D_imp)
    glm = pe.PyNormalizedGlm(N, D_imp)
    sts = []
    for m in range(M):
        T = 600
        dt = 1.0
        S = np.random.randint(0,10,T).astype(np.double)

        # Filter the spike train
        filtered_S = []
        for n in range(N):
            filtered_S.append(np.random.randn(T, D_imp).astype(np.double))

        st = pe.PySpikeTrain(N, T, dt,  S, D_imp, filtered_S)
        sts.append(st)

        glm.add_spike_train(st)

    print glm.log_likelihood()
    print glm.log_probability()

    return glm, sts

def convert_data_to_spiketrain(datas, n_post, D_imp=5, dt_max=0.3):
    M = len(datas)

    # Initialize a basis
    from eigenglm.utils.basis import create_basis
    basis_params = \
    {
        'type' : 'cosine',
        'n_eye' : 0,
        'n_cos' : D_imp,
        'a': 1.0/120,
        'b': 0.5,
        'orth' : True,
        'norm' : False
    }
    basis = create_basis(basis_params)

    spiketrains = []
    for m in range(M):
        data = datas[m]
        dt = data['dt']
        S = data['S']
        T,N = S.shape

        # Interpolate basis at the resolution of the data
        (L,_) = basis.shape
        Lt_int = dt_max // dt
        t_int = np.linspace(0,1, Lt_int)
        t_bas = np.linspace(0,1,L)
        ibasis = np.zeros((len(t_int), D_imp))
        for b in np.arange(D_imp):
            ibasis[:,b] = np.interp(t_int, t_bas, basis[:,b])

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

        Sn_post = S[:,n_post].copy(order='C')
        st = pe.PySpikeTrain(N, T, dt,  Sn_post, D_imp, filtered_S)

        spiketrains.append(st)

    return spiketrains

def test_w_ir_grads(glm, sts):
    # Get the initial LL and weights
    ll0 = glm.log_likelihood()
    w0 = glm.get_w_ir()
    n_pre = 0

    # Get the gradient
    dll_dw = glm.get_dll_dw(sts[0], n_pre)
    print "dll_dw ", dll_dw

    for m in range(1, len(sts)):
        dll_dw += glm.get_dll_dw(sts[m], n_pre)

    # Add a small amount of noise to w
    wf = w0.copy()
    delta_w = 1e-6 * np.random.randn(glm.D_imp)
    wf[n_pre,:] += delta_w
    glm.set_w_ir(wf)

    wf_test = glm.get_w_ir()
    assert np.allclose(wf, wf_test)

    # Compute true and expected llf
    llf_true = glm.log_likelihood()
    llf_exp = ll0 + np.dot(dll_dw, delta_w)

    print "True dll:\t", llf_true - ll0
    print "Exp dll:\t", llf_exp - ll0

def test_g_ir_grads(glm, sts):
    # Get the initial LL and weights
    ll0 = glm.log_likelihood()
    g0 = glm.get_g_ir()
    print "g0: ", g0
    n_pre = 0

    # Get the gradient
    dll_dg = glm.get_dll_dg(sts[0], n_pre)
    print "dll_dg ", dll_dg

    for m in range(1, len(sts)):
        dll_dg += glm.get_dll_dg(sts[m], n_pre)

    # Add a small amount of noise to w
    gf = g0.copy()
    delta_g = np.zeros(glm.D_imp)
    delta_g[0] = 1e-6
    # delta_g = 1e-6 * np.random.randn(glm.D_imp)
    gf[n_pre,:] += delta_g
    glm.set_g_ir(gf)

    gf_test = glm.get_g_ir()
    assert np.allclose(gf, gf_test)

    # Compute true and expected llf
    llf_true = glm.log_likelihood()
    llf_exp = ll0 + np.dot(dll_dg, delta_g)

    print "True dll:\t", llf_true - ll0
    print "Exp dll:\t", llf_exp - ll0

def test_network_column(glm):
    print "A: ", glm.get_A()
    print "W: ", glm.get_W()

    W = np.random.randn(glm.N)
    for n in range(glm.N):
        print "Setting W[%d] = %.3f" % (n, W[n])
        glm.set_W(n, W[n])
        print glm.get_W()

    A = np.random.rand(glm.N) < 0.5
    A = A.astype(np.double)
    for n in range(glm.N):
        print "Setting A[%d] = %d" % (n, A[n])
        glm.set_A(n, A[n])
        print glm.get_A()

    print glm.log_probability()

def test_coord_descent(glm, sts):
    # Plot the first data
    plt.figure()
    st = sts[0]
    T = st.S.shape[0]
    fr = glm.get_firing_rate(st)
    lns = plt.plot(np.arange(st.T), fr)
    plt.ion()
    plt.show()

    raw_input("Press any key to continue...\n")

    emp_rate = np.sum(st.S)/st.T
    print "Empirical rate: ", emp_rate, " spks/bin"
    N_steps = 1000
    for n in range(N_steps):
        glm.coord_descent_step(0.0001)
        bias = glm.get_bias()
        # rate = glm.get_firing_rate(st)[0]
        ll = glm.log_likelihood()

        if np.mod(n, 25) == 0:
            # print "Iter: ", n, "\tBias: ", bias, "\tRate: ", rate, " spks/bin\tLL:", ll
            print "Iter: ", n, "\tBias: ", bias, " spks/bin\tLL:", ll
            plt.plot(np.arange(T), glm.get_firing_rate(st))
            plt.pause(0.001)

def test_resample(glm, sts):
    # Plot the first data
    plt.figure()
    st = sts[0]
    T = st.S.shape[0]
    fr = glm.get_firing_rate(st)
    lns = plt.plot(np.arange(st.T), fr)
    plt.ion()
    plt.show()

    raw_input("Press any key to continue...\n")

    emp_rate = np.sum(st.S)/st.T
    print "Empirical rate: ", emp_rate, " spks/bin"
    N_steps = 1000
    for n in range(N_steps):
        glm.resample()
        bias = glm.get_bias()
        # rate = glm.get_firing_rate(st)[0]
        ll = glm.log_likelihood()

        if np.mod(n, 25) == 0:
            # print "Iter: ", n, "\tBias: ", bias, "\tRate: ", rate, " spks/bin\tLL:", ll
            print "Iter: ", n, "\tBias: ", bias, " spks/bin\tLL:", ll
            # plt.plot(np.arange(T), glm.get_firing_rate(st))
            lns[0].set_data(np.arange(T), glm.get_firing_rate(st))
            plt.ylim(0,15)
            plt.pause(0.001)

# Run the script
# datafile = '../data/2014_10_10-16_16/data.pkl'
# with open(datafile) as f:
#     data = cPickle.load(f)
#     N = data['N']
#     n_post = 0
#     D_imp = 5
#     glm = pe.PyGlm(N, D_imp)
#
#     sts = convert_data_to_spiketrain([data], n_post, D_imp=D_imp)
#     for st in sts:
#         glm.add_spike_train(st)
#
#     print glm.log_likelihood()
#     print glm.log_probability()

glm, sts = create_test_data()

# for i in range(1):
#     # test_w_ir_grads(glm, sts)
#     test_g_ir_grads(glm, sts)

# test_coord_descent(glm, sts)
# test_resample(glm, sts)

test_network_column(glm)

