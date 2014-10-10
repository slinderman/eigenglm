import numpy as np
import pyeigenglm as pe

import matplotlib.pyplot as plt

def create_test_data():
    # Create M spike trains
    M = 6
    N = 2
    D_imp = 1
    glm = pe.PyGlm(N, D_imp)
    sts = []
    for m in range(M):
        T = 10000
        dt = 1.0
        S = np.random.randint(0,10,T).astype(np.double)

        # Filter the spike train
        filtered_S = []
        for n in range(N):
            filtered_S.append(np.random.randn(T, D_imp).astype(np.double))
            # filtered_S.append(np.zeros((T, D_imp)).astype(np.double))

        st = pe.PySpikeTrain(N, T, dt,  S, D_imp, filtered_S)
        sts.append(st)

        glm.add_spike_train(st)

    print glm.log_likelihood()
    print glm.log_probability()

    return glm, sts

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

def test_coord_descent(glm, sts):
    # Plot the first data
    plt.figure()
    st = sts[0]
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
            # plt.plot(np.arange(T), glm.get_firing_rate(st))
            plt.pause(0.001)


# Run the script
glm, sts = create_test_data()

for i in range(5):
    test_w_ir_grads(glm, sts)

test_coord_descent(glm, sts)
