import pynpglm3 as glm
import numpy as np

dtype = np.double
T = 100
dt = dtype(0.1)
S = np.random.randint(0,10,T).astype(dtype)
I_bias = dtype(1.0)
I_stim = np.random.randn(T).astype(dtype)


# Impulse responses
D_imp = 5
ir = np.random.randn(T,D_imp).astype(dtype)
w_ir = np.random.randn(D_imp).astype(dtype)
I_net = np.empty(T).astype(dtype)
glm.compute_I_net(T, D_imp, ir, w_ir, I_net)

ll = glm.log_likelihood(T, dt, S, I_bias, I_stim, I_net)
print "ll: ", ll



