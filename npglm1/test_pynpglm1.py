import numpy as np
import pyeigenglm as pe

import matplotlib.pyplot as plt

# Create a spike train
T = 100
N = 2
dt = 0.001
S = np.random.randint(0,10,T).astype(np.double)

# Filter the spike train
filtered_S = []
D_imp = 1
for n in range(N):
    # filtered_S.append(np.random.randn(T, D_imp).astype(np.double))
    filtered_S.append(np.zeros((T, D_imp)).astype(np.double))

st = pe.PyNpSpikeTrain(N, T, dt,  S, D_imp, filtered_S)

glm = pe.PyNpGlm(N, D_imp)
glm.add_spike_train(st)

print glm.log_likelihood()
print glm.log_probability()

plt.figure()
fr = np.empty(T)
glm.firing_rate(st, fr)
plt.plot(np.arange(T), fr)
plt.show()
print fr
#
# emp_rate = np.sum(S)/T
# print "Empirical rate: ", emp_rate, " spks/bin"
# N_steps = 10000
# for n in range(N_steps):
#     glm.coord_descent_step(0.001)
#     rate = np.log(1. + np.exp(glm.bias))
#     ll = glm.log_likelihood()
#
#     if np.mod(n, 25) == 0:
#         print "Iter: ", n, "\Rate: ", rate, " spks/bin\tLL:", ll
