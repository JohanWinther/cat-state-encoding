#!/usr/bin/env python
# coding: utf-8

# $ \newcommand{\ket}[1]{\left|{#1}\right\rangle}
# \newcommand{\bra}[1]{\left\langle{#1}\right|} $
# $\newcommand{\au}{\hat{a}^\dagger}$
# $\newcommand{\ad}{\hat{a}}$
# $\newcommand{\bu}{\hat{b}^\dagger}$
# $\newcommand{\bd}{\hat{b}}$
# # Cat state encoding
# The main goal is to find control pulses which will realise the state transfer:
# $$ \underbrace{(c_0\ket{0} + c_1\ket{1})}_{\text{Qubit}}\underbrace{\ket{0}}_{\text{Cavity}} \rightarrow \ket{0}(c_0\ket{C_0} + c_1 \ket{C_1}) $$
# where $ \ket{C_0} \propto \ket{-\alpha} + \ket{\alpha} $ is the logical zero and $ \ket{C_1} \propto \ket{-i\alpha} + \ket{i\alpha} $ is the logical one. The method is to optimise such that the six cardinal points on the Bloch sphere realise these cavity cat states and puts the qubit to the ground state.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import datetime


# In[ ]:


from qutip import identity, sigmax, sigmay, sigmaz, sigmam, sigmap, tensor, projection, create
from qutip import Qobj, basis, coherent
from qutip.superoperator import liouvillian, sprepost
from qutip.qip import hadamard_transform
from qutip.visualization import plot_wigner, plot_wigner_fock_distribution
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo

file_name = 'Test1'


# In[ ]:


b = projection(2, 1, 1)
#b = Qobj([[0,0],[0,1]]).dag()
b.dag()**2 * b**2
sigmap()


# # Physics
# ## Hamiltonian
# $$ \hat{H} =  \underbrace{(\omega_a - \chi_{sa}\au\ad)\bu\bd}_{\text{Storage-ancilla coupling}} +\, \omega_s\au\ad  \,-\, \frac{K_s}{2}\au{}^2\ad{}^2 \,-\, \frac{K_a}{2}\bu{}^2\bd{}^2 \,+\, \underbrace{\epsilon_a(t)\bu + \epsilon_a^*(t)\bd}_{\text{Qubit drive}} \,+\, \underbrace{\epsilon_s(t)\au + \epsilon_s^*(t)\ad}_{\text{Res drive}} $$
# 
# $$ \bu\bd = \ket{1}\bra{1} $$

# In[ ]:


N = 30 # Hilbert space size
alpha = 3

Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Sm = sigmam()
Si = identity(2)
Ri = identity(N)
a  = create(N)
b  = projection(2, 1, 1)

# Hamiltonian - RWA JC, qubit-storage coupling
w_q = 1.0    # Energy of the 2-level system.
w_r = 1.0    # Resonator freq
X_qr= 1.0    # qubit-storage coupling strength
K_r   = 1.0    # Kerr res
K_q   = 1.0    # Kerr qubit

H0 = ( w_r* tensor(Si, a.dag()*a)
    + (w_q - X_qr * tensor(Si, a.dag()*a)) * tensor(b.dag()*b, Ri)
    - K_r/2 * tensor(Si,a.dag()**2 * a**2) - K_q/2 * tensor(b.dag()**2 * b**2, Ri))

#Amplitude damping
#Damping rate:
#gamma = 0.1
#L0 = liouvillian(H, [np.sqrt(gamma)*Sm])

#sigma X control
#LC_x = liouvillian(Sx)
#sigma Y control
#LC_y = liouvillian(Sy)
#sigma Z control
#LC_z = liouvillian(Sz)

#Drift
#drift = L0
drift = H0
#Controls - 
q_r = b.dag()
q_i = b
r_r = a.dag()
r_i = a

ctrls = [tensor(q_r, Ri), tensor(q_i, Ri),tensor(Si, r_r),tensor(Si, r_i)]

# Starting state
phi = tensor(basis(2,1), basis(N,0))

N_alpha = 1/(2*(1+np.exp(-2*abs(alpha)^2)))
res_targ_0 = (coherent(N, alpha*j) + coherent(N,-alpha*j)).unit()
#print(phi)
#print(res_targ_0)
# target for map evolution
phi_targ = tensor(basis(2,0), res_targ_0)
#print(phi_targ)


# In[ ]:


def plot_wigners(states):
    #f = plt.figure(figsize=(6*len(states), 6))
    for i, state in enumerate(states):
        #a = f.add_subplot(1,len(states),i+1)
        plot_wigner_fock_distribution(state,)#fig=f,ax = a)
        #a.axis('equal')

states = [phi, phi_targ]
states = [s.ptrace(1) for s in states]
plot_wigners(states)


# In[ ]:


# Time slot length
l_ts = 1e-9
# Time allowed for the evolution (sec)
evo_time = 500e-9
# Number of time slots
n_ts = int(evo_time//l_ts + 1)


# In[ ]:


# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 30
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'RND'
#Set to None to suppress output files
#f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)
f_ext = None
result = cpo.optimize_pulse(drift, ctrls, phi, phi_targ, n_ts, evo_time, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                out_file_ext=f_ext, init_pulse_type=p_type, 
                log_level=log_level, gen_stats=True)


# In[ ]:


result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=result.wall_time)))


# In[ ]:


fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for i in range(1):
    ax1.step(result.time, 
             np.hstack((result.initial_amps[:, i], result.initial_amps[-1, i])), 
             where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for i in range(1):
    ax2.step(result.time, 
             np.hstack((result.final_amps[:, i], result.final_amps[-1, i])), 
             where='post')
fig1.tight_layout()


# In[ ]:


res_final = result.evo_full_final.ptrace(1)
plot_wigner(res_final)


# In[ ]:


plot_wigner(phi_targ.ptrace(1))


# In[ ]:


plot_wigner(phi.ptrace(0))


# In[ ]:




