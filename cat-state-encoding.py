
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

# In[1104]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import animation, rc
from IPython.display import HTML


# In[1105]:


from qutip import *
#from qutip import identity, sigmax, sigmay, sigmaz, sigmam, sigmap, tensor, projection, create, destroy, displace
#from qutip import Qobj, basis, coherent, mesolve, fock
#from qutip import expect
from qutip.superoperator import liouvillian, sprepost
from qutip.qip import hadamard_transform
from qutip.visualization import plot_wigner, plot_wigner_fock_distribution
from qutip.ipynbtools import plot_animation
import qutip.logging_utils as logging
#from qutip import Bloch
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.WARN
#log_level = logging.WARN
#QuTiP control modules
import qutip.control.pulseoptim as cpo
from numpy import pi, sqrt
file_name = 'Test1'


# In[1106]:


from time import time

def printTime(start):
    end = time()
    duration = end - start
    if duration < 60:
        return "used: " + str(round(duration, 2)) + "s."
    else:
        mins = int(duration / 60)
        secs = round(duration % 60, 2)
        if mins < 60:
            return "used: " + str(mins) + "m " + str(secs) + "s."
        else:
            hours = int(duration / 3600)
            mins = mins % 60
            return "used: " + str(hours) + "h " + str(mins) + "m " + str(secs) + "s."


# # Physics
# ## Hamiltonian
# $$ \hat{H} =  \underbrace{(\omega_a - \chi_{sa}\au\ad)\bu\bd}_{\text{Storage-ancilla coupling}} +\, \omega_s\au\ad  \,-\, \frac{K_s}{2}\au{}^2\ad{}^2 \,-\, \frac{K_a}{2}\bu{}^2\bd{}^2 \,+\, \underbrace{\epsilon_a(t)\bu + \epsilon_a^*(t)\bd}_{\text{Qubit drive}} \,+\, \underbrace{\epsilon_s(t)\au + \epsilon_s^*(t)\ad}_{\text{Res drive}} $$
# 
# $$ \bu\bd = \ket{1}\bra{1} = \sigma_-\sigma_+ $$

# In[1323]:


N = 14 # Hilbert space size
alpha = sqrt(1)

Si = identity(2)
Ri = identity(N)
I  = tensor(Si,Ri)
Sx = tensor(sigmax(), Ri)
Sy = tensor(sigmay(), Ri)
Sz = tensor(sigmaz(), Ri)
Sm = tensor(sigmam(), Ri)
Sp = tensor(sigmap(), Ri)
a  = tensor(Si, destroy(N))
b  = Sp
d  = tensor(Si, displace(N,1))
di  = tensor(Si, displace(N,1j))

def hamiltonian(w_q, w_r, chi_qr, use_dispersive = True, use_kerr = False):
    # Hamiltonian - RWA JC, qubit-storage coupling
    #w_q = 2*pi*6.2815      # Energy of the 2-level system (GHz)
    #w_r = 2*pi*8.3056      # Resonator freq
    #chi_qr= 2*pi*1.97e-3     # qubit-storage coupling strength
    K_r   = 2*pi*0.45e-3   # Kerr res
    K_q   = 2*pi*297e-3    # Kerr qubit 200-300 MHz

    #w_r = 2.0 * 2 * pi      # resonator frequency
    #w_q = 3.0 * 2 * pi      # qubit frequency
    #chi_qr = 0.025 * 2 * pi   # parameter in the dispersive hamiltonian

    delta = abs(w_r - w_q)    # detuning
    g = sqrt(delta * chi_qr)  # coupling strength that is consistent with chi

    #H_occ = w_r*a.dag()*a + w_q*b.dag()*b
    H_occ = -w_q/2.0 * Sz  + w_r* a.dag()*a

    if use_dispersive:
        #H_coup = - chi_qr * a.dag()*a * b.dag()*b
        H_coup =  chi_qr * (a.dag() * a + I/2) * Sz
    else:
        #H_coup = g * (a.dag() * b + a * b.dag())
        H_coup = g * Sx *(a.dag() + a)
    if use_kerr:
        H_kerr = - K_r/2 * a.dag()**2 * a**2 - K_q/2 * b.dag()**2 * b**2
    else:
        H_kerr = tensor(qzero(2), qzero(N))
    
    H0 = H_occ + H_coup + H_kerr
    #H0 = I
    #f, ax = plt.subplots(1,2)
    
    #H1 = -w_q*Sz + w_r*a.dag()*a +  chi_qr * (a.dag() * a + I/2) * Sz
    #ax[0].imshow(np.real(H0.full()))
    #ax[1].imshow(np.real(H1.full()))
    return H0

#decay = [np.sqrt(gamma)*b]
#L0 = liouvillian(H0, decay)
#sigma X control
#LC_x = liouvillian(Sx)
#sigma Y control
#LC_y = liouvillian(Sy)
#sigma Z control
#LC_z = liouvillian(Sz)


w_q_true = 2*pi*6.2815      # Energy of the 2-level system (GHz) (NORMALISED)
w_q = 0.7563

w_r_true = 2*pi*8.3056      # Resonator freq
w_r = 1.0           # In units of w_q
#w_r = 0.75

#chi_qr= 2*pi*1.97e-3     # qubit-storage coupling strength
chi_qr = 2.37189366e-4
#chi_qr = 0

H0 = hamiltonian(w_q, w_r, chi_qr, use_dispersive=True, use_kerr=True)
#H1 = hamiltonian(w_q, w_r, chi_qr, use_dispersive=False, use_kerr=False)

drift = H0
#Controls - 

#ctrls = [liouvillian(Sx), liouvillian(Sz)]
#ctrls = [b.dag(), b]
ctrls = [Sx, Sz, d]
ctrls = [b, b.dag(), a.dag(), a]

#Damping rate:
gamma = 2*pi*2e-6 * w_r_true

# Starting state
#N_alpha = 1/(2*(1+np.exp(-2*abs(alpha)^2)))
#logical_0 = (coherent(N, alpha) + coherent(N,-alpha)).unit()
#logical_1 = (coherent(N, alpha*1j) + coherent(N,-alpha*1j)).unit()

# Start state
phi = tensor(basis(2,1), coherent(N,0))
# Target state
phi_targ = tensor(basis(2,0), (coherent(N,1j*alpha)).unit())
#phi_targ = phi


# # System check
# Some tests to see if the system is setup correctly

# In[1324]:


check_systems = False


# Is $\Delta \gg g$?

# ## Time evolution

# In[1325]:


if check_systems:
    #psi0 = tensor(basis(2,1), basis(N,4))
    psi0 = tensor((basis(2,1).unit()), coherent(N,1))
    psi0 = phi
    #psi0 = tensor(1.73*basis(2,0)+1*basis(2,1).unit(), coherent(N,0.5))
    #psi0 = tensor(basis(2,0), basis(N,4))
    t_tot = 1e5 * w_r_true        # nanoseconds
    t_tot = 100
    tlist = np.linspace(0,t_tot,1000)
    decay = False
    if decay:
        rate = [np.sqrt(gamma) * a]
    else:
        rate = []
    res = mesolve(H0, psi0, tlist, rate, [],options=Odeoptions(nsteps=2000000),progress_bar=True)


# ### Expectation values

# In[1326]:


if check_systems:
    nc_list = expect(a.dag()*a, res.states)
    nq_list = expect(b.dag()*b, res.states)

    fig, ax = plt.subplots(sharex=True,figsize=(8,5))
    ax.plot(tlist, nc_list, label="Cavity")
    ax.plot(tlist, nq_list, label="Atom excited state")
    ax.legend()
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Occupation probability');
    fig.tight_layout()


# ### Cavity quadratures

# In[1327]:


if check_systems:
    xc_list = expect((a + a.dag()), res.states)
    yc_list = expect(-1j*(a - a.dag()), res.states)

    fig, [ax,ax2] = plt.subplots(1,2,sharex=False, figsize=(12,4))

    ax.plot(tlist, xc_list, 'r', linewidth=2, label="q")
    ax.plot(tlist, yc_list, 'b', linewidth=2, label="p")
    ax.set_xlabel("Time (ns)", fontsize=16)
    ax.legend()

    ax2.plot(tlist, xc_list, 'r', linewidth=2, label="q")
    ax2.plot(tlist, yc_list, 'b', linewidth=2, label="p")
    ax2.set_xlabel("Time (ns)", fontsize=16)
    ax2.set_xlim(0,250)
    ax2.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(1,1,sharex=False, figsize=(12,4))

    ax.plot(xc_list,yc_list, 'k.', linewidth=2, label="q")
    ax.set_xlabel("q", fontsize=16)
    ax.set_ylabel("p", fontsize=16)
    ax.axis('equal')
    fig.tight_layout()


# ### Spectrum of resonator and qubit

# tlist2 = np.linspace(0, 2000, 10000)
# start = time()
# corr_vec = correlation_2op_2t(H0, psi0, None, tlist2, [], a.dag(), a, solver='me',options=Odeoptions(nsteps=5000))
# elapsed = printTime(start)
# print(elapsed)
# w, S = spectrum_correlation_fft(tlist2, corr_vec)

# print(elapsed)

# fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12,4))
# 
# ax.plot(tlist2, np.real(corr_vec), 'r', linewidth=2, label="resonator")
# ax.set_ylabel("correlation", fontsize=16)
# ax.set_xlabel("Time (ns)", fontsize=16)
# ax.legend()
# #ax.set_xlim(0,100)
# fig.tight_layout()
# 
# fig, ax = plt.subplots(figsize=(9,3))
# ax.plot(w / (2 * pi), abs(S))
# ax.set_xlabel(r'$\omega$', fontsize=18)
# ax.set_xlim(0.3,0.35)
# #ax.set_xlim(w_r/(2*pi)-.5, w_r/(2*pi)+.5);
# 
# fig, ax = plt.subplots(figsize=(9,3))
# ax.plot((w-w_r)/chi_qr, abs(S))
# ax.set_xlabel(r'$(\omega-\omega_r)/\chi$', fontsize=18);
# #ax.set_xlim(-5080,-5070);

# In[1328]:


def plot_states(states, is_qubit = False):
    f = plt.figure(figsize=(6*len(states), 6));
    for i, state in enumerate(states):
        if is_qubit:
            ax = f.add_subplot(1,len(states),i+1, projection='3d')
            bl = Bloch(fig=f, axes=ax)
            bl.add_states(state)
            bl.render(fig=f, axes=ax);
            #bl.show()
        else:
            plot_wigner_fock_distribution(state);
    f.show();
    

states = [phi, phi_targ]
qubit_states = [s.ptrace(0) for s in states]
res_states = [s.ptrace(1) for s in states]
plot_states(qubit_states, True)
plot_states(res_states)


# In[1329]:


# Time slot length
l_ts = 1
# Time allowed for the evolution (nanosec)
evo_time = 1 * w_r_true
evo_time = 2
# Number of time slots
n_ts = int(evo_time//l_ts + 1)
n_ts = 50


# In[1330]:


# Fidelity error target
fid_err_targ = 1e-8
# Maximum iterations for the optisation algorithm
max_iter = 1000
# Maximum (elapsed) time allowed in seconds
max_wall_time = 60*1
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'ZERO'
#Set to None to suppress output files
#f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)
f_ext = None


# In[1331]:


print("Initial fidelity: {}".format(fidelity((phi),(phi_targ))))
result = cpo.optimize_pulse(drift, ctrls, phi, phi_targ, n_ts, evo_time, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                out_file_ext=f_ext, init_pulse_type=p_type, 
                log_level=log_level, gen_stats=True,
                fid_params={'phase_option':'SU'},
                fid_type='TRACEDIFF',)
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=result.wall_time)))


# In[1332]:


states = [phi, phi_targ, result.evo_full_final]
qubit_states = [s.ptrace(0) for s in states]
res_states = [s.ptrace(1) for s in states]
plot_states(qubit_states, True)
plot_states(res_states)


# In[1333]:


def plot_control_pulses(result):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.set_title("Initial control amps")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Control amplitude")
    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.set_title("Optimised Control Sequences")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control amplitude")
    for i in range(len(ctrls)):
        ax1.step(result.time, 
                 np.hstack((result.initial_amps[:, i], result.initial_amps[-1, i])), 
                 where='post')
        ax2.step(result.time, 
         np.hstack((result.final_amps[:, i], result.final_amps[-1, i])), 
         where='post')
    fig1.tight_layout()
    
plot_control_pulses(result)

