
# coding: utf-8

# # Optimization of a State-to-State Transfer in a Two-Level-System

# In[ ]:


# NBVAL_IGNORE_OUTPUT
get_ipython().run_line_magic('load_ext', 'watermark')
#%load_ext autoreload
#%autoreload 2
import qutip
import numpy as np
import scipy
from ipywidgets import interact
import ipywidgets as widgets
import matplotlib
import matplotlib.pylab as plt
import krotov
import os
import copy
import subprocess
from bisect import bisect_left
import matplotlib2tikz
from scipy.signal import savgol_filter
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('watermark', '-v --iversions')
π = np.pi
sqrt = np.sqrt
basis = qutip.basis
tensor = qutip.tensor
coherent = qutip.coherent
from datetime import datetime
def current_time():
    return datetime.now().isoformat()[:16].replace('T',' ')


# In[ ]:


L = 2 # Qubit truncated Hilbert size
N = 8 # Resonator truncated Hilbert size
α = 2 # Mean photon number of resonator coherent states


# # Plotting and helper functions

# In[ ]:


def to_two_level(state):
    if state.type is 'oper':
        return qutip.Qobj(state[0:2,0:2])
    else:
        return qutip.Qobj(state[0:2])

def plot_population(n, tlist):
    fig, ax = plt.subplots(figsize=(7.5,4))
    leg = []
    for i in range(len(n)):
        ax.plot(tlist, n[i], label=str(i))
        leg.append('$|'+str(i)+'\\rangle$')
    ax.legend()
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Occupation')
    ax.legend(leg)
    #plt.show(fig)
    return fig

def plot_pulse(pulse, tlist, T=None, fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(7.5,4))
    else:
        ax = fig.axes[0]
    if callable(pulse):
        pulse = np.array([pulse(t, args=None) for t in tlist])
    if np.any(np.iscomplex(pulse)):
        ax.plot(tlist, np.real(pulse))
        ax.plot(tlist, np.imag(pulse))
        ax.legend(['Re', 'Im'])
    else:
        ax.plot(tlist, pulse)
    if T is not None:
        ax.plot(tlist, [S(t, T) for t in tlist], color='k', linestyle='--', linewidth=1)
        ax.plot(tlist, [-S(t, T) for t in tlist], color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Pulse amplitude')
    ax.set_ylim([-pulse_max(0)*1.05,pulse_max(0)*1.05])
    #plt.show()
    return fig

def plot_system(ψ):
    bl = qutip.Bloch()
    bl.add_states(to_two_level(ψ.ptrace(0)))
    bl.show()
    qutip.visualization.plot_wigner_fock_distribution(to_two_level(ψ.ptrace(1)))
def plot_resonator(ψ):
    fig, ax = plt.subplots(1,len(ψ), figsize=(3*len(ψ),3))
    if len(ψ)<=1:
        ψ = ψ[0]
        axis = ax
        qutip.visualization.plot_wigner(ψ.ptrace(1), fig=fig, ax=axis, alpha_max = 2*α)
        axis.axis_equal = True
    else:
        for (ϕ, axis) in zip(ψ, ax):
            qutip.visualization.plot_wigner(ϕ.ptrace(1), fig=fig, ax=axis, alpha_max = 2*α)
            axis.axis_equal = True
        
def plot_cardinal(ψ):
    bl = qutip.Bloch()
    bl.vector_color = ['r','g','b','g','b','r']
    [bl.add_states(to_two_level(ϕ.ptrace(0)), 'vector') for ϕ in ψ]
    bl.show()
    return bl

def plot_all(dyn, ψ):
    ψ_i = [g.states[0] for g in dyn]
    ψ_f = [g.states[-1] for g in dyn]
    ψ_t = [ϕ[1] for ϕ in ψ]
    plot_cardinal(ψ_i)
    plot_resonator(ψ_i)
    plot_cardinal(ψ_t)
    plot_resonator(ψ_t)
    plot_cardinal(ψ_f)
    plot_resonator(ψ_f)
def plot_evolution(dyn, steps=1, color_list=[[1,0,0]]):
    points = [s.ptrace(0) for s in dyn.states[0:-1:steps]]
    bl = qutip.Bloch()
    bl.vector_color = 'r'
    bl.point_color = color_list
    bl.point_marker = 'o'
    bl.add_states(points, 'point')
    bl.show()
    return bl
def get_objectives(T=None):
    if use_rotating:
        objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in state_rot(ϕ, T)]
    else:
        objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in ϕ]
    return objectives
def plot_matrix_final_target(target_state, final_state, xlabels, ylabels, el=30, az=135):
    fig, ax = qutip.visualization.matrix_histogram(final_state * target_state.dag(), xlabels, ylabels, colorbar=False, limits=[-1,1])
    qutip.visualization.matrix_histogram(proj(target_state), xlabels, ylabels, colorbar=False, limits=[-1,1], fig=fig, ax=ax)
    facecolors = np.zeros((6*L**2,4))*0.1
    edgecolors = np.tile([0,0,0,0.9], (6*L**2,1))
    ax.get_children()[2].set_facecolors(facecolors)
    ax.get_children()[2].set_edgecolors(edgecolors)
    ax.set_zticks(np.arange(-1,1,0.25))
    ax.view_init(elev=el, azim=az)
    return (fig, ax)


# In[ ]:


def F_oc(fw_states_T, objectives, tau_vals=None, **kwargs):
    return krotov.functionals.F_ss(fw_states_T, objectives, tau_vals, **kwargs)

def calc_fidelity(tau_vals):
    return np.abs(np.sum(tau_vals)/len(tau_vals))**2

def print_fidelity(**args):
    fid = calc_fidelity(np.array(args['tau_vals']))
    print("          F_t = {} | F = {} | F_t - F = {}".format(F_oc_tar, fid, F_oc_tar-fid))
def plot_fid_convergence(info_vals):
    fig, ax = plt.subplots(1,1)
    ax.plot(info_vals)
    ax.set_xticks(np.arange(0, len(info_vals), step=1))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity')
    #ax.set_ylim((-0.2,.2))
    plt.show()
def plot_fid_convergence(ax, info_vals, T):
    ax.plot3D(range(0,len(info_vals)), [T]*len(info_vals), info_vals)


# In[ ]:


def qubit_occupation(dyn):
    occ = [basis(L,i)*basis(L,i).dag() for i in range(0,L)]
    states = [state.ptrace(0) for state in dyn.states]
    n = qutip.expect(occ, states)
    fig = plot_population(n, dyn.times)
    return fig
def resonator_occupation(dyn, export=False):
    occ = [basis(N,i)*basis(N,i).dag() for i in range(0,N)]
    states = [state.ptrace(1) for state in dyn.states]
    n = qutip.expect(occ, states)
    if export is True:
        return n
    else:
        fig = plot_population(n, dyn.times);
        return fig
def plot_norm(result):
    state_norm = lambda i: result.states[i].norm()
    states_norm=np.vectorize(state_norm)

    fig, ax = plt.subplots()
    ax.plot(result.times, states_norm(np.arange(len(result.states))))
    ax.set_title('Norm loss', fontsize = 15)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('State norm')
    plt.show(fig)


# In[ ]:


def plot_spectrum(pulse, tlist, mark_freq=None, pos=1, xlim=None, mark_color=['k','k','k']):
    samples = len(tlist)
    sample_interval = tlist[-1]/samples
    signal_qubit = np.pad(pulse, (0, power_two-samples), mode='constant')
    samples = power_two
    time = np.linspace(0, samples*sample_interval, samples)

    #signal_qubit = pulse
    signal_spectrum = np.fft.fftshift(np.fft.fft(signal_qubit))
    freqs = np.fft.fftshift(np.fft.fftfreq(samples, d=sample_interval))

    fig, ax = plt.subplots(figsize=(10,5))
    start_idx = bisect_left(freqs, xlim[0]/(2*π))
    end_idx = bisect_left(freqs, xlim[1]/(2*π))
    ax.plot(freqs[start_idx:end_idx+1], np.abs(signal_spectrum[start_idx:end_idx+1])/len(signal_qubit))  # in GHz
    if mark_freq is not None:
        if not isinstance(mark_freq, list):
            mark_freq = [mark_freq]
        mf = np.array(mark_freq)/(2*π)
        if pos==1:
            ax.set_xlim(0, 2*mf[0])
        elif pos==-1:
            ax.set_xlim(-2*mf[0], 0)
        elif xlim is not None:
            ax.set_xlim(xlim[0]/(2*π), xlim[1]/(2*π))
        [ax.axvline(x=m_f, ymin=0, ymax=1, color=col, linestyle='--', linewidth=1) for (m_f, col) in zip(mf, mark_color)]
        #[ax.axvline(x=m_f, ymin=0, ymax=1, linestyle='--', linewidth=1) for (m_f, col) in zip(mf, mark_color)]
    ax.set_title('Qubit pulse spectrum')
    ax.set_xlabel('f (GHz)');
    return fig


# In[ ]:


def fid(final, target):
    return (np.abs((final.dag()*target).full())**2)[0][0]
def proj(ψ, ϕ=None):
    if ϕ is None:
        return ψ * ψ.dag()
    else:
        return ψ * ϕ.dag()


# In[ ]:


def plot_results_3d(results):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Iteration')
    ax.set_zlabel('Fidelity')
    ax.set_ylabel('Pulse length')
    ax.set_zlim(0,1.1)
    for (r, T) in results:
        plot_fid_convergence(ax, r.info_vals[1:], T)
    ax.view_init(elev=20, azim=340)
    return (fig, ax)

def plot_results_iteration(results, fig=None):
    if fig is None:
        fig = plt.figure()
        ax = plt.axes()
    else:
        ax = fig.axes[0]
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity')
    for (r, T) in results:
        ax.plot(range(0,len(r.info_vals)-1), r.info_vals[1:])
    #print('F = {}'.format(r.info_vals[-1]))
    return (fig, ax)

def plot_results_pulse_length_iterations(results):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('Pulse length (ns)')
    ax.set_ylabel('Iterations')
    x = [T for (r,T) in results]
    y = [r.iters[-1] for (r,T) in results]
    ax.stem(x, y)#, linestyle='None', color='k',marker='.')
    return (fig, ax)
    
def plot_results_pulse_length(results, iteration=-1, ax=None, shape='o',color='k'):
    if ax is None:
        ax = plt.axes()
    #else:
        #ax.clear()
    ax.set_xlabel('Pulse length')
    ax.set_ylabel('Fidelity')
    
    T_list = [T for (r, T) in results]
    fid_list = [r.info_vals[min(len(r.info_vals)-1,iteration)] for (r, T) in results]
    #for (r, T) in results:
    #    it = 
        
    ax.plot(T_list, fid_list, shape+color)
    ax.set_ylim(0,1.1)
    return ax


# In[ ]:


def F_oc(fw_states_T, objectives, tau_vals=None, **kwargs):
    return krotov.functionals.F_ss(fw_states_T, objectives, tau_vals, **kwargs)

def calc_fidelity(tau_vals):
    return np.abs(np.sum(tau_vals)/len(tau_vals))**2

def print_fidelity(**args):
    fid = calc_fidelity(np.array(args['tau_vals']))
    print("          F_t = {} | F = {} | F_t - F = {}".format(F_oc_tar, fid, F_oc_tar-fid))
def plot_fid_convergence(info_vals):
    fig, ax = plt.subplots(1,1)
    ax.plot(info_vals)
    ax.set_xticks(np.arange(0, len(info_vals), step=1))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity')
    #ax.set_ylim((-0.2,.2))
    plt.show()
def plot_fid_convergence(ax, info_vals, T):
    ax.plot3D(range(0,len(info_vals)), [T]*len(info_vals), info_vals)


# In[ ]:


from datetime import datetime
def current_time():
    return datetime.now().isoformat()[:16].replace('T',' ')


# In[ ]:


def pulse_max(σ):
    A = 1.56246130414 # Chosen such that the integral of any Blackman pulse = π
#    A = A/2
    σ = np.max((σ,3))
    return A/(np.sqrt(2*π)*σ)


# # System setup

# In[ ]:


σ_max = 3 # ns (gaussian pulse limit)
amp_max = pulse_max(0)

# Below are settings for testing, optimization settings are set in the optimization section
T = 60
σ = T/6
steps = 32*int(np.ceil(T))
tlist = np.linspace(0, T, steps)


# In[ ]:


cat_0 = (coherent(N, α) + coherent(N,-α)).unit()
cat_1 = (coherent(N, 1j*α) + coherent(N,-1j*α)).unit()

Si = qutip.operators.identity(L)
Ri = qutip.operators.identity(N)
I = tensor(Si,Ri)

zero_q = qutip.operators.qzero(L)
zero_r = qutip.operators.qzero(N)
zero_q = tensor(zero_q, Ri)
zero_r = tensor(Si, zero_r)

zero = zero_q + zero_r

σ_z = proj(qutip.basis(L, 0)) - proj(qutip.basis(L, 1))
σ_y = 1j*(proj(qutip.basis(L, 1),qutip.basis(L, 0)) - proj(qutip.basis(L, 0), qutip.basis(L, 1)))
σ_x = proj(qutip.basis(L, 0),qutip.basis(L, 1)) - proj(qutip.basis(L, 1), qutip.basis(L, 0))
σ_z = tensor(σ_z, Ri)
σ_y = tensor(σ_y, Ri)
σ_x = tensor(σ_x, Ri)

b = qutip.operators.destroy(L)
a = qutip.operators.destroy(N)
b = tensor(b, Ri)
a = tensor(Si, a)

ω_r = 8.3056 * 2 * π      # resonator frequency
ω_q = 6.2815 * 2 * π      # qubit frequency
K_q   = -2*π*297e-3    # Kerr qubit 200-300 MHz
K_r   = 2*π*4.5e-6   # Kerr res 1-10 Khz

ω_ef = ω_q + K_q
ω_gf = ω_q + K_q/2

χ = 0.025 * 2 * π   # parameter in the dispersive hamiltonian

Δ = abs(ω_r - ω_q)    # detuning
g = sqrt(Δ * χ)  # coupling strength that is consistent with chi
γ = 1e1  # Dissipation (unused)

use_rotating = True
def hamiltonian(ω=1.0, ampl0=1, use_rotating=True, pulses=None, tlist=None, start_pulse=None, T=1, σ=σ):
    """Two-level-system Hamiltonian
    
    Args:
        ω (float): energy separation of the qubit levels
        ampl0 (float): constant amplitude of the driving field
    """
    
    #H_occ = w_r*a.dag()*a + w_q*b.dag()*b
    #H_occ_q = qutip.Qobj(ω_q*np.diag(range(L)))
    #H_occ_q = tensor(H_occ_q, Ri)
    H_occ_q = ω_q * b.dag()*b
    H_occ_r = ω_r * a.dag()*a
    H_occ =  H_occ_q + H_occ_r
    
    
    use_dispersive = True
    use_kerr = True
    #if use_dispersive:
    #    H_coup = - χ * a.dag()*a * b.dag()*b
        #H_coup =  χ * (a.dag()*a + I/2) * σ_z
    #else:
    #    H_coup = g * (a.dag() * b + a * b.dag())
    #    H_coup = g * σ_x *a.dag() + a
    if use_kerr:
        H_kq =  -K_q/2 * b.dag()**2 * b**2
        H_kr =  -K_r/2 * a.dag()**2 * a**2
    else:
        H_kq = zero_q
        H_kr = zero_r
    
    H_d = zero_q + zero_q
    
    if use_rotating:
        H_d += H_kq + H_kr #+ H_coup
        
        H_qr = (b.dag() + b)
        H_qi = 1j*(b.dag() - b)
        H_rr = (a.dag() + a)
        H_ri = 1j*(a.dag() - a)
        
        H_cc = g*(a*b.dag() + a.dag()*b)
        H_cs = g*1j*(a*b.dag() - a.dag()*b)
        
        if start_pulse is None:
            ϵ_qr = lambda t, args: ampl0
            ϵ_qi = lambda t, args: ampl0
            ϵ_rr = lambda t, args: ampl0
            ϵ_ri = lambda t, args: ampl0
        else:
            ϵ_qr = shape_field(lambda t, args: ampl0, start_pulse, T, σ)
            ϵ_qi = shape_field(lambda t, args: ampl0, start_pulse, T, σ)
            ϵ_rr = shape_field(lambda t, args: ampl0, start_pulse, T, σ)
            ϵ_ri = shape_field(lambda t, args: ampl0, start_pulse, T, σ)
        
        
        ϵ_cc = lambda t, args: np.cos((ω_q - ω_r)*t)
        ϵ_cs = lambda t, args: np.sin((ω_q - ω_r)*t)
        
        if pulses:
            ϵ_qr = pulses[0]
            ϵ_qi = pulses[1]
            ϵ_rr = pulses[2]
            ϵ_ri = pulses[3]
            ϵ_cc = pulses[4]
            ϵ_cs = pulses[5]
        
        return [H_d, [H_qr, ϵ_qr], [H_qi, ϵ_qi], [H_rr, ϵ_rr], [H_ri, ϵ_ri], [H_cc, ϵ_cc], [H_cs, ϵ_cs]]
    else:
        H_coup = g*(a.dag()*b+a*b.dag())
        H_d += H_occ + H_kq + H_kr +H_coup
        
        H_q = b
        H_qc = b.dag()
        H_r = a
        H_rc = a.dag()
        
        H_cc = I
        H_cs = I
        
        if pulses:
            ϵ_q = pulses[0]
            ϵ_qc = pulses[1]
            ϵ_r = pulses[2]
            ϵ_rc = pulses[3]
            ϵ_cc = [0]*len(pulses[0])
            ϵ_cs = [0]*len(pulses[0])
        else:
            ϵ_q = lambda t, args: 0
            ϵ_qc = lambda t, args: 0
            ϵ_r = lambda t, args: 0
            ϵ_rc = lambda t, args: 0
            ϵ_cc = lambda t, args: 0
            ϵ_cs = lambda t, args: 0
        
        return [H_d, [H_q, ϵ_q], [H_qc, ϵ_qc], [H_r, ϵ_r], [H_rc, ϵ_rc], [H_cc, ϵ_cc], [H_cs, ϵ_cs]]
    
# Converts basis state coefficients into the corresponding states of the qubit-resonator system 
def coeffs_to_state(c,init = True):
    if init:
        ψ = tensor((c[0]*basis(L,0) + c[1]*basis(L,1)).unit() , (basis(N,0)))
    else:
        ψ = tensor((basis(L,0)) , (c[0]*cat_0 + c[1]*cat_1).unit())
    return ψ

# Feeds a list of coeffients into the function above
def states(coeffs):
    return [[coeffs_to_state(c,True),coeffs_to_state(c,False)] for c in coeffs]


# In[ ]:


H = hamiltonian(ampl0=1, use_rotating=True)
coeffs = [(1,0), (1,-1), (1,1j), (1,1), (1,-1j), (0,1)] # Six states which will be transferred
#coeffs = [(1,0)]
ϕ = states(coeffs)
F_err = 1e-6 # Infidelity goal
F_oc_tar = 1-F_err # Fidelity goal


# In[ ]:


# Rotates the target states into the rotating frame
def state_rot(ϕ, T):
    ϕ = copy.deepcopy(ϕ)
    for i in range(len(ϕ)):
        U_0 = (1j*ω_q*b.dag()*b*T).expm()*(1j*ω_r*a.dag()*a*T).expm()
        ϕ[i] = U_0 * ϕ[i]
    return ϕ

#if use_rotating:
    #objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in ϕ]
#    objectives = [krotov.Objective(initial_state=ψ[0], target=state_rot(ψ[1],T), H=H) for ψ in ϕ]
#else:
#    objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in ϕ]


# In[ ]:


# Envelope function which will keep the pulse zero at edges with rise and fall time = 2 ns
def S(t, T=6*σ, σ=σ):
    rise_time = 2
    return amp_max*krotov.shapes.flattop(t, t_start=0, t_stop=T, t_rise=rise_time, t_fall=rise_time, func='sinsq')

# Blackman guess pulse
def start_pulse(t, T=6*σ, σ=σ):
    if σ is not None:
        return pulse_max(σ)*krotov.shapes.blackman(t, t_start=0, t_stop=6*σ)
    else:
        return krotov.shapes.blackman(t, t_start=0, t_stop=T)

def zero_pulse(t, T=T, σ=4):
    return 0
def unit_pulse(t, T=T, σ=4):
    return 1

def shape_field(ϵ, sf, T, σ):
    """Applies the shape function S(t) to the guess field"""
    ϵ_shaped = lambda t, args: ϵ(t, args)*sf(t, T=T, σ=σ)
    return ϵ_shaped

S_unit = [unit_pulse]*4
S_zero = [zero_pulse]*4
#S_start = [lambda t, T=T, σ=σ: 0.01*unit_pulse(t, T=T, σ=σ),start_pulse]
S_start = [zero_pulse, start_pulse, zero_pulse, zero_pulse]
S_funs = [S]*4
for i, H_i in enumerate(H[1:5]):
    H_i[1] = shape_field(H_i[1], S_start[i], T, σ)


# ## Simulate dynamics of the guess field
# 
# Before heading towards the optimization
# procedure, we first simulate the
# dynamics under the guess field
# $\epsilon_{0}(t)$.

# In[ ]:


for i, H_i in enumerate(H[1:]):
    plot_pulse(H_i[1], tlist)


# In[ ]:


guess_dynamics = [ob.mesolve(tlist, progress_bar=True, options=qutip.Options(nsteps=50000)) for ob in objectives]


# In[ ]:


final_state = guess_dynamics[0].states[-1]
plot_cardinal([ϕ[0][0].ptrace(0), ϕ[0][1].ptrace(0), final_state.ptrace(0)])
qutip.visualization.plot_wigner_fock_distribution(ϕ[0][1].ptrace(1))
qutip.visualization.plot_wigner_fock_distribution(final_state.ptrace(1))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


for i in range(len(guess_dynamics)):
    qubit_occupation(guess_dynamics[i]);
    resonator_occupation(guess_dynamics[i]);
#plot_norm(guess_dynamics[0])


# In[ ]:


tlist2 = np.linspace(0, T, 1000*int(np.ceil(T)))
rotating_pulses = [np.array([H_i[1](t,0) for t in tlist]) for H_i in H[1:]]
Ω_q = rotating_pulses[0]+1j*rotating_pulses[1]
Ω_q = np.interp(tlist2, tlist, Ω_q)
Ω_r = rotating_pulses[2]+1j*rotating_pulses[3]
Ω_r = np.interp(tlist2, tlist, Ω_r)
pulses_lab = [Ω_q*np.exp(1j*ω_q*tlist2), np.conj(Ω_q)*np.exp(-1j*ω_q*tlist2), Ω_r*np.exp(1j*ω_r*tlist2), np.conj(Ω_r)*np.exp(-1j*ω_r*tlist2)]
H_lab = hamiltonian(ampl0=1, use_rotating=False, pulses=pulses_lab)
objectives_lab = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H_lab) for ψ in ϕ]
r = krotov.result.Result()
r.optimized_objectives = objectives_lab
opt_dynamics_lab = [ob.mesolve(tlist2, progress_bar=True, options=qutip.Options(nsteps=50000)) for ob in r.optimized_objectives]


# In[ ]:


#plot_all(guess_dynamics, ϕ)
plot_evolution(guess_dynamics, steps=5)


# In[ ]:


qubit_pulses = [H[2][1](t, 0) for t in tlist]
res_pulses = [H[4][1](t, 0) for t in tlist]
#qubit_pulses_filtered = apply_spectral_filter(copy.deepcopy(qubit_pulses), tlist, 0, 0.5)
plot_spectrum(qubit_pulses, tlist, mark_freq=[0, -K_q, -K_q/2], pos=0, xlim=[-2*π,2*π], title="Qubit")
plot_spectrum(res_pulses, tlist, mark_freq=[0, -K_r, -K_r/2], pos=0, xlim=[-π/4,π/4], title="Resonator")
#plot_spectrum(qubit_pulses_filtered, tlist, mark_freq=[0, K_q], pos=0, xlim=[-2*K_q,2*K_q])
#plot_pulse(qubit_pulses, tlist)
#plot_pulse(qubit_pulses_filtered, tlist)
#plot_spectrum(qubit_pulses[1], time_list, mark_freq=[ω_q, ω_q + K_q, ω_q - K_q], pos=0)


# ## Optimize

# In[ ]:


# Frequency limits for spectral filter (not used)
ω_0 = 0
ω_1 = np.abs(K_q/2)


# In[ ]:


def modify_params(**kwargs):
    
    spectral_filtering = False
    
    if spectral_filtering:
        # Spectral filtering
        tlist = kwargs['tlist']
        dt = tlist[1] - tlist[0]  # assume equi-distant time grid
        n = len(tlist) - 1  # = len(pulse)
        # remember that pulses are defined on intervals of tlist
        ω = np.abs(np.fft.fftfreq(n, d=dt / (2.0 * π)))
        # the normalization factor 2π means that w0 and w1 are angular
        # frequencies, corresponding directly to energies in the Hamiltonian
        # (ħ = 1).
        flt = (ω_0 <= ω) * (ω <= ω_1)
        # flt is the (boolean) filter array, equivalent to an array of values 0
        # and 1 

        shape_arrays = kwargs['shape_arrays']
        for (i, (pulse, shape)) in enumerate(zip(kwargs['optimized_pulses'], shape_arrays)):
            spectrum = np.fft.fft(pulse)
            # apply the filter by element-wise multiplication
            spectrum[:] *= flt[:]
            # after the inverse fft, we should also multiply with the
            # update shape function. Otherwise, there is no guarantee that
            # the filtered pulse will be zero at t=0 and t=T (assuming that
            # is what the update shape is supposed to enforce). Also, it is
            # important that we overwrite `pulse` in-place (pulse[:] = ...)
            kwargs['optimized_pulses'][i][:] = np.fft.ifft(spectrum).real * shape
    
    # Limit pulse amplitude to 1 and smoothen
    for i in range(len(kwargs['optimized_pulses'])-2):
        #kwargs['optimized_pulses'][i] = savgol_filter(kwargs['optimized_pulses'][i], 9, 2)
        
        pulse_max = np.max(np.abs(kwargs['optimized_pulses'][i]))
        if pulse_max > amp_max:
            kwargs['optimized_pulses'][i] = (amp_max*np.array(kwargs['optimized_pulses'][i])/pulse_max)
        kwargs['optimized_pulses'][i] = np.fmax(np.fmin(kwargs['optimized_pulses'][i], kwargs['shape_arrays'][i]), -np.array(kwargs['shape_arrays'][i]))
        
        #conv = 3*σ
        #if (conv % 2 == 0): conv += 1
        #kwargs['optimized_pulses'][i] = savgol_filter(kwargs['optimized_pulses'][i], conv, 2)
    #for i in range(len(kwargs['optimized_pulses'])):
    #    if kwargs['iteration'] % 20 == 0:
    #        plot_pulse(kwargs['optimized_pulses'][i], kwargs['tlist'][:-1], kwargs['tlist'][-1])
    #if kwargs['iteration'] % 20 == 0:
    #    for i in range(len(kwargs['fw_states_T'])):
    #        final_state = kwargs['fw_states_T'][i]
    #        plot_cardinal([final_state])
    #        qutip.visualization.plot_wigner_fock_distribution(final_state.ptrace(1))
            #plot_spectrum(kwargs['optimized_pulses'][i], kwargs['tlist'][:-1], mark_freq=[0, -K_q, -K_q/2], mark_color=['r','g','b'], pos=0, xlim=[-(2*π), (2*π)])
    # Update λ
    #fac = 1
    #steps = 5
    #λₐ = kwargs['lambda_vals'][0]
    #for i in range(len(kwargs['lambda_vals'])):
    #    kwargs['lambda_vals'][i] = λₐ * fac
    #    lambda_a = λₐ * fac
    #print("λₐ = {}".format(kwargs['lambda_vals']))


# In[ ]:


# Reset results
opt_result = None


# In[ ]:


def convergence_reason(opt_result):
    if opt_result == None:
        return True
    reasons = ['monotonic', 'iterations']
    for r in reasons:
        if opt_result.message.find(r)>0:
            return r
    if opt_result.message.find('F_oc')>0 or opt_result.message.find('Δ')>0:
        return False


# In[ ]:


folder = 'results'
results = [(krotov.result.Result.load(os.path.join(os.getcwd(),folder,file), objectives=get_objectives(T=float(file.split('_')[-1][:-4]))), float(file.split('_')[-1][:-4])) for file in os.listdir(folder) if file[-4:]=='.dat']
results = [results[-1]]
print(results[-1][1])
pulses = results[0][0].optimized_controls
old_tlist = results[0][0].tlist


# In[ ]:


def run_optim(T, lambda_a, ϕ, pulses = None, old_tlist=None):
    σ = T/6
    print('T = {}'.format(T))
    #steps = 500
    tlist = np.linspace(0, T, 24*int(np.ceil(T)))
    s_pulse = None
    if pulses:
        pulses = [np.interp(tlist, old_tlist, pulse) for pulse in pulses]
        print(len(tlist))
        print(len(pulses[0]))
        H = hamiltonian(ampl0=1, use_rotating=True, start_pulse=s_pulse, T=T, pulses=pulses)
    else:
        H = hamiltonian(ampl0=1, use_rotating=True, start_pulse=s_pulse, T=T)
    
    
    
    S_start = [zero_pulse, start_pulse, zero_pulse, zero_pulse]
    S_funs = [S, S, S, S, lambda t, T, σ:0, lambda t, T, σ:0]
    #for i, H_i in enumerate(H[1:]):
    #    if i < 4:
    #        H_i[1] = shape_field(H_i[1], S_start[i], T, σ)
        #    plot_pulse(H_i[1], tlist)
        #else:
        #    plot_pulse(H_i[1], tlist)
    
    objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in state_rot(ϕ, T)]
    #objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in ϕ]
    if 'latest.dat' in os.listdir('results'):
        opt_result = krotov.result.Result.load('results/latest.dat', objectives=objectives)
    else:
        opt_result = None
    # Check if guess pulse realises
    #guess_dynamics = [ob.mesolve(tlist, options=qutip.Options(nsteps=50000)) for ob in objectives]
    #final_state = guess_dynamics[0].states[-1]
    #dm = final_state * ϕ[0][1].dag()
    #fid = np.abs((final_state.dag() * ϕ[0][1]).full()[0][0])**2
    #print('F = {}'.format(fid))
    #if fid > F_oc_tar:
    #    print('Guess pulse realises transfer already.')
    #    return True
    if pulses:
        pulse_options = {id(H_i[1]): { 'lambda_a':lambda_a, 'shape':lambda t: S_funs[i](t, T=T, σ=σ)} for i, H_i in enumerate(H[1:])}
    else:
        pulse_options = {(H_i[1]): dict(lambda_a=lambda_a, shape=lambda t: S_funs[i](t, T=T, σ=σ)) for i, H_i in enumerate(H[1:5])}
        pulse_options.update({(H_i[1]): dict(lambda_a=lambda_a, shape=lambda t: 0) for i, H_i in enumerate(H[5:])})
    
    #pulse_options = {
    #    H[2][1]: dict(lambda_a=lambda_a, shape=0),
    #    H[1][1]: dict(lambda_a=lambda_a, shape=lambda t: S_funs[0](t, T=T, σ=σ)),    
    #}
    #while convergence_reason(opt_result):
    #    if convergence_reason(opt_result) == 'monotonic':
    #        break
            #lambda_a *= 2
        #    print('λₐ = {}'.format(lambda_a))
        #    pulse_options = {H_i[1]: dict(lambda_a=lambda_a, shape=lambda t: S_funs[i](t, T)) for i, H_i in enumerate(H[1:])}
        #iters = 5
        #if opt_result is not None:
        #    iters = opt_result.iters[-1] + iters
    iters = 10 # Save to file every 10th iteration
    if opt_result is not None:
        iters = opt_result.iters[-1] + iters
    
    opt_result = krotov.optimize_pulses(
        objectives,
        pulse_options=pulse_options,
        tlist=tlist,
        propagator=krotov.propagators.expm,
        chi_constructor=krotov.functionals.chis_ss,
        info_hook=krotov.info_hooks.chain(
            krotov.info_hooks.print_table(J_T=krotov.functionals.F_ss),
            print_fidelity
        ),
        check_convergence=krotov.convergence.Or(
            krotov.convergence.value_above(F_oc_tar, name='F_oc'),
            krotov.convergence.delta_below(1e-6),
            #krotov.convergence.delta_below(F_err*1e-1),
            #krotov.convergence.check_monotonic_fidelity,
        ),
        modify_params_after_iter = modify_params,
        iter_stop=iters,
        continue_from = opt_result,
        store_all_pulses=False,
        skip_initial_forward_propagation=False,
        parallel_map=(
            qutip.parallel_map,
            qutip.parallel_map,
            krotov.parallelization.parallel_map_fw_prop_step,
        )
    )
    print(opt_result.message)
    opt_result.dump(os.path.join(os.getcwd(),'results','latest.dat'.format(current_time(),T)))


# In[ ]:


while True:
    step_size = pulse_max(0)*0.5
    λ = 1/step_size
    #ϕ = [[ tensor(basis(L,0), basis(N,0)), tensor((basis(L,0)+basis(L,1)).unit(), basis(N,0)) ]]

    coeffs = [(1,0), (1,-1), (1,1j), (1,1), (1,-1j), (0,1)]
    ϕ = states(coeffs)

    #existing_times = [float(file.split('_')[4][:-4]) for file in os.listdir('results')]
    #t_times = np.flip(np.arange(1,21.5,1))
    t_times = [20.]
    for tot in t_times:
        #if tot not in [float(file.split('_')[4][:-4]) for file in os.listdir('results')]:
            #plot_cardinal(state_rot(ϕ, tot)[0])
        #    if tot.is_integer():
        #        tot = int(tot)
        run_optim(tot, λ, ϕ)
        #else:
        #    print('T = {} already exists'.format(tot))


# ## Simulate dynamics of the optimized field
# 
# Having obtained the optimized
# control field, we can now
# plot it and calculate the
# population dynamics under
# this field.

# In[ ]:


def plot_results_3d(results):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Iteration')
    ax.set_zlabel('Fidelity')
    ax.set_ylabel('Pulse length')
    ax.set_zlim(0,1.1)
    for (r, T) in results:
        plot_fid_convergence(ax, r.info_vals[1:], T)
    ax.view_init(elev=20, azim=340)
    return (fig, ax)

def plot_results_iteration(results):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity')
    for (r, T) in results:
        ax.plot(range(0,len(r.info_vals)-1), r.info_vals[1:])
    #print('F = {}'.format(r.info_vals[-1]))
    return (fig, ax)
    
def plot_results_pulse_length(results, iteration=-1, ax=None):
    if ax is None:
        ax = plt.axes()
    else:
        ax.clear()
    ax.set_xlabel('Pulse length')
    ax.set_ylabel('Fidelity')
    for (r, T) in results:
        it = min(len(r.info_vals)-1,iteration)
        ax.plot(T, r.info_vals[it], 'ok')
    ax.set_ylim(0,1.1)
    return ax


# In[ ]:


folder = 'best_results_resonator'
results = [(krotov.result.Result.load(os.path.join(os.getcwd(),folder,file), objectives=get_objectives(T=float(file.split('_')[-1][:-4]))), float(file.split('_')[-1][:-4])) for file in os.listdir(folder) if file[-4:]=='.dat']
get_ipython().run_line_magic('matplotlib', 'inline')
plot_results_3d(results)

#plot_results_pulse_length(results, iteration=0)

plot_results_iteration(results)


# In[ ]:


plot_results_pulse_length(results, iteration=20, ax=ax)
ax = plt.axes()
def interactive_plot(iteration):
    plot_results_pulse_length(results, iteration=iteration, ax=ax)
interact(interactive_plot, iteration=widgets.IntSlider(min=0,max=900,step=1,value=0));


# In[ ]:


import matplotlib2tikz
matplotlib2tikz.save("mytikz.tex")


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Analyze

# In[ ]:


coeffs = [(1,0), (1,-1), (1,1j), (1,1), (1,-1j), (0,1)]
#coeffs = [(1,np.exp(1j*π/2))]
ϕ = states(coeffs)


# In[ ]:


# Move the file latest.dat in results to cat-code with the suffix *_T.dat e.g. cat_60.dat
folder = 'cat-code'
results = [(krotov.result.Result.load(os.path.join(os.getcwd(),folder,file), objectives=get_objectives(T=float(file.split('_')[-1][:-4]))), float(file.split('_')[-1][:-4])) for file in os.listdir(folder) if file[-4:]=='.dat']
results = [results[-1]]
print(results[-1][1])


# In[ ]:


print(results[0][0].info_vals[-1])

#hours = sum(results[0][0].iter_seconds)/(60*60)
#f"{int(np.floor(hours))}:{int(np.floor((60*(hours % np.floor(hours)))))}"


# In[ ]:


r = results[-1][0]
T = results[-1][1]
steps2 = len(results[0][0].tlist)*1000
tlist = r.tlist

c = r.optimized_controls
tlist2 = np.linspace(0, tlist[-1], steps2)
Ω_q = c[0]+1j*c[1]
Ω_q = np.interp(tlist2, tlist, Ω_q)
Ω_r = c[2]+1j*c[3]
Ω_r = np.interp(tlist2, tlist, Ω_r)

pulses_lab = [np.conj(Ω_q)*np.exp(1j*ω_q*tlist2), np.conj(Ω_q)*np.exp(-1j*ω_q*tlist2), np.conj(Ω_r)*np.exp(1j*ω_r*tlist2), np.conj(Ω_r)*np.exp(-1j*ω_r*tlist2)]
#opt_dynamics = [ob.propagate(tlist, propagator=krotov.propagators.expm) for ob in r.optimized_objectives]
print('Done')


# In[ ]:


occs = []
#plot_evolution(opt_dynamics)
#for i in range(len(opt_dynamics)):
#fig = plot_pulse(Ω_q, tlist)
#fig.axes[0].set_ylabel('$\Omega_q$')
#matplotlib2tikz.save("../Figures/Results/cat_pulse_shape_{}_q.tikz".format(str(T).replace('.',',')),
#                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')

#fig = plot_pulse(c[1], tlist)
#fig.axes[0].set_ylabel('Im($\Omega_q$)')
#matplotlib2tikz.save("../Figures/Results/cat_pulse_shape_{}_q_Imag.tikz".format(str(T).replace('.',',')),
#                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')

#fig = plot_pulse(Ω_r, tlist)
#fig.axes[0].set_ylabel('$\Omega_r$')
#matplotlib2tikz.save("../Figures/Results/cat_pulse_shape_{}_r.tikz".format(str(T).replace('.',',')),
#                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')

#fig = plot_pulse(c[3], tlist)
#fig.axes[0].set_ylabel('Im($\Omega_r$)')
#matplotlib2tikz.save("../Figures/Results/cat_pulse_shape_{}_r_Imag.tikz".format(str(T).replace('.',',')),
#                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')

    #fig = qubit_occupation(opt_dynamics[i])
    #matplotlib2tikz.save(f"../Figures/Results/cat_qubit_occ_{str(T).replace('.',',')}_{i}.tikz",
    #                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')

    #n = resonator_occupation(opt_dynamics[i],export=False);
    #occs.append([max(n[i]) for i in range(8)])
    #matplotlib2tikz.save(f"../Figures/Results/cat_res_occ_{str(T).replace('.',',')}_{i}.tikz",
    #                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')
    
power_two = 2**25 # Pad pulse with power_two zeros to increase frequency spectrum resolution
fig = plot_spectrum(pulses_lab[0], tlist2, mark_freq=[ω_q],mark_color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c'], pos=0, xlim=[(2*π)*5.5, (2*π)*6.9])
fig.axes[0].set_title('')
fig.axes[0].legend(['Spectrum',r'$\omega_{q}-\kappa_q$',r'$\omega_{q}$',r'$\omega_{q}+\kappa_q$'])
#matplotlib2tikz.save("../Figures/Results/cat_pulse_spectrum_qubit_{}_zoom.tikz".format(str(T).replace('.',',')),
#                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')

#fig = plot_spectrum(pulses_lab[0], tlist2, mark_freq=[ω_q, 2*ω_r-ω_q, ω_r, ω_r/2, ω_r/4],mark_color=[u'#ff7f0e',u'#1f77b4','k','k','k','k'], pos=0, xlim=[0*(2*π)*5.5, (2*π)*12])
#fig.axes[0].set_title('')
#fig.axes[0].legend(['Spectrum',r'$\omega_{q}$',r'$2\omega_r-\omega_{q}$',r'$n\omega_r$'])
#matplotlib2tikz.save("../Figures/Results/cat_pulse_spectrum_qubit_{}.tikz".format(str(T).replace('.',',')),
#                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')

#fig = plot_spectrum(pulses_lab[2], tlist2, mark_freq=[ω_r],mark_color=[u'#1f77b4', u'#ff7f0e', u'#2ca02c'], pos=0, xlim=[(2*π)*7.6, (2*π)*8.9])
#fig.axes[0].set_title('')
#fig.axes[0].legend(['Spectrum',r'$\omega_{r}$',r'$\omega_{r}-\kappa_q$',r'$\omega_{r}+\kappa_q$'])
#matplotlib2tikz.save("../Figures/Results/cat_pulse_spectrum_res_{}_zoom.tikz".format(str(T).replace('.',',')),
#                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')

#fig = plot_spectrum(pulses_lab[2], tlist2, mark_freq=[ω_r, ω_q, 2*ω_r-ω_q],mark_color=[u'#ff7f0e', u'#2ca02c',u'#1f77b4', u'#ff7f0e', u'#2ca02c'], pos=0, xlim=[0*(2*π)*7.6, (2*π)*12])
#fig.axes[0].set_title('')
#fig.axes[0].legend(['Spectrum',r'$\omega_{r}$',r'$\omega_{q}$',r'$2\omega_{r}-\omega_q$'])
#matplotlib2tikz.save("../Figures/Results/cat_pulse_spectrum_res_{}.tikz".format(str(T).replace('.',',')),
#                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')

#    start_state = final_state = opt_dynamics[i].states[0]
#    start_state_q = final_state.ptrace(0)
#    final_state = opt_dynamics[i].states[-1]
#    final_state_q = final_state.ptrace(0)
#    final_state_r = final_state.ptrace(1)
#    target_state = r.objectives[i].target
#    target_state_q = r.objectives[i].target.ptrace(0)
#    target_state_r = r.objectives[i].target.ptrace(1)
#    print((final_state.dag()*target_state)[0][0])
#    qutip.visualization.hinton(final_state*final_state.dag())
    #fig = plot_cardinal([start_state_q,target_state_q,final_state_q])
    #fig.save(name="../Figures/Results/cat_bloch_sphere_{}.png".format(str(T).replace('.',',')))

    #fig, ax = qutip.visualization.plot_wigner(final_state_r, figsize=(4.5,4.5), alpha_max=5)
    #ax.set_title('')
    #fig.savefig(fname=f"../Figures/Results/cat_wigner_{str(T).replace('.',',')}_{i}.png")

    #fig, ax = qutip.visualization.plot_wigner_fock_distribution(final_state_r)
    #fig, ax = qutip.visualization.plot_wigner_fock_distribution(target_state_r)
    #fig.savefig(fname="../Figures/Results/cat_wigner_tar_{}.png".format(str(T).replace('.',',')))

#subprocess.call("../Figures/Results/move_files.sh", shell=False)


# # Check pulse solution for states around the Bloch Sphere

# In[ ]:


def get_new_objectives(T=None, phi=states(coeffs)):
    if use_rotating:
        objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in state_rot(phi, T)]
    else:
        objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in ϕ]
    return objectives


# In[ ]:


def fid_from_state(coeffs):
    coeffs = [coeffs]
    folder = 'cat-code'
    phi = states(coeffs)
    results = [(krotov.result.Result.load(os.path.join(os.getcwd(),folder,file), objectives=get_new_objectives(T=float(file.split('_')[-1][:-4]),phi=phi)), float(file.split('_')[-1][:-4])) for file in os.listdir(folder) if file[-4:]=='.dat']
    results = [results[-1]]
    r = results[-1][0]
    T = results[-1][1]
    tlist = r.tlist
    opt_dynamics = [ob.propagate(tlist, propagator=krotov.propagators.expm) for ob in r.optimized_objectives]
    final_state = opt_dynamics[0].states[-1]
    target_state = r.objectives[0].target
    #qutip.visualization.plot_wigner(target_state.ptrace(1))
    print([coeffs[0][0], coeffs[0][1], fid(final_state, target_state)])


# In[ ]:


bloch_space = []
for theta in np.linspace(0,π,40):
    for phi in np.linspace(0,2*π,20):
        coef = (np.cos(theta)  ,  np.exp(1j*phi)*np.sin(theta))
        bloch_space.append(coef)
bloch_space = bloch_space[19:-19]
bloch_space


# In[ ]:


bl = qutip.Bloch()

bl.point_marker = 'o'
bl.point_color = 'r'
bl.add_states(qutip.basis(2,0), kind='point')
bl.add_states(qutip.basis(2,1), kind='point')
bl.show()


# In[ ]:


import multiprocessing
state_list = bloch_space
pool = multiprocessing.Pool(processes=4)
pool.map(fid_from_state, state_list)
pool.close()
pool.join()
print('DONE')


# In[ ]:


#result_list = [eval(res) for res in result_list.split('\n')[1:-1]]
#result_list = result_list[1:-1]
#fid_list = [res[2] for res in result_list]
for c in [(0,np.exp(1j*0))]:
    fid_from_state(c)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
bl = qutip.Bloch()
bl.point_marker = 'o'
colors = [[(res[2]-fid_min)/(fid_max-fid_min),0 , (fid_max-res[2])/(fid_max-fid_min)] for res in result_list]
bl.point_color = colors
for res in result_list:
    bl.add_states((states([(res[0],res[1])])[0][0]).ptrace(0), kind='point')
bl.view = (-90,90)
bl.show()


# In[ ]:


#color_list = [[i/len(opt_dynamics[0].states),i/len(opt_dynamics[0].states),i/len(opt_dynamics[0].states)] for i in range(len(opt_dynamics[0].states))]
color_list = linear_gradient('#FFF0F0','#0000FF', n = len(opt_dynamics[0].states))
#bl = plot_evolution(opt_dynamics[1],steps=1, color_list=color_list)
plot_cardinal([opt_dynamics[2].states[0]])


# In[ ]:


qutip.operators.destroy(2)*[basis(2,0),basis(2,1)]


# In[ ]:


def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
        curr_vector = [
          int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
          for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return np.array(RGB_list)/255


# # Other plots and gif generation

# In[ ]:


occs2


# In[ ]:


#occs2 = [[occ[i] for occ in occs] for i in range(8)]
#print(occs2)
fig, ax= plt.subplots()
ax.plot([f'$|{i}\\rangle$' for i in range(8)],occs2,'.')
ax.set_xlabel('Resonator fock state')
ax.set_ylabel('Maximum occupation')
#ax.legend(['\(\ket{0}\)','\(\ket{1}\)','\((\ket{0}+\ket{1})/\sqrt{2}\)','\((\ket{0}-\ket{1})/\sqrt{2}\)','\((\ket{0}+i\ket{1})/\sqrt{2}\)','\((\ket{0}-i\ket{1})/\sqrt{2}\)'])
#matplotlib2tikz.save("../Figures/Results/cat_max_occ.tikz",
#                     figureheight = '\\figureheight',figurewidth = '\\figurewidth')
#subprocess.call("../Figures/Results/move_files.sh", shell=False)


# In[ ]:


def gif_fun(start, opt):
    import matplotlib.pyplot as plt
    import qutip
    fig, ax = plt.subplots(figsize=(16,4))
    for (state, i) in zip(opt.states, range(len(opt.states))):
        if f'{str(i).zfill(4)}.png' not in os.listdir('gif'):
            qutip.visualization.plot_fock_distribution(state.ptrace(1), fig=fig, ax=ax);
            fig.savefig(fname=f'fock_gif/{str(i).zfill(4)}.png');
            ax.clear();


# In[ ]:


def gif_fun(start, opt_dynamics):
    import matplotlib.pyplot as plt
    import qutip
    fig, ax = plt.subplots(figsize=(16,4))
    L=8
    tlist = opt_dynamics.times
    occ = [qutip.basis(L,i)*qutip.basis(L,i).dag() for i in range(0,L)]
    states = [state.ptrace(1) for state in opt_dynamics.states]
    n = qutip.expect(occ, states)
    for i in range(len(n)):
        ax.plot(tlist, n[i])
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Occupation')
    time_line = ax.axvline(tlist[1919], color='k',linestyle='--', linewidth=1)
    occ_points = [ax.scatter(tlist[1919], n[p][0], s=25,edgecolors='k',linewidth=1) for p in range(len(n))]
    for i in range(len(tlist)):
        time_line.set_xdata([tlist[i],tlist[i]])
        for p in range(len(n)):
            occ_points[p].set_offsets([tlist[i],n[p][i]])
        fig.savefig(fname=f'res_occupation_gif/{str(i).zfill(4)}.png');
#qutip.visualization.plot_wigner_fock_distribution(opt_dynamics[0].states[1919].ptrace(1), fig=fig, axes=ax)


# In[ ]:


pulses


# In[ ]:


fig, ax = plt.subplots(figsize=(16,4));
fig, ax = plot_results_iteration(results, fig=fig);
time_line = ax.axvline(0, color='k',linestyle='--', linewidth=1)
point = ax.scatter(0,0, s=25,edgecolors='k',linewidth=1)
for i in range(911):
    point.set_offsets([range(911)[i],results[-1][0].info_vals[i]])
    time_line.set_xdata([i,i])
    fig.savefig(fname=f'cat_fid_gif/{str(i).zfill(3)}.png');


# In[ ]:


fig, ax = plt.subplots(figsize=(16,4));
for i in range(len(results[-1][0].all_pulses)):
    plot_pulse((results[-1][0].all_pulses[i][2][:1919]+1j*results[-1][0].all_pulses[i][3][:1919]),results[-1][0].tlist[:1919], fig=fig);
    fig.savefig(fname=f'cat_pulse_res_optim_gif/{str(i).zfill(4)}.png');
    ax.clear();


# In[ ]:


import multiprocessing  

pool = multiprocessing.Pool(processes=4)
pool.starmap(gif_fun, [(0, opt_dynamics[0]),(480, opt_dynamics[0]),(2*480, opt_dynamics[0]),(3*480, opt_dynamics[0])])
pool.close()
pool.join()   
print('done')


# In[ ]:


xlabels = ['$|0\\rangle$','$|1\\rangle$','$|2\\rangle$']
ylabels = ['$\\langle 0|$','$\\langle 1|$','$\\langle 2|$']
final_state = results[-1][0].states[0]
final_state = opt_dynamics[0].states[-1]
target_state = results[0][0].objectives[0].target
#plot_matrix_final_target(target_state, final_state, xlabels, ylabels, el=45, az=150)
#plot_matrix_final_target(target_state, final_state, xlabels, ylabels, el=10, az=150)
plot_cardinal([target_state, final_state])
#plot_evolution(opt_dynamics)
qutip.visualization.plot_wigner_fock_distribution(final_state.ptrace(1))
qutip.visualization.plot_wigner_fock_distribution(target_state.ptrace(1))

print(fid(final_state, target_state))
qutip.visualization.hinton(final_state.ptrace(0)*final_state.ptrace(0).dag())


# In[ ]:


np.abs((ω_q - ω_r)/(2*π))


# In[ ]:


plot_spectrum(pulses_lab[0], tlist2, mark_freq=[ω_q, ω_ef, ω_gf, ω_r, ω_r+K_r/2, ω_r+K_r],mark_color=['r','g','b','r','g','b'], pos=0, xlim=[2*π*5, 2*π*9]);
plot_spectrum(pulses_lab[2], tlist2, mark_freq=[ω_r, ω_r+K_r/2, ω_r+K_r , 2*ω_r - ω_q, 2*ω_r -ω_ef, 2*ω_r -ω_gf,],mark_color=['r','g','b','r','g','b'], pos=0, xlim=[ω_r*0.90, ω_r*1.4]);


# In[ ]:


from matplotlib.font_manager import findfont, FontProperties
font = findfont(FontProperties(family=['sans-serif']))
font


# # Simulate dynamics in lab frame (old and unused)

# In[ ]:


#rotating_pulses = [np.vectorize(H_i[1])(tlist, None) for H_i in H[1:]]
rotating_pulses = oct_result.optimized_controls

tlist2 = np.linspace(0, T, int(np.ceil(200*T/T_q)))
Ω = rotating_pulses[0]+1j*rotating_pulses[1]
Ω = np.interp(tlist2, tlist, Ω)
pulses_lab = [Ω*np.exp(1j*ω_q*tlist2), np.conj(Ω)*np.exp(-1j*ω_q*tlist2)]
H_lab = hamiltonian(ω=1.0, ampl0=1, use_rotating=False, pulses=pulses_lab)
objectives_lab = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H_lab) for ψ in ϕ]
opt_dynamics_lab = [ob.mesolve(tlist2, progress_bar=True, options=qutip.Options(nsteps=50000)) for ob in objectives_lab]

tlist2 = np.linspace(0, T, 1000*int(np.ceil(T)))
rotating_pulses = [np.array([H_i[1](t,0) for t in tlist]) for H_i in H[1:]]
Ω_q = rotating_pulses[0]+1j*rotating_pulses[1]
Ω_q = np.interp(tlist2, tlist, Ω_q)
Ω_r = rotating_pulses[2]+1j*rotating_pulses[3]
Ω_r = np.interp(tlist2, tlist, Ω_r)
pulses_lab = [Ω_q*np.exp(1j*ω_q*tlist2), np.conj(Ω_q)*np.exp(-1j*ω_q*tlist2), Ω_r*np.exp(1j*ω_r*tlist2), np.conj(Ω_r)*np.exp(-1j*ω_r*tlist2)]
H_lab = hamiltonian(ampl0=1, use_rotating=False, pulses=pulses_lab)
objectives_lab = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H_lab) for ψ in ϕ]
opt_dynamics_lab = [ob.mesolve(tlist2, progress_bar=True, options=qutip.Options(nsteps=50000)) for ob in r.optimized_objectives]


# In[ ]:


plot_pulse(Ω, tlist2)
[plot_pulse(p, tlist2) for p in pulses_lab];


# In[ ]:


qubit_occupation(opt_dynamics_lab[0])
#plot_evolution(opt_dynamics_lab)


# In[ ]:


plot_cardinal([opt_dynamics_lab[0].states[0]])
plot_cardinal([opt_dynamics_lab[0].states[-1]])


# In[ ]:


qubit_pulses = pulses_lab
time_list = tlist2


# In[ ]:


qubit_pulses = opt_result.optimized_controls
time_list = tlist


# In[ ]:


plot_spectrum(qubit_pulses[0], time_list, mark_freq=[ω_q, ω_q + K_q, ω_q - K_q], pos=1)
plot_spectrum(qubit_pulses[1], time_list, mark_freq=[ω_q, ω_q + K_q, ω_q - K_q], pos=-1)

