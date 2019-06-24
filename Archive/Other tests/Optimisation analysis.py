
# coding: utf-8

# In[289]:


import numpy as np
import matplotlib.pyplot as plt
import krotov
import qutip
import scipy
import os
import copy
π = np.pi
sqrt = np.sqrt
basis = qutip.basis
tensor = qutip.tensor
coherent = qutip.coherent


# In[290]:


def plot_fid_convergence(ax, info_vals, T):
    ax.plot3D(range(0,len(info_vals)), [T]*len(info_vals), info_vals)


# In[291]:


L = 3
def proj(ψ, ϕ=None):
    if ϕ is None:
        return ψ * ψ.dag()
    else:
        return ψ * ϕ.dag()
Si = qutip.operators.identity(L)
ZERO = qutip.operators.qzero(L)

σ_z = proj(qutip.basis(L, 0)) - proj(qutip.basis(L, 1))
σ_y = 1j*(proj(qutip.basis(L, 1),qutip.basis(L, 0)) - proj(qutip.basis(L, 0), qutip.basis(L, 1)))
σ_x = proj(qutip.basis(L, 0),qutip.basis(L, 1)) - proj(qutip.basis(L, 1), qutip.basis(L, 0))
b = qutip.operators.destroy(L)
I = Si

ω_r = 8.3056 * 2 * π      # resonator frequency
ω_q = 6.2815 * 2 * π      # qubit frequency
K_q   = -2*π*297e-3    # Kerr qubit 200-300 MHz

ω_ef = ω_q + K_q
ω_gf = ω_q + K_q/2
use_rotating = True
def hamiltonian(ω=1.0, ampl0=1, use_rotating=True, pulses=None, tlist=None):
    """Two-level-system Hamiltonian
    
    Args:
        ω (float): energy separation of the qubit levels
        ampl0 (float): constant amplitude of the driving field
    """
    
    K_r   = 2*π*0.45e-3   # Kerr res
    #K_q   = -2*π*297e-3    # Kerr qubit 200-300 MHz
    ω_r = 8.3056 * 2 * π      # resonator frequency
    ω_q = 6.2815 * 2 * π      # qubit frequency
    χ = 0.025 * 2 * π   # parameter in the dispersive hamiltonian

    Δ = abs(ω_r - ω_q)    # detuning
    g = sqrt(Δ * χ)  # coupling strength that is consistent with chi
    γ = 1e1
    
    #H_occ = w_r*a.dag()*a + w_q*b.dag()*b
    if L==3:
        H_occ_q = qutip.Qobj(np.diag([0, ω_q, 2*ω_q]))
    else:
        H_occ_q = qutip.Qobj(np.diag([0, ω_q]))
    #H_occ_r = ω_r * a.dag()*a
    H_occ =  H_occ_q#  + H_occ_r
    
    
    use_dispersive = True
    use_kerr = True
    #if use_dispersive:
    #    #H_coup = - chi_qr * a.dag()*a * b.dag()*b
    #    H_coup =  χ * (a.dag()*a + I/2) * σ_z
    #else:
        #H_coup = g * (a.dag() * b + a * b.dag())
    #    H_coup = g * σ_x *a.dag() + a
    if use_kerr:
        H_kerr = - K_q/2 * b.dag()**2 * b**2
    else:
        H_kerr = ZERO
    
    H_d = ZERO
    
    if use_rotating:
        H_d += H_kerr
        
        H_qr = (b.dag() + b)
        H_qi = 1j*(b.dag() - b)
        #H_rr = (a + a.dag())
        #H_ri = 1j*(a.dag() - a)
        
        
        ϵ_qr = lambda t, args: ampl0
        ϵ_qi = lambda t, args: ampl0
        #ϵ_rr = lambda t, args: ampl0
        #ϵ_ri = lambda t, args: ampl0
        
        # Random pulses (doesn't really work)
        #ϵ = lambda t, tlist, R: R[np.where(tlist<=t)[0][-1]]
        #O = np.random.rand(len(tlist))
        #ϵ_qr = lambda t, args: ϵ(t, tlist, O)
        #O = np.random.rand(len(tlist))
        #ϵ_qi = lambda t, args: ϵ(t, tlist, O)
        
        
        if pulses:
            ϵ_qr = pulses[0]
            ϵ_qi = pulses[1]
        #    ϵ_rr = np.zeros(len(pulses[0]))
        #    ϵ_ri = np.zeros(len(pulses[0]))

        return [H_d, [H_qr, ϵ_qr], [H_qi, ϵ_qi]]#, [H_rr, ϵ_rr], [H_ri, ϵ_ri]]
    else:
        H_d += H_occ + H_kerr#+ H_coup
        
        H_q = b
        H_qc = b.dag()
        #H_rr = ZERO
        #H_ri = ZERO
        

        ϵ_q = lambda t, args: 1j*ampl0*np.exp(1j*ω_q*t)
        ϵ_qc = lambda t, args: -1j*ampl0*np.exp(-1j*ω_q*t)
        #ϵ_rr = lambda t, args: ampl0
        #ϵ_ri = lambda t, args: ampl0
        
        if pulses:
            ϵ_q = pulses[0]
            ϵ_qc = pulses[1]
            #ϵ_rr = np.zeros(len(pulses[0]))
            #ϵ_ri = np.zeros(len(pulses[0]))
        
        return [H_d, [H_q, ϵ_q], [H_qc, ϵ_qc]]#, [H_rr, ϵ_rr], [H_ri, ϵ_ri]]


# In[347]:


def state_rot(ϕ, T):
    ϕ = copy.deepcopy(ϕ)
    if L == 3:
        rot_evo = qutip.Qobj([[1, 0, 0],[0, np.exp(-1j * ω_q * T), 0],[0, 0, 0]])
    else:
        rot_evo = qutip.Qobj([[1, 0],[0, np.exp(-1j * ω_q * T)]])
    ϕ[0][1] = rot_evo * ϕ[0][1]
    return ϕ

H = hamiltonian(ampl0=1, use_rotating=True)
ϕ = [[ basis(L,0), (basis(L,0)-basis(L,1)).unit() ]]
ϕ = [[ basis(L,0), basis(L,1) ]]

def get_objectives(T=None):
    if use_rotating:
        objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in state_rot(ϕ, T)]
    else:
        objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in ϕ]
    return objectives


# In[348]:


def plot_population(n, tlist):
    fig, ax = plt.subplots(figsize=(15,4))
    leg = []
    for i in range(len(n)):
        ax.plot(tlist, n[i], label=str(i))
        leg.append(str(i))
    ax.legend()
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Occupation')
    ax.legend(leg)
    plt.show(fig)
def qubit_occupation(dyn):
    occ = [basis(L,i)*basis(L,i).dag() for i in range(0,L)]
    n = qutip.expect(occ, dyn.states)
    plot_population(n, dyn.times)
def plot_pulse(pulse, tlist):
    fig, ax = plt.subplots(figsize=(15,4))
    if callable(pulse):
        pulse = np.array([pulse(t, args=None) for t in tlist])
    if np.any(np.iscomplex(pulse)):
        ax.plot(tlist, np.real(pulse))
        ax.plot(tlist, np.imag(pulse))
        ax.legend(['Re', 'Im'])
    else:
        ax.plot(tlist, pulse)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Pulse amplitude')
    plt.show(fig)
def plot_spectrum(pulse, tlist, mark_freq=None, pos=1, xlim=None, mark_color=['k','k','k']):
    samples = len(tlist)
    sample_interval = tlist[-1]/samples
    time = np.linspace(0, samples*sample_interval, samples)

    signal_qubit = pulse
    signal_spectrum = np.fft.fftshift(np.fft.fft(signal_qubit))
    freqs = np.fft.fftshift(np.fft.fftfreq(samples, d=sample_interval))

    plt.figure(figsize=(10,5))
    plt.plot(freqs, np.abs(signal_spectrum))  # in GHz
    if mark_freq is not None:
        if not isinstance(mark_freq, list):
            mark_freq = [mark_freq]
        mf = np.array(mark_freq)/(2*π)
        if pos==1:
            plt.xlim(0, 2*mf[0])
        elif pos==-1:
            plt.xlim(-2*mf[0], 0)
        elif xlim is not None:
            plt.xlim(xlim[0]/(2*π), xlim[1]/(2*π))
        [plt.axvline(x=m_f, ymin=0, ymax=1, color=col, linestyle='--', linewidth=1) for (m_f, col) in zip(mf,mark_color)]
    plt.title('Qubit pulse spectrum')
    plt.xlabel('f (GHz)');
    
def plot_cardinal(ψ):
    bl = qutip.Bloch()
    bl.vector_color = ['r','g','b','g','b','r']
    [bl.add_states(to_two_level(ϕ), 'vector') for ϕ in ψ]
    bl.show()
def to_two_level(state):
    if state.type is 'oper':
        return qutip.Qobj(state[0:2,0:2])
    else:
        return qutip.Qobj(state[0:2])
def plot_evolution(dyn, steps=1):
    for d in dyn:
        points = [to_two_level(s) for s in d.states[0:-1:steps]]
        bl = qutip.Bloch()
        bl.vector_color = 'r'
        bl.point_color = 'r'
        bl.point_marker = 'o'
        bl.add_states(points, 'point')
        bl.show()
        bl = qutip.Bloch()
        bl.vector_color = 'r'
        bl.point_color = 'r'
        bl.point_marker = 'o'
        bl.view = [bl.view[0], 80]
        bl.add_states(points, 'point')
        bl.show()
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


# In[353]:


results = [(krotov.result.Result.load(os.path.join(os.getcwd(),'results',file), objectives=get_objectives(T=float(file.split('_')[-1][:-4]))), float(file.split('_')[-1][:-4])) for file in [os.listdir('results')[-1]] if file[-4:]=='.dat']

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Iteration')
ax.set_zlabel('Fidelity')
ax.set_ylabel('Pulse length')
for (r, T) in results:
    plot_fid_convergence(ax, r.info_vals, T)

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('Iteration')
ax.set_ylabel('Fidelity')
for (r, T) in results:
    ax.plot(range(0,len(r.info_vals)), r.info_vals)
    print('F = {}'.format(r.info_vals[-1]))


# In[354]:


T_q = (2*π)/ω_q
steps2 = len(results[0][0].tlist)*1000
for (r,_) in results:
    tlist = r.tlist
    #opt_dynamics = [ob.mesolve(tlist, progress_bar=True) for ob in r.objectives]
    #qubit_occupation(opt_dynamics[0])
    
    c = r.optimized_controls
    tlist2 = np.linspace(0, tlist[-1], steps2)
    Ω = c[0]+1j*c[1]
    Ω = np.interp(tlist2, tlist, Ω)
    pulses_lab = [Ω*np.exp(1j*ω_q*tlist2), np.conj(Ω)*np.exp(-1j*ω_q*tlist2)]
    opt_dynamics = [ob.mesolve(tlist, progress_bar=True) for ob in r.optimized_objectives]
    plot_pulse(r.guess_controls[0], tlist)
    plot_pulse(r.guess_controls[1], tlist)
    plot_pulse(c[0], tlist)
    plot_pulse(c[1], tlist)
    plot_pulse(pulses_lab[0], tlist2)
    plot_pulse(pulses_lab[1], tlist2)
    qubit_occupation(opt_dynamics[0])
    plot_spectrum(pulses_lab[0], tlist2, mark_freq=[ω_q, ω_ef, ω_gf],mark_color=['r','g','b'], pos=0, xlim=[ω_q*0.9, ω_q*1.1])
    #plot_spectrum(pulses_lab[1], tlist2, mark_freq=[ω_q, ω_ef, ω_gf], pos=0, xlim=[-ω_q*0.95, -ω_q*1.05])
    #H_lab = hamiltonian(ampl0=1, use_rotating=False, pulses=pulses_lab)
    #objectives_lab = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H_lab) for ψ in ϕ]
    
    


# In[356]:


xlabels = ['$|0\\rangle$','$|1\\rangle$','$|2\\rangle$']
ylabels = ['$\\langle 0|$','$\\langle 1|$','$\\langle 2|$']
final_state = opt_dynamics[0].states[-1]
#target_state = get_objectives(tlist[-1])[0].target
target_state = results[0][0].objectives[0].target
plot_matrix_final_target(-target_state, final_state, xlabels, ylabels, el=45, az=150)
plot_matrix_final_target(-target_state, final_state, xlabels, ylabels, el=10, az=150)
plot_cardinal([target_state, final_state])
plot_evolution(opt_dynamics)

