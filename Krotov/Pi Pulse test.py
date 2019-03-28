
# coding: utf-8

# # Optimization of a State-to-State Transfer in a Two-Level-System

# In[596]:


# NBVAL_IGNORE_OUTPUT
get_ipython().run_line_magic('load_ext', 'watermark')
import qutip
import numpy as np
import scipy
import matplotlib
import matplotlib.pylab as plt
import krotov
get_ipython().run_line_magic('watermark', '-v --iversions')
π = np.pi
sqrt = np.sqrt
basis = qutip.basis
tensor = qutip.tensor
coherent = qutip.coherent


# $\newcommand{tr}[0]{\operatorname{tr}}
# \newcommand{diag}[0]{\operatorname{diag}}
# \newcommand{abs}[0]{\operatorname{abs}}
# \newcommand{pop}[0]{\operatorname{pop}}
# \newcommand{aux}[0]{\text{aux}}
# \newcommand{opt}[0]{\text{opt}}
# \newcommand{tgt}[0]{\text{tgt}}
# \newcommand{init}[0]{\text{init}}
# \newcommand{lab}[0]{\text{lab}}
# \newcommand{rwa}[0]{\text{rwa}}
# \newcommand{bra}[1]{\langle#1\vert}
# \newcommand{ket}[1]{\vert#1\rangle}
# \newcommand{Bra}[1]{\left\langle#1\right\vert}
# \newcommand{Ket}[1]{\left\vert#1\right\rangle}
# \newcommand{Braket}[2]{\left\langle #1\vphantom{#2} \mid
# #2\vphantom{#1}\right\rangle}
# \newcommand{op}[1]{\hat{#1}}
# \newcommand{Op}[1]{\hat{#1}}
# \newcommand{dd}[0]{\,\text{d}}
# \newcommand{Liouville}[0]{\mathcal{L}}
# \newcommand{DynMap}[0]{\mathcal{E}}
# \newcommand{identity}[0]{\mathbf{1}}
# \newcommand{Norm}[1]{\lVert#1\rVert}
# \newcommand{Abs}[1]{\left\vert#1\right\vert}
# \newcommand{avg}[1]{\langle#1\rangle}
# \newcommand{Avg}[1]{\left\langle#1\right\rangle}
# \newcommand{AbsSq}[1]{\left\vert#1\right\vert^2}
# \newcommand{Re}[0]{\operatorname{Re}}
# \newcommand{Im}[0]{\operatorname{Im}}$
# The purpose of this example is not to solve an especially interesting physical
# problem but to give a rather simple example of how the package can be used in
# order to solve an optimization problem.
# 
# ## Define the Hamiltonian
# 
# In the
# following the Hamiltonian, guess field and
# states are defined.
# 
# The Hamiltonian
# $\op{H}_{0} = - \omega \op{\sigma}_{z}$
# represents a
# simple qubit with energy
# level splitting $\omega$ in the basis
# $\{\ket{0},\ket{1}\}$. The control
# field
# $\epsilon(t)$ is assumed to couple via
# the
# Hamiltonian $\op{H}_{1}(t) =
# \epsilon(t) \op{\sigma}_{x}$ to the qubit,
# i.e., the control
# field effectively
# drives
# transitions between both qubit
# states. For now, we initialize the control
# field as constant.

# In[597]:


N = 12
α = 2


# # Plotting functions

# In[598]:


def plot_population(result):
    fig, ax = plt.subplots()
    ax.plot(result.times, result.expect[0], label='0')
    ax.plot(result.times, result.expect[1], label='1')
    ax.legend()
    ax.set_xlabel('time')
    ax.set_ylabel('population')
    plt.show(fig)

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

def plot_system(ψ):
    bl = qutip.Bloch()
    bl.add_states(ψ.ptrace(0))
    bl.show()
    qutip.visualization.plot_wigner_fock_distribution(ψ.ptrace(1))
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
    [bl.add_states(ϕ.ptrace(0), 'vector') for ϕ in ψ]
    bl.show()

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
def plot_evolution(dyn, steps=1):
    for d in dyn:
        points = [s.ptrace(0) for s in d.states[0:-1:steps]]
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


# In[599]:


def cat(N,α,P = 1):
    return (coherent(N,α) + P*coherent(N,-α)).unit()
def fid(result, target):
    return (np.abs((result.states[-1].dag()*target).full())**2)[0][0]


# In[778]:


Ri = qutip.operators.identity(N)
Si = qutip.operators.identity(2)
ZERO = tensor(qutip.operators.qzero(2), qutip.operators.qzero(N))

σ_z = qutip.operators.sigmaz()
σ_y = qutip.operators.sigmay()
σ_x = qutip.operators.sigmax()
a = qutip.operators.destroy(N)

σ_z = qutip.tensor(σ_z,Ri)
σ_y = qutip.tensor(σ_y,Ri)
σ_x = qutip.tensor(σ_x,Ri)
b = qutip.tensor(qutip.operators.destroy(2),Ri)
a = qutip.tensor(Si,a)
I = qutip.tensor(Si,Ri)

cat_0 = cat(N,α)
cat_1 = cat(N,α*1j)
ω_r = 8.3056 * 2 * π      # resonator frequency
ω_q = 6.2815 * 2 * π      # qubit frequency
use_rotating = True
def hamiltonian(ω=1.0, ampl0=1, use_rotating=True, pulses=None):
    """Two-level-system Hamiltonian
    
    Args:
        ω (float): energy separation of the qubit levels
        ampl0 (float): constant amplitude of the driving field
    """
    
    K_r   = 2*π*0.45e-3   # Kerr res
    K_q   = 2*π*297e-3    # Kerr qubit 200-300 MHz
    #K_r   = 1
    #K_q   = 1
    ω_r = 8.3056 * 2 * π      # resonator frequency
    ω_q = 6.2815 * 2 * π      # qubit frequency
    χ = 0.025 * 2 * π   # parameter in the dispersive hamiltonian

    Δ = abs(ω_r - ω_q)    # detuning
    g = sqrt(Δ * χ)  # coupling strength that is consistent with chi

    #H_occ = w_r*a.dag()*a + w_q*b.dag()*b
    H_occ = -ω_q/2.0 * σ_z  + ω_r* a.dag()*a
    use_dispersive = True
    use_kerr = False
    if use_dispersive:
        #H_coup = - chi_qr * a.dag()*a * b.dag()*b
        H_coup =  χ * (a.dag() * a + I/2) * σ_z
    else:
        #H_coup = g * (a.dag() * b + a * b.dag())
        H_coup = g * σ_x *(a.dag() + a)
    if use_kerr:
        H_kerr = - K_r/2 * a.dag()**2 * a**2 - K_q/2 * b.dag()**2 * b**2
    else:
        H_kerr = tensor(qutip.operators.qzero(2), qutip.operators.qzero(N))
    
    H_d = ZERO
    
    if use_rotating:
        H_qr = (b.dag() + b)
        H_qi = 1j*(b.dag() - b)
        H_rr = (a + a.dag())
        H_ri = 1j*(a.dag() - a)

        ϵ_qr = lambda t, args: ampl0*0
        ϵ_qi = lambda t, args: ampl0
        ϵ_rr = lambda t, args: ampl0*0
        ϵ_ri = lambda t, args: ampl0*0

        return [H_d, [H_qr, ϵ_qr], [H_qi, ϵ_qi], [H_rr, ϵ_rr], [H_ri, ϵ_ri]]
    else:
        H_d += H_occ #+ H_coup + H_kerr
        
        H_q = b
        H_qc = b.dag()
        H_rr = ZERO
        H_ri = ZERO
        
        if pulses:
            ϵ_q = pulses[0]
            ϵ_qc = pulses[1]
            ϵ_rr = np.zeros(steps)
            ϵ_ri = np.zeros(steps)
        else:
            ϵ_q = lambda t, args: 1j*ampl0*np.exp(1j*ω_q*t)
            ϵ_qc = lambda t, args: -1j*ampl0*np.exp(-1j*ω_q*t)
            ϵ_rr = lambda t, args: ampl0*0
            ϵ_ri = lambda t, args: ampl0*0
        return [H_d, [H_q, ϵ_q], [H_qc, ϵ_qc], [H_rr, ϵ_rr], [H_ri, ϵ_ri]]

def coeffs_to_state(c,init = True):
    if init:
        ψ = tensor((c[0]*basis(2,0) + c[1]*basis(2,1)).unit() , (basis(N,0)))
    else:
        ψ = tensor((basis(2,0)) , (c[0]*cat_0 + c[1]*cat_1).unit())
    return ψ

def states(coeffs):
    return [[coeffs_to_state(c,True),coeffs_to_state(c,False)] for c in coeffs]
H = hamiltonian(ampl0=1, use_rotating=True)
coeffs = [(1,0), (1,-1), (1,1j), (1,1), (1,-1j), (0,1)]
st = states(coeffs)
ϕ = [[ tensor(basis(2,0), coherent(N,α)), tensor((basis(2,0)-1j*basis(2,1)).unit(), coherent(N,α)) ]]


# The projectors $\op{P}_0 = \ket{0}\bra{0}$ and $\op{P}_1 = \ket{1}\bra{1}$ are
# introduced since they allow for calculating the
# population in the respective
# states later on.

# In[779]:


def proj(ψ):
    return ψ * ψ.dag()


# ## Define the optimization target

# First we define the time grid of the
# dynamics, i.e., by taking the following
# values as an example, we define the
# initial state to be at time $t=0$ and
# consider a total propagation time of
# $T=5$. The entire time grid is divided into
# $n_{t}=500$ equidistant time steps.

# In[780]:


T_r = (2*π)/ω_r
T_q = (2*π)/ω_q
T = 0.5 * T_q
T = π/2
steps = 500
tlist = np.linspace(0, T, steps)


# Next, we define the optimization targets, which is technically a list of
# objectives, but here it has just one entry defining a simple state-to-state
# transfer
# from initial state $\ket{\Psi_{\init}} = \ket{0}$ to the target state
# $\ket{\Psi_{\tgt}} = \ket{1}$, which we want to reach at final time $T$. Note
# that we also have to pass the Hamiltonian $\op{H}(t)$ that determines the
# dynamics of
# the system to the optimization objective.

# In[781]:


def state_rot(ϕ):
    rot_evo = tensor(qutip.Qobj([[1, 0],[0, np.exp(-1j * ω_q * T)]]), Ri)
    ϕ[0][1] = rot_evo * ϕ[0][1]
    return ϕ
if use_rotating:
    objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in state_rot(ϕ)]
else:
    objectives = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H) for ψ in ϕ]


# In addition, we have to define and assign a shape function $S(t)$ for the update
# in each control iteration to each
# control field that will be updated. This shape
# usually takes care of
# experimental limits such as the necessity of finite ramps
# at the beginning and
# end of the control field or other conceivable limitations
# for field shapes: wherever $S(t)$ is zero, the optimization will not change the
# value of the control from the original guess.

# In[782]:


def S(t):
    """Shape function for the field update"""
    #return 1
    #return krotov.shapes.blackman(t, t_start=0, t_stop=T)
    return krotov.shapes.flattop(t, t_start=0, t_stop=T, t_rise=0.1*T, t_fall=0.1*T, func='sinsq')
def S2(t):
    return 0


# At this point, we also change the initial control field $\epsilon_{0}(t)$ from a
# constant to a shaped pulse that switches on smoothly from zero and again
# switches off at the final time $T$. We re-use the shape function $S(t)$ that we
# defined for the updates for this purpose (although generally, $S(t)$ for the
# updates has nothing to with the shape of the control field).

# In[783]:


def shape_field(ϵ):
    """Applies the shape function S(t) to the guess field"""
    ϵ_shaped = lambda t, args: ϵ(t, args)*S(t)
    return ϵ_shaped

for H_i in H[1:]:
    H_i[1] = shape_field(H_i[1])


# Having defined the shape function $S(t)$ and having shaped the guess field, we
# now tell the optimization to also use $S(t)$ as the update-shape for
# $\epsilon_0(t)$. In addition, we have to choose `lambda_a` for each control
# field. It controls the update magnitude of the respective field in each
# iteration.

# In[784]:


opt_lambda = [.5, .5, 0, 0]
S_funs = [S,S,S2,S2]
pulse_options = {H_i[1]: dict(lambda_a=opt_lambda[0], shape=S_funs[i]) for i, H_i in enumerate(H[1:])}


# It is convenient to introduce the function `print_fidelity`, which can be passed
# to the optimization procedure and will be called after each iteration and thus
# provides additional feedback about the optimization progress.

# In[785]:


def print_fidelity(**args):
    F_re = np.average(np.array(args['tau_vals']).real)
    print("    F = {}".format(F_re))
    #return F_re


# ## Simulate dynamics of the guess field
# 
# Before heading towards the optimization
# procedure, we first simulate the
# dynamics under the guess field
# $\epsilon_{0}(t)$.

# The following plot shows the guess field $\epsilon_{0}(t)$, which is, as chosen
# above, just a constant field (with a smooth switch-on and switch-off)

# In[786]:


for H_i in H[1:]:
    plot_pulse(H_i[1], tlist)


# The next line solves the equation of motion for the defined objective, which
# contains the initial state $\ket{\Psi_{\init}}$ and the Hamiltonian $\op{H}(t)$
# defining its evolution.

# In[787]:


guess_dynamics = [ob.mesolve(tlist, progress_bar=True, options=qutip.Options(nsteps=50000)) for ob in objectives]


# The plot of the population dynamics shows that the guess field does not transfer
# the initial state $\ket{\Psi_{\init}} = \ket{0}$ to the desired target state
# $\ket{\Psi_{\tgt}} = \ket{1}$.

# In[788]:


#plot_all(guess_dynamics, ϕ)
plot_evolution(guess_dynamics, steps=1)


# In[789]:


#plot_all(guess_dynamics, ϕ)


# ## Optimize
# 
# In the following we optimize the guess field $\epsilon_{0}(t)$ such
# that the intended state-to-state transfer $\ket{\Psi_{\init}} \rightarrow
# \ket{\Psi_{\tgt}}$ is solved.
# 
# The cell below carries out the optimization. It
# requires, besides the
# previously
# defined optimization `objectives`, information
# about the
# optimization functional
# $F$ (via `chi_constructor`) and the
# propagation method that should be used. In
# addition, the number of total
# iterations is required and, as an option, we pass
# an info-hook that after each
# iteration combines a complete printout of the state
# of the optimization with the
# `print_fidelity` function defined above.
# 
# Here, we
# choose $F = F_{re}$ with
# \begin{equation}
# F_{re}
# =
# \Re\Braket{\Psi(T)}{\Psi_{\tgt}}
# \end{equation}
# 
# with
# $\ket{\Psi(T)}$ the
# forward propagated state of $\ket{\Psi_{\init}}$.

# In[790]:


# Reset results
oct_result = None


# In[791]:


iters = 10
if oct_result is not None:
    iters = oct_result.iters[-1] + iters

oct_result = krotov.optimize_pulses(
    objectives,
    pulse_options=pulse_options,
    tlist=tlist,
    propagator=krotov.propagators.expm,
    chi_constructor=krotov.functionals.chis_re,
    info_hook=krotov.info_hooks.chain(
        krotov.info_hooks.print_table(J_T=krotov.functionals.F_ss),
        print_fidelity
    ),
    check_convergence=krotov.convergence.Or(
        krotov.convergence.value_below(1e-3, name='J_T'),
        krotov.convergence.delta_below(1e-5),
        #krotov.convergence.check_monotonic_error,
    ),
    iter_stop=iters,
    continue_from = oct_result
)


# In[792]:


oct_result


# ## Simulate dynamics of the optimized field
# 
# Having obtained the optimized
# control field, we can now
# plot it and calculate the
# population dynamics under
# this field.

# In[793]:


[plot_pulse(c, tlist) for c in oct_result.optimized_controls]


# In[794]:


plot_pulse(np.abs(oct_result.optimized_controls[0]+1j*oct_result.optimized_controls[1]), tlist)
plot_pulse(np.angle(oct_result.optimized_controls[0]+1j*oct_result.optimized_controls[1]), tlist)


# In contrast to the dynamics under the guess field, the optimized field indeed
# drives the initial state $\ket{\Psi_{\init}} = \ket{0}$ to the desired target
# state $\ket{\Psi_{\tgt}} = \ket{1}$.

# In[795]:


opt_dynamics = [ob.mesolve(tlist, progress_bar=True) for ob in oct_result.optimized_objectives]


# In[796]:


plot_evolution(opt_dynamics)
#plot_all(opt_dynamics, ϕ)


# # Simulate dynamics in lab frame

# In[797]:


Ω = oct_result.optimized_controls[0]+1j*oct_result.optimized_controls[1]
pulses_lab = [Ω*np.exp(1j*ω_q*tlist), np.conj(Ω)*np.exp(-1j*ω_q*tlist)]
H_lab = hamiltonian(ω=1.0, ampl0=1, use_rotating=False, pulses=pulses_lab)
objectives_lab = [krotov.Objective(initial_state=ψ[0], target=ψ[1], H=H_lab) for ψ in ϕ]
opt_dynamics_lab = [ob.mesolve(tlist, progress_bar=True) for ob in objectives_lab]


# In[798]:


plot_pulse(Ω, tlist)
[plot_pulse(p, tlist) for p in pulses_lab]


# In[799]:


plot_evolution(opt_dynamics_lab)


# In[814]:


plot_cardinal([opt_dynamics_lab[0].states[0]])
plot_cardinal([opt_dynamics_lab[0].states[-1]])

