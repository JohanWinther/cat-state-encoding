
# coding: utf-8

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
# \newcommand{bra}[1]{\left\langle#1\right\vert}
# \newcommand{ket}[1]{\left\vert#1\right\rangle}
# \newcommand{braket}[2]{\left\langle #1\vphantom{#2} \mid
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

# 
# ## Define the Hamiltonian
# 
# The Hamiltonian
# $$\op{H}_{0} = \begin{pmatrix}
#     0 & 0 & 0\\
#     0 & \omega_q & 0\\
#     0 & 0 & 2\omega_q + K_q
# \end{pmatrix}$$
# represents an anharmonic oscillator with energy
# level splitting $\omega_q$ in the basis
# $\{\ket{0},\ket{1}\}$ and anharmonicity $K_q$. The control
# field
# $\epsilon(t)$ is assumed to couple via
# the
# Hamiltonian $\op{H}_{1}(t) =
# \Omega(t)e^{i\omega_q t} \op{b} + \Omega^*(t)e^{-i\omega_q t}\op{b}^\dagger$ to the system,
# i.e., the control
# field effectively
# drives
# transitions between the
# states.

# # Pulse optimisation for a $\ket{0}\rightarrow\ket{1}$ transfer

# Pulse optimisation is best to do in the rotating frame where the full Hamiltonian becomes
# $$\op{H}(t) = \begin{pmatrix}
#     0 & \Omega(t) & 0\\
#     \Omega^*(t) & 0 & \sqrt{2}\Omega(t)\\
#     0 & \sqrt{2}\Omega^*(t) & K_q
# \end{pmatrix}$$
# The pulse optimisation will optimise the shape of $\Omega(t)$ which will act as an envelope for the carrier signal in the lab frame.

# ## Hardware constraints
# Since the pulse needs to be physically realisable these constraints have been imposed:
# - Maximum amplitude limitation
# - Start and end of pulse is zero
# - Pulse shape is smoothed
# 
# ## Optimisation
# Optimisation is done for pulse lengths $T$ from 0.4 ns to 36 ns (100 time steps) and the fidelity is calculated as 
# $F=|\braket{\Psi(T)}{\Psi_{\text{tar}}}|^2$.
# When $F> 0.99999$ or the gradient is lower than $10^{-6}$ the optimisation stops. Optimisation is done separately for the real and imaginary part of $\Omega(t)$ due to software constraints.

# ![image.png](attachment:image.png)

# ## T = 21 ns, F > 0.99999

# ### Optimized pulse shapes

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ### Pulse spectrum
# ![image.png](attachment:image.png)

# ### Occupation evolution
# ![image.png](attachment:image.png)

# ### Bloch sphere evolution
# ![image.png](attachment:image.png)

# ### Density matrix
# ![image.png](attachment:image.png)

# # T = 9 ns, F ~ 0.5

# ### Optimised pulses
# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ### Pulse spectrum
# ![image.png](attachment:image.png)

# ### Occupation evolution
# ![image.png](attachment:image.png)

# ### Bloch sphere evolution
# ![image.png](attachment:image.png)

# ### Density matrix
# ![image.png](attachment:image.png)

# # Transfer: $\ket{0} \rightarrow \ket{2}$

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)
