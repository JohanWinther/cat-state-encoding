
# coding: utf-8

# $ \newcommand{\cat}[2][\phantom{i}]{\ket{C^{#2}_{#1\alpha}}} $
# $ \newcommand{\ket}[1]{|#1\rangle} $
# $ \newcommand{\bra}[1]{\langle#1|} $
# $ \newcommand{\braket}[2]{\langle#1|#2\rangle} $
# $\newcommand{\au}{\hat{a}^\dagger}$
# $\newcommand{\ad}{\hat{a}}$
# $\newcommand{\bu}{\hat{b}^\dagger}$
# $\newcommand{\bd}{\hat{b}}$

# # Cat Code Preparation with Optimal Control
# <sup>Johan Winther</sup>
# 
# ## Goal
# Obtain a set of pulses which will encode the quantum information of a qubit with "cat codes" (and vice versa).
# 
# <sub>N. Ofek, A. Petrenko, R. Heeres, P. Reinhold, Z. Leghtas, B. Vlastakis, Y. Liu, L. Frunzio,
# S. M. Girvin, L. Jiang, M. Mirrahimi, M. H. Devoret &amp; R. J. Schoelkopf, ‘Extending the lifetime of a quantum bit with error correction in superconducting circuits’, Nature; London, vol. 536, no. 7617, pp. 441–445, Aug. 2016.</sub>

# # Outline
# * Why cat codes?
# * Optimal control (GRAPE)
# * Using optimal control to generate cat codes
# * My work so far

# # Why use cat codes for error correction?

# The cat code is comprised of the logical basis:
# ![image.png](attachment:image.png)

# <p style="text-align: center;">Notation: $ \ket{0}_L = \cat{\pm},\,\, \ket{1}_L = \cat[i]{\pm} $ </p>

# $ \ket{\psi} = c_0 \ket{C_\alpha^\pm} + c_1 \ket{C_{i\alpha}^\pm} $
# ![image.png](attachment:image.png)

# ## Crash course in Optimal control (GRAPE)

# ![image.png](attachment:image.png)

# We (usually) optimise for fidelity $\newcommand{\tr}[0]{\operatorname{tr}} f_{PSU} = \tfrac{1}{d} \big| \tr \{X_{targ}^{\dagger} X(T)\} \big| $

# # Optimal control for cat codes
# Jaynes-Cummings (dispersive)
# $$ \hat{H} = \omega_s\au\ad \,+ (\omega_a - \chi_{sa}\au\ad)\bu\bd $$
# $$-\, \frac{K_s}{2}\au{}^2\ad{}^2 \,-\, \frac{K_a}{2}\bu{}^2\bd{}^2 $$
# $$+\, \underbrace{\epsilon_a(t)\bu + \epsilon_a^*(t)\bd}_{\text{Qubit drive}} \,+\, \underbrace{\epsilon_s(t)\au + \epsilon_s^*(t)\ad}_{\text{Res drive}} $$
# 
# $$ \bu\bd = \ket{e}\bra{e} = \sigma_-\sigma_+ $$

# ![image.png](attachment:image.png)

# * Use optimisation to find the pulse envelope which will realise the unitary $     \hat{U}_t \underbrace{(c_0\ket{g} + c_1\ket{e})}_{\text{ancilla}}\underbrace{\ket{0}}_{\text{res}} = \underbrace{\ket{g}}_{\text{ancilla}} \underbrace{(c_0\cat{+} + c_1\cat[i]{+})}_{\text{res}} $

# * Practically this means we want to optimise for $K$ state transfers at the same time $ F_{oc} = \frac{1}{K^2} | \sum_k^K \braket{\psi_k(T)}{\psi_k^{\text{tar}}} |^2 $ where we encode many points on the Bloch sphere in the cat code basis.

# In[7]:


from numpy import sqrt
π = 3.1415926
ω_r = 8.3056 * 2 * π      # resonator frequency
ω_q = 6.2815 * 2 * π      # qubit frequency
K_q   = -2*π*297e-3    # Kerr qubit 200-300 MHz
K_r   = 2*π*4.5e-6   # Kerr res 1-10 Khz

ω_ef = ω_q + K_q
ω_gf = ω_q + K_q/2

χ = 25e-3 * 2 * π   # parameter in the dispersive hamiltonian

Δ = abs(ω_r - ω_q)    # detuning
g = sqrt(Δ * χ)  # coupling strength that is consistent with chi
print(g)


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ### My work so far
# * Use the pulse optimisation tool in `QuTiP` (quantum simulation toolbox in Python), or other framework

# * Project status - more difficult than expected
# * Even for the simple things, e.g. bit flip pulse, there are problems with convergence and numerical errors
# * Custom constraints on the pulses aren't implemented yet (nor general optimization goals) in QuTiP
# * I will try `Krotov`, another python toolbox which uses the Krotov method instead of GRAPE

# * Goal of the thesis is to realise this method and then eventually evaluate possible extensions:
#     * Other bosonic codes besides "2 lobe"-cat codes
#     * Optimise the coefficients of Fock states (theoretical curiosity)

# ## Thank you for listening! Any questions?
