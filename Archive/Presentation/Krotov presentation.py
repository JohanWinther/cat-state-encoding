
# coding: utf-8

# ## Krotov’s method
# D. M. Reich, M. Ndong, and C. P. Koch, J. Chem. Phys. 136, 104103 (2012) (arXiv:1008.5126)
# 
#  - Krotov’s method is one of the two leading gradient-based optimization algorithms used in numerical quantum optimal control. (other one is GRAPE)
#  - The goal is to optimise the shape of control pulses to realise a state transfer or a quantum gate.
#  - Krotov simulates the dynamics of a quantum system under a set of initial controls, and evaluates the result with respect to an optimization functional to be minimized. It then iteratively modifies the controls to guarantee a monotonically decreasing value in the optimization functional.

# ## Features of the Python package: [Krotov](https://krotov.readthedocs.io/)
# - Built on top of [QuTiP](http://qutip.org) (which already implements GRAPE and CRAB)
# - Simultaneously optimize over an arbitrary list of objectives (e.g. multiple state-to-state transfers)
# - Optimize over multiple control fields at the same time
# - Arbitrary equations of motion (through a propagator callback function)
# - Arbitrary optimization functionals (through chi_constructor callback function)
# - Allows injection of arbitrary code (through modify_params_after_iter function)
# - Customizable parallelization of the propagation of different objectives
# - Customizable analysis and convergence check
# - Support for dissipative dynamics (Liouville space)
# - Convenience constructors for “ensemble optimization” to obtain robust controls

# ## Advantages over GRAPE in QuTiP
# - Can start from an arbitrary set of guess controls. In the GRAPE implementation, guess pulses can only be chosen from a specific set of options (including “random”). This makes sense for a control field that is piecewise constant with relatively few switching points, but is very disadvantageous for time-continuous controls.
# - Krotov’s method has complete flexibility in which propagation method is used, while QuTiP’s GRAPE only allows to choose between fixed number of methods for time-propagation. Supplying a problem-specific propagator is not possible.
# - Save optimization into a dump file to continue later or restore after crashes.
# - Multiple convergence criteria.
# - Possible to change (some) parameters between iterations.

# ## How to use Krotov

# ```bash
# pip install krotov
# ```

# In[2]:


import krotov


# - Define the necessary quantum operators and states using QuTiP.
# - Create a list of objectives, as instances of `krotov.Objective`
# - Call `krotov.optimize_pulses` to perform an optimization of an arbitrary number of control fields over all the objectives.

# # Example of $|0\rangle \rightarrow |1\rangle$ opt.
# https://krotov.readthedocs.io/en/latest/notebooks/01_example_simple_state_to_state.html

# In[3]:


print(krotov.__citation__)

