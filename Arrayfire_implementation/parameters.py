import arrayfire as af
import numpy as np

# Parameter n for grid resolution.
n = 30

t       = 30
delta_t = 0.01

viscosity = 1 / 100

# Boundary conditions for the domain.
u_i_0 = 1.
u_j_0 = 0.

u_i_N_minus1 = 0.
u_j_N_minus1 = 0.

v_i_0 = 0.
v_j_0 = 0.

v_i_N_minus1 = 0.
v_j_N_minus1 = 0.

u_init = af.np_to_af_array(np.zeros([(n - 2) ** 2]))
v_init = af.np_to_af_array(np.zeros([(n - 2) ** 2]))


#####
delta_x = 1.0 / (n - 1)
