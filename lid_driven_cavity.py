import arrayfire as af
af.set_backend('opencl')
af.set_device(0)
import numpy as np
import os
import h5py

print(af.info())

from matplotlib import pyplot as pl
from tqdm import trange

import params


def add_boundary_conditions(u, v):

    u_bc = af.np_to_af_array(np.zeros([params.n, params.n]))
    v_bc = af.np_to_af_array(np.zeros([params.n, params.n]))

    u = af.moddims(u, params.n - 2, params.n - 2)
    v = af.moddims(v, params.n - 2, params.n - 2)

    u_bc[1:-1, 1:-1] = u
    v_bc[1:-1, 1:-1] = v

    u_bc[:, 0] += params.u_i_0

    return u_bc, v_bc

def sparse_matrix(n, Lambda, projection='false'):
    if projection == 'false':
        em1 = af.constant(-1, n ** 2)
        em1[n - 1:n ** 2:n] = 0

        ep1 = af.constant(-1, n ** 2)
        ep1[n:n ** 2:n] = 0

        sp_matrix  = af.identity(n ** 2, n ** 2, dtype=af.Dtype.f64) * (1 - 4 * Lambda)

        sp_matrix[n:, :-n] += Lambda * af.identity(n ** 2 - n, n ** 2 - n)
        sp_matrix[:-n, n:] += Lambda * af.identity(n ** 2 - n, n ** 2 - n)

        for i in range(n ** 2):
            for j in range(n ** 2):

                if (i == j + 1):
                    sp_matrix[i, j] -= Lambda * em1[j]

                if (i == j - 1):
                    sp_matrix[i, j] -= Lambda * ep1[j]

    if projection == 'true':
        sp_matrix = af.constant(0, n ** 2, n ** 2, dtype=af.Dtype.f64) * (1 - 4 * Lambda)
        ep1 = af.constant(1, n ** 2, 1, dtype=af.Dtype.f64)
        ep1[n: n ** 2: n] = 0
        ep1[1: n ** 2: n] = 2

        epn = af.constant(1, n ** 2, 1, dtype=af.Dtype.f64)
        epn[0: 2 * n] = 2

        em1 = af.constant(1, n ** 2, 1, dtype=af.Dtype.f64)
        em1[n - 1: n ** 2: n] = 0
        em1[n - 2: n ** 2: n] = 2

        emn = af.constant(1, n ** 2, 1, dtype=af.Dtype.f64)
        emn[n ** 2 - 2 * n: n ** 2] = 2

        d = af.constant(-4, n ** 2, 1, dtype=af.Dtype.f64)

        for i in range(n ** 2):
            for j in range(n ** 2):
                if (i == j):
                    sp_matrix[i, j] = d[i]
                if (i == j + 1):
                    sp_matrix[i, j] = em1[j]
                if (i == j + n):
                    sp_matrix[i, j] = emn[j]
                if (i == j - 1):
                    sp_matrix[i, j] = ep1[i + 1]
                if (i == j - n):
                    sp_matrix[i, j] = epn[i + n]
        sp_matrix *= Lambda

        sp_matrix[-1, :]  = 0
        sp_matrix[-1, -1] = 1


    return sp_matrix


def diffusion_step(u, v, Lambda, A_diffusion_inverse):
    '''
    The diffusion step. Lambda = visc * delta_t / (delta_x ** 2)

    The matlab code uses -lambda
    '''
    length = int(u.shape[0] ** 0.5)
    u[:length] -= Lambda

    u_new = af.matmul(A_diffusion_inverse, u)
    v_new = af.matmul(A_diffusion_inverse, v)

    return u_new, v_new

def div_velocity(u, v):
    '''
    Calculates the divergence of velocity vector (RHS of projection step).
    '''
    divergence_velocity = af.np_to_af_array(np.zeros([params.n, params.n]))
    reshaped_u   = af.moddims(u, params.n, params.n)
    reshaped_v   = af.moddims(v, params.n, params.n)

    divergence_velocity[1:-1, 1:-1] = (reshaped_u[:-2, 1:-1] - reshaped_u[2:, 1:-1] \
                              + reshaped_v[1:-1, :-2] - reshaped_v[1:-1, 2:])\
                              * (params.n - 1) / 2.0

    divergence_velocity[0, 1:-1]  = -reshaped_u[1, 1:-1] * (params.n - 1) / 1.0
    divergence_velocity[-1, 1:-1] = reshaped_u[-2, 1:-1] * (params.n - 1) / 1.0
    divergence_velocity[1:-1, 0]  = -reshaped_v[1:-1, 1] * (params.n - 1) / 1.0

    divergence_velocity[1:-1, -1] = reshaped_v[1:-1, -2] * (params.n - 1) / 1.0

    divergence_velocity = af.flat(divergence_velocity)

    return divergence_velocity

def projection_step_poisson(u, v, A_projection_inv):
    '''
    Solve the poisson equation to obtain q for the projection step.
    '''
    A_inverse = A_projection_inv
    divergence_velocity = div_velocity(u, v)
    q = af.matmul(A_inverse, divergence_velocity)

    return q

def grad_q(q, u, v):
    '''
    done - Matching qx qnd qy
    '''
    q = af.moddims(q, params.n, params.n)

    q_x = af.np_to_af_array(np.zeros([params.n, params.n]))
    q_y = af.np_to_af_array(np.zeros([params.n, params.n]))

    q_x[1:-1, 1:-1] = (q[:-2, 1:-1] - q[2:, 1:-1]) * (params.n - 1) / 2.0
    q_y[1:-1, 1:-1] = (q[1:-1, :-2] - q[1:-1, 2:]) * (params.n - 1) / 2.0

    # Horizontal boundary conditions, qx = 0
    q_y[0, 1:-1]  = (q[0, :-2] - q[0, 2:])   * (params.n - 1) / 2.0
    q_y[-1, 1:-1] = (q[-1, :-2] - q[-1, 2:]) * (params.n - 1) / 2.0

    # Vertical boundary conditions, qy = 0
    q_x[1:-1, 0]  = (q[:-2, 0] - q[2:, 0])   * (params.n - 1) / 2.0
    q_x[1:-1, -1] = (q[:-2, -1] - q[2:, -1]) * (params.n - 1) / 2.0
    #UNEXPLAINED SWITCHING in the second part of numerator in octave

    q_x = af.flat(q_x)
    q_y = af.flat(q_y)

    return q_x, q_y

def projection_step(u, v, A_projection):

    N_squared = params.n ** 2
    u_boundary_condition, v_boundary_condition = add_boundary_conditions(u, v)
    q = projection_step_poisson(u_boundary_condition, v_boundary_condition, A_projection)
    gradient_q = grad_q(q, u, v)

    u_boundary_condition = af.flat(u_boundary_condition)
    v_boundary_condition = af.flat(v_boundary_condition)

    u4 = u_boundary_condition - gradient_q[0]
    v4 = v_boundary_condition - gradient_q[1]

    u4 = af.flat(af.moddims(u4, params.n, params.n)[1:-1, 1:-1])
    v4 = af.flat(af.moddims(v4, params.n, params.n)[1:-1, 1:-1])

    return u4, v4

def advection_step(u, v):
    '''
    done - Matching u_new and v_new
    '''
    u_w_bc, v_w_bc = add_boundary_conditions(u, v)
    u = af.moddims(u_w_bc, params.n, params.n)
    v = af.moddims(v_w_bc, params.n, params.n)

    u_tile = u[1:-1, 1:-1]
    v_tile = v[1:-1, 1:-1]

    u_new = af.np_to_af_array(np.zeros([params.n, params.n]))
    v_new = af.np_to_af_array(np.zeros([params.n, params.n]))

    x0_tile = af.tile(af.np_to_af_array(np.linspace(0, 1, params.n)),
                      1, params.n)[1:-1, 1:-1]
    y0_tile = af.transpose(x0_tile)

    reference_0_tile = af.constant(0, params.n - 2, params.n - 2, dtype=af.Dtype.f64)
    reference_1_tile = af.constant(1, params.n - 2, params.n - 2, dtype=af.Dtype.f64)

    x1_tile = af.minof(af.maxof(x0_tile - params.delta_t * u_tile, reference_0_tile),
                       reference_1_tile)
    y1_tile = af.minof(af.maxof(y0_tile - params.delta_t * v_tile, reference_0_tile),
                       reference_1_tile)

    i_left_tile = af.minof(af.cast(x1_tile * (params.n - 1), af.Dtype.s64), (params.n - 2) *
                           reference_1_tile)
    i_left_flat = af.flat(i_left_tile)

    i_right_tile = i_left_tile + 1
    i_right_flat = af.flat(i_right_tile)

    j_bottom_tile = af.minof(af.cast(y1_tile * (params.n - 1), af.Dtype.s64), (params.n - 2) *
                             reference_1_tile)
    j_bottom_flat = af.flat(j_bottom_tile * params.n)

    j_top_tile = j_bottom_tile + 1
    j_top_flat = af.flat(j_top_tile * params.n)

    x_left_tile  = i_left_tile / (params.n - 1)
    x_right_tile = i_right_tile / (params.n - 1)

    y_bottom_tile = j_bottom_tile / (params.n - 1)
    y_top_tile    = j_top_tile / (params.n - 1)

    print(x_left_tile)

    flat_u = af.flat(u)
    flat_v = af.flat(v)


    u_top_left_tile = af.moddims(flat_u[i_left_flat + j_top_flat], params.n - 2,
                                params.n - 2)

    u_top_right_tile = af.moddims(flat_u[i_right_flat + j_top_flat], params.n - 2,
                                params.n - 2)

    u_bottom_left_tile = af.moddims(flat_u[i_left_flat + j_bottom_flat], params.n - 2,
                                params.n - 2)

    u_bottom_right_tile = af.moddims(flat_u[i_right_flat + j_bottom_flat], params.n - 2,
                                params.n - 2)

    v_top_left_tile = af.moddims(flat_v[i_left_flat + j_top_flat], params.n - 2,
                                params.n - 2)

    v_top_right_tile = af.moddims(flat_v[i_right_flat + j_top_flat], params.n - 2,
                                params.n - 2)

    v_bottom_left_tile = af.moddims(flat_v[i_left_flat + j_bottom_flat], params.n - 2,
                                params.n - 2)

    v_bottom_right_tile = af.moddims(flat_v[i_right_flat + j_bottom_flat], params.n - 2,
                                params.n - 2)


    u_upper_tile = u_top_left_tile\
                 + (u_top_right_tile - u_top_left_tile)\
                 * (x1_tile - x_left_tile) / params.delta_x


    u_lower_tile = u_bottom_right_tile\
                 + (u_bottom_left_tile - u_bottom_right_tile)\
                 * (x1_tile - x_left_tile) / params.delta_x

    u_new_tile = u_lower_tile + (u_upper_tile - u_lower_tile)\
               * (y1_tile - y_bottom_tile) / params.delta_x

    v_upper_tile = v_top_left_tile + (v_top_right_tile - v_top_left_tile)\
                 * (x1_tile - x_left_tile) / params.delta_x

    v_lower_tile = v_bottom_right_tile + (v_bottom_left_tile - v_bottom_right_tile)\
                 * (x1_tile - x_left_tile) / params.delta_x

    v_new_tile = v_lower_tile + (v_upper_tile - v_lower_tile)\
               * (y1_tile - y_bottom_tile) / params.delta_x

    u_new = af.flat(u_new_tile)
    v_new = af.flat(v_new_tile)

    return u_new, v_new


def time_evolution():
    u = params.u_init
    v = params.v_init
    h = 1 / (params.n -  1)
    delta_t = params.delta_t
    time    = params.t
    no_of_timesteps = time / delta_t
    Lambda = params.viscosity * delta_t / (params.delta_x ** 2)
    Lambda_projection = 1 / (params.delta_x ** 2)
    A_diffusion_inverse = af.np_to_af_array(np.linalg.inv(np.array(sparse_matrix(params.n - 2,
        -Lambda))))
    A_projection_inv = af.np_to_af_array(np.linalg.inv(np.array(sparse_matrix(params.n,
        Lambda_projection, 'true'))))


    for t_n in trange(int(no_of_timesteps)):
        u1, v1 = diffusion_step(u, v, -Lambda, A_diffusion_inverse) # note the -lambda

        u2, v2 = projection_step(u1, v1, A_projection_inv)

        u3, v3 = advection_step(u2, v2)
        u, v   = projection_step(u3, v3, A_projection_inv)

        # h5py files
        if(t_n % 100 == 0):
            with h5py.File('results/h5py/dump_timestep_%06d' %(int(t_n)) + '.hdf5', 'w') as hf:
                u_final, v_final = add_boundary_conditions(u, v)
                hf.create_dataset('u', data=u_final)
                hf.create_dataset('v', data=v_final)


        if (af.max(u) > 100):
            break

    return

time_evolution()
