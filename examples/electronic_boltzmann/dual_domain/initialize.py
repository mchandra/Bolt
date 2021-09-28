"""
Functions which are used in assigning the I.C's to
othe system.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc

import coords

from bolt.lib.utils.coord_transformation \
    import compute_shift_indices, jacobian_dq_dx, jacobian_dx_dq, sqrt_det_g

def initialize_f(q1, q2, p1, p2, p3, params):
   
    PETSc.Sys.Print("Initializing f")
    k = params.boltzmann_constant
    
    params.mu          = 0.*q1 + params.initial_mu
    params.T           = 0.*q1 + params.initial_temperature
    params.vel_drift_x = 0.*q1
    params.vel_drift_y = 0.*q1
    params.phi         = 0.*q1

    params.mu_ee       = params.mu.copy()
    params.T_ee        = params.T.copy()
    params.vel_drift_x = 0.*q1 + 0e-3
    params.vel_drift_y = 0.*q1 + 0e-3
    params.j_x         = 0.*q1
    params.j_y         = 0.*q1

    params.E_band   = params.band_energy(p1, p2)
    params.vel_band = params.band_velocity(p1, p2)

    E_upper = params.E_band + params.charge[0]*params.phi

    if (params.p_space_grid == 'cartesian'):
        p_x = p1
        p_y = p2
    elif (params.p_space_grid == 'polar2D'):
        p_x = p1 * af.cos(p2)
        p_y = p1 * af.sin(p2)
    else:
        raise NotImplementedError('Unsupported coordinate system in p_space')

    # Load shift indices for all 4 boundaries into params. Required to perform
    # mirroring operations along boundaries at arbitrary angles.
    params.shift_indices_left, params.shift_indices_right, \
    params.shift_indices_bottom, params.shift_indices_top = \
            compute_shift_indices(q1, q2, p1, p2, p3, params)   

    params.x, params.y = coords.get_cartesian_coords(q1, q2)
    params.q1 = q1; params.q2 = q2
    [[params.dx_dq1, params.dx_dq2], [params.dy_dq1, params.dy_dq2]] = jacobian_dx_dq(q1, q2)
    [[params.dq1_dx, params.dq1_dy], [params.dq2_dx, params.dq2_dy]] = jacobian_dq_dx(q1, q2)
    params.sqrt_det_g = sqrt_det_g(q1, q2)

    f = (1./(af.exp( (E_upper - params.vel_drift_x*p_x
                              - params.vel_drift_y*p_y
                              - params.mu
                    )/(k*params.T) 
                  ) + 1.
           ))


    af.eval(f)
    return(f)


def initialize_E(q1, q2, params):
    
    E1 = 0.*q1
    E2 = 0.*q1
    E3 = 0.*q1

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = 0.*q1
    B2 = 0.*q1
    B3 = 0.*q1

    return(B1, B2, B3)


