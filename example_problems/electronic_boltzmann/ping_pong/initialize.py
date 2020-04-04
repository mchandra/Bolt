"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc
import domain

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

    #TODO : Remove the following variables from here,
    # already defined in domain.py
    N_p1 = domain.N_p1
    N_p2 = domain.N_p2
    N_p3 = domain.N_p3
    N_q1 = domain.N_q1
    N_q2 = domain.N_q2
    N_g  = domain.N_ghost
    N_s = len(params.mass) # Number of species

    # Initialize to zero
    f = 0.*q1*p1

    # Parameters to define a gaussian in space (representing a 2D ball)
    A        = N_p2 # Amplitude (required for normalization)
    sigma_q1 = 0.05 # Standard deviation in q1
    sigma_q2 = 0.05 # Standard deviation in q2
    q1_0     = 0.5 # Center in q1
    q2_0     = 0.5 # Center in q2


    # TODO: This will work with polar2D coordinates only for the moment
    # Particles lying on the ball need to have the same velocity (direction)
    theta_0_index = (5*N_p2/8) - 1 # Direction of initial velocity

    f[theta_0_index, :, :]  = A*af.exp(-(((q1-q1_0)**2)/(2*sigma_q1**2) + \
                                       ((q2-q2_0)**2)/(2*sigma_q2**2)))

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
