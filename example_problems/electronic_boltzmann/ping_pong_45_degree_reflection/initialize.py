"""
Functions which are used in assigning the I.C's to
othe system.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc
import domain
import coords

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

    # Initialize to zero
    f = 0*q1*p1

    # Parameters to define a gaussian in space (representing a 2D ball)
    A        = domain.N_p2 # Amplitude (required for normalization)
    sigma_x = 0.05 # Standard deviation in x
    sigma_y = 0.05 # Standard deviation in y
    x_0     = 0.75 # Center in x
    y_0     = 0.5 # Center in y

    # TODO: This will work with polar2D p-space only for the moment
    # Particles lying on the ball need to have the same velocity (direction)
    #theta_0_index = (5*N_p2/8) - 1 # Direction of initial velocity
    theta_0_index = int(6*domain.N_p2/8) # Direction of initial velocity
    print ("Initial angle : ")
    af.display(p2[theta_0_index])

    x, y = coords.get_cartesian_coords(q1, q2)

    # Inject the boundary angles into params
    left_edge = 0; right_edge = -1
    params.theta_left   = af.moddims(coords.get_theta(q1, q2, "left")[0, 0, left_edge, :], f.dims()[3])
    params.theta_right  = af.moddims(coords.get_theta(q1, q2, "right")[0, 0, right_edge, :], f.dims()[3])

    bottom_edge = 0; top_edge = -1
    params.theta_top    = af.moddims(coords.get_theta(q1, q2, "top")[0, 0, :, top_edge], f.dims()[2])
    params.theta_bottom = af.moddims(coords.get_theta(q1, q2, "bottom")[0, 0, :, bottom_edge], f.dims()[2])

    # Load shift indices
    load_shift_indices_left(q1, q2, p1, p2, p3, params)   
    load_shift_indices_right(q1, q2, p1, p2, p3, params)   
    load_shift_indices_bottom(q1, q2, p1, p2, p3, params)   
    load_shift_indices_top(q1, q2, p1, p2, p3, params)   
 
    f[theta_0_index, :, :]  = A*af.exp(-( (x-x_0)**2/(2*sigma_x**2) + \
                                          (y-y_0)**2/(2*sigma_y**2)
                                        )
                                      )

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


def load_shift_indices_left(q1, q2, p1, p2, p3, params):
    
    # Generate the shift_indices 2D array for the left boundary

    N_theta = domain.N_p2
    shifts  = -(params.theta_left/np.pi)*N_theta #Shifts have to be in the negative direction

    shift_indices = (0.*q1*p1)[:, 0, 0, :] # Initialize to zero, shape : N_theta x N_q2+2*N_g

    temp = af.range(shift_indices.dims()[0]) # Should be of shape N_theta
    index = 0
    for value in shifts:
        shift_indices[:, 0, 0, index] = af.shift(temp.dims()[0]*index+temp, int(value.scalar()))
        index = index + 1

    params.shift_indices_left = af.moddims(shift_indices, shift_indices.dims()[0]*shift_indices.dims()[3])
    return

def load_shift_indices_right(q1, q2, p1, p2, p3, params):
    
    # Generate the shift_indices 2D array for the right boundary

    N_theta = domain.N_p2
    shifts  = -(params.theta_right/np.pi)*N_theta

    shift_indices = (0.*q1*p1)[:, 0, 0, :] # Initialize to zero, shape : N_theta x N_q2+2*N_g

    temp = af.range(shift_indices.dims()[0]) # Should be of shape N_theta
    index = 0
    for value in shifts:
        shift_indices[:, 0, 0, index] = af.shift(temp.dims()[0]*index+temp, int(value.scalar()))
        index = index + 1

    #params.shift_indices_right = shift_indices
    params.shift_indices_right = af.moddims(shift_indices, shift_indices.dims()[0]*shift_indices.dims()[3])
    return


def load_shift_indices_bottom(q1, q2, p1, p2, p3, params):
    
    # Generate the shift_indices 2D array for the right boundary

    N_theta = domain.N_p2
    shifts  = -(params.theta_bottom/np.pi)*N_theta

    shift_indices = (0.*q1*p1)[:, 0, :, 0] # Initialize to zero, shape : N_theta x N_q1+2*N_g

    temp = af.range(shift_indices.dims()[0]) # Should be of shape N_theta
    index = 0
    for value in shifts:
        shift_indices[:, 0, index, 0] = af.shift(temp.dims()[0]*index+temp, int(value.scalar()))
        index = index + 1

    params.shift_indices_bottom = af.moddims(shift_indices, shift_indices.dims()[0]*shift_indices.dims()[2])
    return

def load_shift_indices_top(q1, q2, p1, p2, p3, params):
    
    # Generate the shift_indices 2D array for the right boundary

    N_theta = domain.N_p2
    shifts  = -(params.theta_top/np.pi)*N_theta

    shift_indices = (0.*q1*p1)[:, 0, :, 0] # Initialize to zero, shape : N_theta x N_q1+2*N_g

    temp = af.range(shift_indices.dims()[0]) # Should be of shape N_theta
    index = 0
    for value in shifts:
        shift_indices[:, 0, index, 0] = af.shift(temp.dims()[0]*index+temp, int(value.scalar()))
        index = index + 1

    #params.shift_indices_top = shift_indices
    params.shift_indices_top = af.moddims(shift_indices, shift_indices.dims()[0]*shift_indices.dims()[2])
    return
