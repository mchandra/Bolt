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
    x_0     = 0.5 # Center in x
    y_0     = 0.5 # Center in y

    # TODO: This will work with polar2D p-space only for the moment
    # Particles lying on the ball need to have the same velocity (direction)
    #theta_0_index = (5*N_p2/8) - 1 # Direction of initial velocity
    theta_0_index = int(6*domain.N_p2/8) # Direction of initial velocity
    
    print ("Initial angle : ")
    af.display(p2[theta_0_index])

    # Load shift indices for all 4 boundaries into params. Required to perform
    # mirroring operations along boundaries at arbitrary angles.
    params.shift_indices_left, params.shift_indices_right, \
    params.shift_indices_bottom, params.shift_indices_top = \
            compute_shift_indices(q1, q2, p1, p2, p3, params)   

    x, y = coords.get_cartesian_coords(q1, q2)
 
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


# TODO : The code below is specific to the zero T formulation. Should it be here?

def compute_shift_indices(q1, q2, p1, p2, p3, params):
    """
    Inject the shift_indices corresponding to a shift operation of -2*theta for the left boundary,
    where theta is the angular variation of the left boundary.
    """
    N_theta       = domain.N_p2

    # Define edge indices
    left_edge = 0; right_edge = -1
    bottom_edge = 0; top_edge = -1
    
    temp = af.range(N_theta) # Indices with no shift. Shape : N_theta


    # Left boundary

    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, 0, :]
    N_q2_local    = shift_indices.dims()[3]

    # Get the angular variation of the left boundary.
    theta_left = coords.get_theta(q1, q2, "left")[0, 0, left_edge, :]
    theta_left = af.moddims(theta_left, N_q2_local) # Convert to 1D array

    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_left
    shifts  = -((2*theta_left)/(2*np.pi))*N_theta 

    # Populate shift_indices 2D array using shifts.
    for index, value in enumerate(shifts):
        shift_indices[:, 0, 0, index] = af.shift(N_theta*index+temp, int(value.scalar()))

    # Convert into a 1D array
    shift_indices_left = af.moddims(shift_indices, N_theta*N_q2_local)
   

    # Right boundary 

    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, 0, :] # Shape : N_theta x 1 x 1 x N_q2+2*N_g

    # Get the angular variation of the right boundary.
    theta_right = coords.get_theta(q1, q2, "right")[0, 0, right_edge, :]
    theta_right = af.moddims(theta_right, N_q2_local) # Convert to 1D array

    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_right
    shifts  = -(theta_right/np.pi)*N_theta


    # Populate shift_indices 2D array using shifts.
    for index, value in enumerate(shifts):
        shift_indices[:, 0, 0, index] = af.shift(N_theta*index+temp, int(value.scalar()))

    # Convert into a 1D array
    shift_indices_right = af.moddims(shift_indices, N_theta*N_q2_local)

    
    # Bottom boundary
    
    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, :, 0] # Shape : N_theta x 1 x  N_q1+2*N_g x 1
    N_q1_local    = shift_indices.dims()[2]

    # Get the angular variation of the bottom boundary.
    theta_bottom = coords.get_theta(q1, q2, "bottom")[0, 0, :, bottom_edge]
    theta_bottom = af.moddims(theta_bottom, N_q1_local) # Convert to 1D array

    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_bottom
    shifts  = -(theta_bottom/np.pi)*N_theta

    # Populate shift_indices 2D array using shifts.
    for index, value in enumerate(shifts):
        shift_indices[:, 0, index, 0] = af.shift(N_theta*index+temp, int(value.scalar()))

    # Convert into a 1D array
    shift_indices_bottom = af.moddims(shift_indices, N_theta*N_q1_local)


    # Top Boundary

    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, :, 0] # Shape : N_theta x 1 x N_q1+2*N_g x 1
    
    # Get the angular variation of the top boundary.
    theta_top = coords.get_theta(q1, q2, "top")[0, 0, :, top_edge]
    theta_top = af.moddims(theta_top, N_q1_local) # Convert to 1D array
 
    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_top
    shifts  = -(theta_top/np.pi)*N_theta


    # Populate shift_indices 2D array using shifts.
    for index, value in enumerate(shifts):
        shift_indices[:, 0, index, 0] = af.shift(N_theta*index+temp, int(value.scalar()))

    #Convert to a 1D array
    shift_indices_top = af.moddims(shift_indices, N_theta*N_q1_local)
    
    return(shift_indices_left, shift_indices_right, shift_indices_bottom, shift_indices_top)
