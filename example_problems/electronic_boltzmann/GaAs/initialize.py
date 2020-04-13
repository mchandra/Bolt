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

    # Load shift indices for all 4 boundaries into params. Required to perform
    # mirroring operations along boundaries at arbitrary angles.
    load_shift_indices_left(q1, q2, p1, p2, p3, params)   
    load_shift_indices_right(q1, q2, p1, p2, p3, params)   
    load_shift_indices_bottom(q1, q2, p1, p2, p3, params)   
    load_shift_indices_top(q1, q2, p1, p2, p3, params)   

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


# TODO : The code below is specific to the zero T formulation. Should it be here?

def load_shift_indices_left(q1, q2, p1, p2, p3, params):
    """
    Inject the shift_indices corresponding to a shift operation of -2*theta for the left boundary,
    where theta is the angular variation of the left boundary.
    """
    # Initialize to zero 
    shift_indices = (0.*q1*p1)[:, 0, 0, :]
    dims  = shift_indices.dims() # Shape : N_theta x 1 x 1 x N_q2+2*N_g

    # Get the angular variation of the left boundary.
    left_edge = 0
    theta_left = af.moddims(coords.get_theta(q1, q2, "left")[0, 0, left_edge, :], dims[3])

    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_left
    N_theta = domain.N_p2
    shifts  = -((2*theta_left)/(2*np.pi))*N_theta 

    # Populate shift_indices 2D array using shifts.
    temp = af.range(dims[0]) # Should be of shape N_theta
    #index = 0
    for index, value in enumerate(shifts):
        shift_indices[:, 0, 0, index] = af.shift(dims[0]*index+temp, int(value.scalar()))
    #    index = index + 1

    # Convert into a 1D array and store in params
    params.shift_indices_left = af.moddims(shift_indices, dims[0]*dims[3])
    return

def load_shift_indices_right(q1, q2, p1, p2, p3, params):
    """
    Inject the shift_indices corresponding to a shift operation of -2*theta for the right boundary,
    where theta is the angular variation of the right boundary
    """
    
    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, 0, :] # Shape : N_theta x 1 x 1 x N_q2+2*N_g
    dims = shift_indices.dims()

    # Get the angular variation of the right boundary.
    right_edge = -1
    theta_right = af.moddims(coords.get_theta(q1, q2, "right")[0, 0, right_edge, :], dims[3])

    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_right
    N_theta = domain.N_p2
    shifts  = -(theta_right/np.pi)*N_theta


    # Populate shift_indices 2D array using shifts.
    temp = af.range(dims[0]) # Should be of shape N_theta
    #index = 0
    for index, value in enumerate(shifts):
        shift_indices[:, 0, 0, index] = af.shift(dims[0]*index+temp, int(value.scalar()))
    #    index = index + 1

    # Convert into a 1D array and store in params
    params.shift_indices_right = af.moddims(shift_indices, dims[0]*dims[3])
    return


def load_shift_indices_bottom(q1, q2, p1, p2, p3, params):
    """
    Inject the shift_indices corresponding to a shift operation of -2*theta for the bottom boundary,
    where theta is the angular variation of the bottom boundary
    """
    
    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, :, 0] # Shape : N_theta x 1 x  N_q1+2*N_g x 1
    dims = shift_indices.dims()

    # Get the angular variation of the bottom boundary.
    bottom_edge = 0
    theta_bottom = af.moddims(coords.get_theta(q1, q2, "bottom")[0, 0, :, bottom_edge], dims[2])

    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_bottom
    N_theta = domain.N_p2
    shifts  = -(theta_bottom/np.pi)*N_theta

    # Populate shift_indices 2D array using shifts.
    temp = af.range(shift_indices.dims()[0]) # Should be of shape N_theta
    #index = 0
    for index, value in enumerate(shifts):
        shift_indices[:, 0, index, 0] = af.shift(dims[0]*index+temp, int(value.scalar()))
    #   index = index + 1

    # Convert into a 1D array and store in params
    params.shift_indices_bottom = af.moddims(shift_indices, dims[0]*dims[2])
    return

def load_shift_indices_top(q1, q2, p1, p2, p3, params):
    """
    Inject the shift_indices corresponding to a shift operation of -2*theta for the top boundary,
    where theta is the angular variation of the top boundary
    """

    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, :, 0] # Shape : N_theta x 1 x N_q1+2*N_g x 1
    dims = shift_indices.dims()
    
    # Get the angular variation of the top boundary.
    top_edge = -1
    theta_top = af.moddims(coords.get_theta(q1, q2, "top")[0, 0, :, top_edge], dims[2])
 
    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_top
    N_theta = domain.N_p2
    shifts  = -(theta_top/np.pi)*N_theta


    # Populate shift_indices 2D array using shifts.
    temp = af.range(dims[0]) # Should be of shape N_theta
    #index = 0
    for index, value in enumerate(shifts):
        shift_indices[:, 0, index, 0] = af.shift(dims[0]*index+temp, int(value.scalar()))
    #    index = index + 1

    #Convert to a 1D array and store in params
    params.shift_indices_top = af.moddims(shift_indices, dims[0]*dims[2])
    return
