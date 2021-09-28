"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc

import domain
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

    params.mu_ee       = params.mu.copy()
    params.T_ee        = params.T.copy()
    params.vel_drift_x = 0.*q1 + 0e-3
    params.vel_drift_y = 0.*q1 + 0e-3
    params.j_x         = 0.*q1
    params.j_y         = 0.*q1

    params.p_x, params.p_y = params.get_p_x_and_p_y(p1, p2)
    params.E_band   = params.band_energy(p1, p2)
    params.vel_band = params.band_velocity(p1, p2)

    # TODO: Injecting get_cartesian_coords into params to avoid circular dependency
    params.get_cartesian_coords = coords.get_cartesian_coords


    # Load shift indices for all 4 boundaries into params. Required to perform
    # mirroring operations along boundaries at arbitrary angles.
    params.shift_indices_left, params.shift_indices_right, \
    params.shift_indices_bottom, params.shift_indices_top = \
            compute_shift_indices(q1, q2, p1, p2, p3, params)

    params.x, params.y = coords.get_cartesian_coords(q1, q2,
                                                     q1_start_local_left=params.q1_start_local_left, 
                                                     q2_start_local_bottom=params.q2_start_local_bottom)

    params.q1 = q1; params.q2 = q2
    [[params.dx_dq1, params.dx_dq2], [params.dy_dq1, params.dy_dq2]] = jacobian_dx_dq(q1, q2,
                                                                                      q1_start_local_left=params.q1_start_local_left, 
                                                                                      q2_start_local_bottom=params.q2_start_local_bottom)
    [[params.dq1_dx, params.dq1_dy], [params.dq2_dx, params.dq2_dy]] = jacobian_dq_dx(q1, q2,
                                                                                      q1_start_local_left=params.q1_start_local_left, 
                                                                                      q2_start_local_bottom=params.q2_start_local_bottom)
    params.sqrt_det_g = sqrt_det_g(q1, q2,
                                       q1_start_local_left=params.q1_start_local_left, 
                                       q2_start_local_bottom=params.q2_start_local_bottom)

    # Calculation of integral measure
    # Evaluating velocity space resolution for each species:
    dp1 = []; dp2 = []; dp3 = []
    N_p1 = domain.N_p1; N_p2 = domain.N_p2; N_p3 = domain.N_p3
    p1_start = domain.p1_start; p1_end = domain.p1_end
    p2_start = domain.p2_start; p2_end = domain.p2_end
    p3_start = domain.p3_start; p3_end = domain.p3_end

    N_species = len(params.mass)
    for i in range(N_species):
        dp1.append((p1_end[i] - p1_start[i]) / N_p1)
        dp2.append((p2_end[i] - p2_start[i]) / N_p2)
        dp3.append((p3_end[i] - p3_start[i]) / N_p3)


    theta = af.atan(params.p_y / params.p_x)
    p_f   = params.fermi_momentum_magnitude(theta)
    
    if (params.p_space_grid == 'cartesian'):
        dp_x = dp1[0]; dp_y = dp2[0]; dp_z = dp3[0]
        params.integral_measure = \
          (4./(2.*np.pi*params.h_bar)**2) * dp_z * dp_y * dp_x

    elif (params.p_space_grid == 'polar2D'):
     	# In polar2D coordinates, p1 = radius and p2 = theta
        # Integral : \int delta(r - r_F) F(r, theta) r dr dtheta
        r = p1; theta = p2
        dp_r = dp1[0]; dp_theta = dp2[0]

        if (params.zero_temperature):
            # Assumption : F(r, theta) = delta(r-r_F)*F(theta)
            params.integral_measure = \
              (4./(2.*np.pi*params.h_bar)**2) * p_f * dp_theta

        else:
            params.integral_measure = \
              (4./(2.*np.pi*params.h_bar)**2) * r * dp_r * dp_theta
            

    else : 
        raise NotImplementedError('Unsupported coordinate system in p_space')

    # Initialize to zero
    f = 0*q1*p1

    # Parameters to define a gaussian in space (representing a 2D ball)
    A        = domain.N_p2 # Amplitude (required for normalization)
    sigma_x = 0.05 # Standard deviation in x
    sigma_y = 0.05 # Standard deviation in y
    x_0     = 0.75 # Center in x
    y_0     = 0. # Center in y

    # TODO: This will work with polar2D p-space only for the moment
    # Particles lying on the ball need to have the same velocity (direction)
    #theta_0_index = (5*N_p2/8) - 1 # Direction of initial velocity
    theta_0_index = int(4*domain.N_p2/8) # Direction of initial velocity
    
    print ("Initial angle : ")
    af.display(p2[theta_0_index])

    f[theta_0_index, :, :]  = A*af.exp(-( (params.x-x_0)**2/(2*sigma_x**2) + \
                                          (params.y-y_0)**2/(2*sigma_y**2)
                                        )
                                      ) +  A*af.exp(-( (params.x-x_0)**2/(2*sigma_x**2) + \
                                          (params.y-(-0.5))**2/(2*sigma_y**2)
                                        )
                                      ) + A*af.exp(-( (params.x-x_0)**2/(2*sigma_x**2) + \
                                          (params.y-0.5)**2/(2*sigma_y**2)
                                        )
                                      )

    # Initialize to zero
    f = 0.*f

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
