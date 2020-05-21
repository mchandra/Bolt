import numpy as np
import arrayfire as af

import domain_1 as domain
import coords_1 as coords

import params as params_common

in_q1_left   = 'mirror'
in_q1_right  = 'mirror'
in_q2_bottom = 'mirror'
in_q2_top    = 'mirror+dirichlet'

@af.broadcast
def f_left(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu
    
    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    
    if (params.source_type == 'AC'):
        vel_drift_x_in  = params.vel_drift_x_in  * np.sin(omega*t)
    elif (params.source_type == 'DC'):
        vel_drift_x_in  = params.vel_drift_x_in
    else:
        raise NotImplementedError('Unsupported source_type')

    if (params.p_space_grid == 'cartesian'):
        p_x = p1 
        p_y = p2
    elif (params.p_space_grid == 'polar2D'):
        p_x = p1 * af.cos(p2)
        p_y = p1 * af.sin(p2)
    else:
        raise NotImplementedError('Unsupported coordinate system in p_space')


    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p_x - mu)/(k*T) ) + 1.)
                     )

    x, y = coords.get_cartesian_coords(q1, q2)

    y_contact_start = params.contact_start
    y_contact_end   = params.contact_end
    
    cond = ((y >= y_contact_start) & \
            (y <= y_contact_end) \
           )


    f_left = cond*fermi_dirac_in + (1 - cond)*f


    af.eval(f_left)
    return(f_left)

@af.broadcast
def f_right(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    if (params.source_type == 'AC'):
        vel_drift_x_out = params.vel_drift_x_out * np.sin(omega*t)
    elif (params.source_type == 'DC'):
        vel_drift_x_out = params.vel_drift_x_out 
    else:
        raise NotImplementedError('Unsupported source_type')
    
    if (params.p_space_grid == 'cartesian'):
        p_x = p1 
        p_y = p2
    elif (params.p_space_grid == 'polar2D'):
        p_x = p1 * af.cos(p2)
        p_y = p1 * af.sin(p2)
    else:
        raise NotImplementedError('Unsupported coordinate system in p_space')

    fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_x_out*p_x - mu)/(k*T) ) + 1.)
                      )

    x, y = coords.get_cartesian_coords(q1, q2)
    
    y_contact_start = params.contact_start
    y_contact_end   = params.contact_end
    
    cond = ((y >= y_contact_start) & \
            (y <= y_contact_end) \
           )

    f_right = cond*fermi_dirac_out + (1 - cond)*f

    af.eval(f_right)
    return(f_right)


@af.broadcast
def f_top(f, t, q1, q2, p1, p2, p3, params):
    
    N_g = domain.N_ghost

    q1_connector_start_index = N_g #TODO
    q1_connector_end_index   = -N_g #TODO

    # Get information about the other domain
    fermi_dirac_2 = params_common.f_2
    print ("fermi_dirac_2 : ", fermi_dirac_2.shape)

    f_top = f
    print ("f top : ", f_top.shape)
    # Top ghost zone is filled with f from the bottom of the other domain
    f_top[:, :, q1_connector_start_index:q1_connector_end_index, -N_g:] = fermi_dirac_2[:, :, N_g:-N_g, N_g:2*N_g]
    
    print ('boundary_conditions.py : ', f_top.shape)
    af.eval(f_top)
    return(f_top)
