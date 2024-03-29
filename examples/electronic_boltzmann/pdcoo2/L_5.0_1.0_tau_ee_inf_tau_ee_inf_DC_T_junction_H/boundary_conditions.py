import numpy as np
import arrayfire as af
import domain

in_q1_left   = 'mirror'
in_q1_right  = 'mirror'
in_q2_bottom = 'mirror+dirichlet'
in_q2_top    = 'mirror+dirichlet'

@af.broadcast
def f_top(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu
    
    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq

    q1_contact_start = params.contact_start
    q1_contact_end   = params.contact_end
    contact_width    = q1_contact_end - q1_contact_start
        
    q1_contact_start_2 = 0.
    q1_contact_end_2   = 1.73/2
    contact_width_2    = q1_contact_end_2 - q1_contact_start_2
        
    vel_drift_y_in    = -params.vel_drift_x_in/contact_width #Right
    vel_drift_y_in_2  = 10.*params.vel_drift_x_in/contact_width_2 #Left

    if (params.p_space_grid == 'cartesian'):
        p_x = p1 
        p_y = p2
    elif (params.p_space_grid == 'polar2D'):
        p_x = p1 * af.cos(p2)
        p_y = p1 * af.sin(p2)
    else:
        raise NotImplementedError('Unsupported coordinate system in p_space')


    fermi_dirac_in   = (1./(af.exp( (E_upper - vel_drift_y_in*p_y - mu)/(k*T) ) + 1.)
                       )
    fermi_dirac_in_2 = (1./(af.exp( (E_upper - vel_drift_y_in_2*p_y - mu)/(k*T) ) + 1.)
                       )

    if (params.contact_geometry=="straight"):
        # Contacts on either side of the device

        cond = ((q1 >= q1_contact_start) & \
                (q1 <= q1_contact_end) \
               )

        cond_2 = ((q1 >= q1_contact_start_2) & \
                (q1 <= q1_contact_end_2) \
               )

        f_left = cond*fermi_dirac_in + cond_2*fermi_dirac_in_2 + (1-cond_2)*(1 - cond)*f

    elif (params.contact_geometry=="turn_around"):
        # Contacts on the same side of the device

        vel_drift_x_out = -params.vel_drift_x_in * np.sin(omega*t)

        fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_x_out*p_x - mu)/(k*T) ) + 1.)
                          )
    
        # TODO: set these parameters in params.py
        cond_in  = ((q2 >= 3.5) & (q2 <= 4.5))
        cond_out = ((q2 >= 5.5) & (q2 <= 6.5))
    
        f_left =  cond_in*fermi_dirac_in + cond_out*fermi_dirac_out \
                + (1 - cond_in)*(1 - cond_out)*f

    af.eval(f_left)
    return(f_left)

@af.broadcast
def f_bottom(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq

    q1_contact_start = params.contact_start
    q1_contact_end   = params.contact_end
    contact_width    = q1_contact_end - q1_contact_start
        
    q1_contact_start_2 = 0.
    q1_contact_end_2   = 1.73/2
    contact_width_2    = q1_contact_end_2 - q1_contact_start_2

    vel_drift_y_out   = -params.vel_drift_x_out/contact_width
    vel_drift_y_out_2 = 10.*params.vel_drift_x_out/contact_width_2
    
    if (params.p_space_grid == 'cartesian'):
        p_x = p1 
        p_y = p2
    elif (params.p_space_grid == 'polar2D'):
        p_x = p1 * af.cos(p2)
        p_y = p1 * af.sin(p2)
    else:
        raise NotImplementedError('Unsupported coordinate system in p_space')

    fermi_dirac_out   = (1./(af.exp( (E_upper - vel_drift_y_out*p_y - mu)/(k*T) ) + 1.)
                        )
    fermi_dirac_out_2 = (1./(af.exp( (E_upper - vel_drift_y_out_2*p_y - mu)/(k*T) ) + 1.)
                        )
    
    if (params.contact_geometry=="straight"):
        # Contacts on either side of the device
        
        cond   = ((q1 >= q1_contact_start) & \
                  (q1 <= q1_contact_end) \
                 )
        cond_2 = ((q1 >= q1_contact_start_2) & \
                  (q1 <= q1_contact_end_2) \
                 )

        f_right = cond*fermi_dirac_out + cond_2*fermi_dirac_out_2 + (1-cond_2)*(1 - cond)*f

    elif (params.contact_geometry=="turn_around"):
        # Contacts on the same side of the device
        
        f_right = f

    af.eval(f_right)
    return(f_right)
