import numpy as np
import arrayfire as af
import domain

in_q1_left   = 'mirror+dirichlet'
in_q1_right  = 'mirror+dirichlet'
in_q2_bottom = 'mirror'
in_q2_top    = 'mirror'

@af.broadcast
def f_left(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    
    contact_width = 0.4 #TODO : Do not hardcode
    contact_start = -11.85 - contact_width/2
    contact_end   = -11.85 + contact_width/2 

    if (params.source_type == 'AC'):
        vel_drift_y_in = params.vel_drift_y_in/contact_width * np.sin(omega*t)
    elif (params.source_type == 'DC'):
        vel_drift_y_in = params.vel_drift_y_in/contact_width
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

    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_y_in*p_x - mu)/(k*T) ) + 1.)
                     ) 
    if params.zero_temperature:
        fermi_dirac_in = fermi_dirac_in - 0.5

    y_contact_start = contact_start#params.contact_start
    y_contact_end   = contact_end#params.contact_end
    
    cond = ((params.y >= y_contact_start) & \
            (params.y <= y_contact_end) \
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

    contact_start = -9.85
    contact_end   = 11.25
    contact_width = contact_end - contact_start

    vel_drift_y_out = params.vel_drift_y_out/contact_width
    
    if (params.p_space_grid == 'cartesian'):
        p_x = p1 
        p_y = p2
    elif (params.p_space_grid == 'polar2D'):
        p_x = p1 * af.cos(p2)
        p_y = p1 * af.sin(p2)
    else:
        raise NotImplementedError('Unsupported coordinate system in p_space')

    fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_y_out*p_x - mu)/(k*T) ) + 1.)
                      )
    if params.zero_temperature:
        fermi_dirac_out = fermi_dirac_out - 0.5
    
    if (params.contact_geometry=="straight"):
        # Contacts on either side of the device

        q2_contact_start = contact_start
        q2_contact_end   = contact_end
        
        cond = ((params.y >= q2_contact_start) & \
                (params.y <= q2_contact_end) \
               )

        f_right = cond*fermi_dirac_out + (1 - cond)*f

    elif (params.contact_geometry=="turn_around"):
        # Contacts on the same side of the device
        
        f_right = f

    af.eval(f_right)
    return(f_right)
