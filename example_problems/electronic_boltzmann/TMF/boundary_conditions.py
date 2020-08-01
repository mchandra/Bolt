import numpy as np
import arrayfire as af
import domain

in_q1_left   = 'mirror'
in_q1_right  = 'mirror'
in_q2_bottom = 'mirror+dirichlet'
in_q2_top    = 'mirror+dirichlet'

@af.broadcast
def f_bottom(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    
    contact_width = 0.4 #TODO : Do not hardcode
    contact_start = -19.6 - contact_width/2
    contact_end   = -19.6 + contact_width/2 

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

    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_y_in*p_y - mu)/(k*T) ) + 1.)
                     ) 
    if params.zero_temperature:
        fermi_dirac_in = fermi_dirac_in - 0.5

    x_contact_start = contact_start#params.contact_start
    x_contact_end   = contact_end#params.contact_end
    
    cond = ((params.x >= x_contact_start) & \
            (params.x <= x_contact_end) \
           )

    f_bottom = cond*fermi_dirac_in + (1 - cond)*f

    af.eval(f_bottom)
    return(f_bottom)



@af.broadcast
def f_top(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq

    contact_width = 0.4 #TODO : Do not hardcode
    contact_start = -12.8 - contact_width/2
    contact_end   = -12.8 + contact_width/2 

    vel_drift_y_out = params.vel_drift_y_out/contact_width
    
    if (params.p_space_grid == 'cartesian'):
        p_x = p1 
        p_y = p2
    elif (params.p_space_grid == 'polar2D'):
        p_x = p1 * af.cos(p2)
        p_y = p1 * af.sin(p2)
    else:
        raise NotImplementedError('Unsupported coordinate system in p_space')

    fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_y_out*p_y - mu)/(k*T) ) + 1.)
                      )
    if params.zero_temperature:
        fermi_dirac_out = fermi_dirac_out - 0.5
    
    if (params.contact_geometry=="straight"):
        # Contacts on either side of the device

        q1_contact_start = contact_start
        q1_contact_end   = contact_end
        
        cond = ((params.x >= q1_contact_start) & \
                (params.x <= q1_contact_end) \
               )

        f_top = cond*fermi_dirac_out + (1 - cond)*f

    elif (params.contact_geometry=="turn_around"):
        # Contacts on the same side of the device
        
        f_top = f

    af.eval(f_top)
    return(f_top)
