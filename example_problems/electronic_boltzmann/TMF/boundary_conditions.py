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
    

    x_contact_start = contact_start#params.contact_start
    x_contact_end   = contact_end#params.contact_end
    
    cond = ((params.x >= x_contact_start) & \
            (params.x <= x_contact_end) \
           )

    f_bottom = cond*fermi_dirac_in + (1 - cond)*f

    af.eval(f_bottom)
    return(f_bottom)

#@af.broadcast
#def f_bottom(f, t, q1, q2, p1, p2, p3, params):
#
#    k       = params.boltzmann_constant
#    E_upper = params.E_band
#    T       = params.initial_temperature
#    mu      = params.initial_mu
#    
#    t     = params.current_time
#    omega = 2. * np.pi * params.AC_freq
#    vel_drift_x_in  = params.vel_drift_x_in
#
#    if (params.p_space_grid == 'cartesian'):
#        p_x = p1 
#        p_y = p2
#    elif (params.p_space_grid == 'polar2D'):
#        p_x = p1 * af.cos(p2)
#        p_y = p1 * af.sin(p2)
#    else:
#        raise NotImplementedError('Unsupported coordinate system in p_space')
#
#
#    fermi_dirac_in = (1./(af.exp( (E_upper - vel_drift_x_in*p_y - mu)/(k*T) ) + 1.)
#                     )
#    # TODO : Testing - set zero everywhere except index N_p2/2 (injection towards right)
##    fermi_dirac_in = vel_drift_x_in*p_x
#
##    print (p_x)
##    fermi_dirac_in[:int(3*domain.N_p2/8)]   = 0.
##    fermi_dirac_in[int(3*domain.N_p2/8)+1:] = 0.
#
#    if (params.contact_geometry=="straight"):
#        # Contacts on either side of the device
#
#        q2_contact_start = params.contact_start
#        q2_contact_end   = params.contact_end
#        
#        cond = ((params.y >= q2_contact_start) & \
#                (params.y <= q2_contact_end) \
#               )
#
#        f_left = cond*fermi_dirac_in + (1 - cond)*f
#
#    elif (params.contact_geometry=="turn_around"):
#        # Contacts on the same side of the device
#
#        vel_drift_x_out = -params.vel_drift_x_in * np.sin(omega*t)
#
#        fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_x_out*p_x - mu)/(k*T) ) + 1.)
#                          )
#    
#        # TODO: set these parameters in params.py
#        cond_in  = ((q2 >= 3.5) & (q2 <= 4.5))
#        cond_out = ((q2 >= 5.5) & (q2 <= 6.5))
#    
#        f_left =  cond_in*fermi_dirac_in + cond_out*fermi_dirac_out \
#                + (1 - cond_in)*(1 - cond_out)*f
#
#    af.eval(f_left)
#    return(f_left)


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
#    fermi_dirac_out = 0.*vel_drift_x_out*p_x
    
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
