import numpy as np
import arrayfire as af
import domain

in_q1_left   = 'mirror+dirichlet'
in_q1_right  = 'mirror+dirichlet'
in_q2_bottom = 'mirror'
in_q2_top    = 'mirror+dirichlet'

@af.broadcast
def f_left(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    N_g = domain.N_ghost
   
    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_in  = params.vel_drift_x_in
    vel_drift_x_out = params.vel_drift_y*0.0

    N_g = domain.N_ghost
    for index in range(N_g):
        vel_drift_x_out[:, :, index, :] = params.vel_drift_x[:, :, N_g, :]

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
    fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_x_out*p_x - mu)/(k*T) ) + 1.)
                     )
    
    if params.zero_temperature:
        fermi_dirac_in = fermi_dirac_in - 0.5
    if params.zero_temperature:
        fermi_dirac_out = fermi_dirac_out - 0.5

    if (params.contact_geometry=="straight"):
        # Contacts on either side of the device

        q2_contact_start = params.contact_start
        q2_contact_end   = params.contact_end
        
        cond = ((params.y >= q2_contact_start) & \
                (params.y <= q2_contact_end) \
               )
        cond_base = (params.y >= q2_contact_end)

        f_left = cond*fermi_dirac_in + cond_base*fermi_dirac_out + (1-cond_base)*(1 - cond)*f

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
def f_right(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    N_g = domain.N_ghost

    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_out = params.vel_drift_x*0.0

    N_g = domain.N_ghost
    for index in range(N_g):
        vel_drift_x_out[:, :, -index-1, :] = params.vel_drift_x[:, :, -N_g-1, :]

    
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
#    fermi_dirac_out = (1./(af.exp( (E_upper - params.vel_drift_x_in*p_x - params.mu)/(k*T) ) + 1.)
#                      )
    if params.zero_temperature:
        fermi_dirac_out = fermi_dirac_out - 0.5
    
    if (params.contact_geometry=="straight"):
        # Contacts on either side of the device

        q2_contact_start = 0.#params.contact_start
        q2_contact_end   = 0.25#params.contact_end
        
        cond = ((params.y >= q2_contact_start) & \
                (params.y <= q2_contact_end) \
               )

        f_right = cond*fermi_dirac_out + (1 - cond)*f

    elif (params.contact_geometry=="turn_around"):
        # Contacts on the same side of the device
        
        f_right = f

    af.eval(f_right)
    return(f_right)


@af.broadcast
def f_top(f, t, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    N_g = domain.N_ghost

    t     = params.current_time
    omega = 2. * np.pi * params.AC_freq
    vel_drift_x_out = params.vel_drift_y*0.0

    N_g = domain.N_ghost
    for index in range(N_g):
        vel_drift_x_out[:, :, :, -index-1] = params.vel_drift_y[:, :, :, -N_g-1]

    
    if (params.p_space_grid == 'cartesian'):
        p_x = p1 
        p_y = p2
    elif (params.p_space_grid == 'polar2D'):
        p_x = p1 * af.cos(p2)
        p_y = p1 * af.sin(p2)
    else:
        raise NotImplementedError('Unsupported coordinate system in p_space')

    fermi_dirac_out = (1./(af.exp( (E_upper - vel_drift_x_out*p_y - mu)/(k*T) ) + 1.)
                      )
    #fermi_dirac_out = (1./(af.exp( (E_upper - params.vel_drift_x_in*p_y - params.mu)/(k*T) ) + 1.)
    #                  )
    
    if params.zero_temperature:
        fermi_dirac_out = fermi_dirac_out - 0.5
    
    
    q2_contact_start = 0.75#params.contact_start
    q2_contact_end   = 1.25#params.contact_end
    
    cond = ((params.x >= q2_contact_start) & \
            (params.x <= q2_contact_end) \
           )

    f_right = cond*fermi_dirac_out + (1 - cond)*f


    af.eval(f_right)
    return(f_right)

