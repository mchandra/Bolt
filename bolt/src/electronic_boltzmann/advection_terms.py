import numpy as np
import arrayfire as af

#from bolt.lib.utils.coord_transformation \
#    import jacobian_dq_dx, sqrt_det_g

"""
Here we define the advection terms for the 
nonrelativistic Boltzmann equation.

The equation that we are solving is:
  df/dt + v_x * df/dq1 + v_y * df/dy 
+ e/m * (E + v X B)_x * df/dv_x 
+ e/m * (E + v X B)_y * df/dv_y 
+ e/m * (E + v X B)_y * df/dv_z = 0
      
In the solver framework this can be described using:
  q1 = x  ; q2 = y
  p1 = v1 = v_x; p2 = v2 = v_y; p3 = v3 = v_z
  A_q1 = C_q1 = v_x = v1
  A_q2 = C_q2 = v_y = v2
  A_v1 = C_v1 = e/m * (E_x + v_y * B_z - v_z * B_y) = e/m * (E1 + v2 * B3 - v3 * B2)
  A_v2 = C_v2 = e/m * (E_y + v_z * B_x - v_x * B_z) = e/m * (E2 + v3 * B1 - v1 * B3)
  A_v3 = C_v3 = e/m * (E_z + v_x * B_y - v_y * B_x) = e/m * (E3 + v1 * B2 - v2 * B1)

"""

def A_q(t, q1, q2, p1, p2, p3, params):
    """
    Return the terms A_q1, A_q2.

    Parameters:
    -----------
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function
    """
    
    A_q1, A_q2 = params.vel_band

    return (A_q1, A_q2)

def C_q(t, q1, q2, p1, p2, p3, params):
    """
    Return the terms C_q1, C_q2.

    Parameters:
    -----------
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function

    """
    C_x, C_y = params.vel_band
    
#    X = q1; Y = q2
#
#    x = X
#    
#    #TODO : Remove from here
#    a = 0.3
#    k = np.pi
#
#    dX_dx = 1.
#    dX_dy = 0.
#
#    dY_dx = a*k*af.cos(k*x)
#    dY_dy = 1.
#    
#    C_X   = dX_dx*C_x + dX_dy*C_y
#    C_Y   = dY_dx*C_x + dY_dy*C_y

    jac = [[params.dq1_dx, params.dq1_dy], [params.dq2_dx, params.dq2_dy]]


    dq1_dx = jac[0][0]; dq1_dy = jac[0][1]
    dq2_dx = jac[1][0]; dq2_dy = jac[1][1]

    C_q1 = C_x*dq1_dx + C_y*dq1_dy
    C_q2 = C_x*dq2_dx + C_y*dq2_dy

    g = params.sqrt_det_g

    return (g*C_q1, g*C_q2)

# This can then be called inside A_p if needed:
# F1 = (params.char....)(E1 + ....) + T1(q1, q2, p1, p2, p3)

def A_p(t, q1, q2, p1, p2, p3,
        E1, E2, E3, B1, B2, B3,
        params
       ):
    """
    Return the terms A_v1, A_v2 and A_v3.

    Parameters:
    -----------
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    fields_solver: The solver object whose method get_fields() is used to 
                   obtain the EM field quantities

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function
    """
    e = params.charge_electron
    c = params.speed_of_light
    B3_mean = params.B3_mean

    v1, v2 = params.vel_band

    dp1_dt = e*(E1 + v2*B3_mean/c) # p1 = hcross * k1
    dp2_dt = e*(E2 - v1*B3_mean/c) # p2 = hcross * k2
    dp3_dt = 0.*p1

    return (dp1_dt, dp2_dt, dp3_dt)

def C_p(t, q1, q2, p1, p2, p3,
        E1, E2, E3, B1, B2, B3,
        params
       ):
    """
    Return the terms C_v1, C_v2 and C_v3.

    Parameters:
    -----------
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    v1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    v3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    fields_solver: The solver object whose method get_fields() is used to 
                   obtain the EM field quantities

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function
    """
    
    v_x, v_y   = params.vel_band
    

    # Magnetotransport requires a momentum space grid that is aligned with the Fermi surface 

    if (params.p_space_grid == 'polar2D' and params.p_dim == 1):
        # Coefficient of df/dtheta term in the Boltzmann equation, C = e B v_F/p_F
        # Writing p_F = m v_F, we get C = e B/m
        # We call omega_c = e B/m
        # Using tau_c = 1/omega_c and l_c = v_F*tau_c,
        # we get C = 1/tau_c = v_F/l_c
        # Writing the Boltzmann equation : 
        # (1/v_F) df/dt + \hat{p}.df/dx - (e B v_F/p_F) df/dtheta = [-(f-f0^MR)/l_mr] + [-(f - f0^MC)/l_mc]
        # The Boltzmann equation then becomes : 
        # (1/v_F) df/dt + \hat{p}.df/dx - (v_F/l_c)     df/dtheta = [-(f-f0^MR)/l_mr] + [-(f - f0^MC)/l_mc]
        dp1_dt = 0.*p1*q1
        dp2_dt = params.fermi_velocity/params.l_c + 0.*q1*p1
        dp3_dt = 0.*p1*q1
    
    elif (params.p_space_grid == 'polar2D' and params.p_dim == 2):

        # The equation being solved is 
        # df/dt + v_x df/dx + v_y df/dy + F_p_r df/dp_r + (1/p_r)*F_p_theta df/dp_theta + F_z df/dp_z = C[f]
        
        # Considering an external magnetic field in the z-place : B = B_z
        # F_p_r     = -e v_p_theta B_z
        # F_p_theta =  e v_p_r B_z
        # F_z       =  0
        
        # Substituting e B_z/m = omega_c, and using the relation omega_c = 1/tau_c = v_F/l_c, we get
        # F_p_r     = -omega_c m v_p_theta = -v_F m v_p_theta/l_c = -p_F v_p_theta/l_c (Using p_F = m v_F)
        # F_p_theta =  omega_c m v_p_r     =  v_F m v_p_r/l_c     =  p_F v_p_r/l_c
        
        # Thus the equation being solved can be written as : 
        # df/dt + v_x df/dx + v_y df/dy - (p_F v_p_theta/l_c) df/dp_r + (1/p_r)*(p_F v_p_r/l_c) df/dp_theta = C[f] 

        p_r     = p1
        p_theta = p2
        p_F     = params.fermi_momentum_magnitude(p_theta)

        if params.fermi_surface_shape == 'circle':

            # TODO : Interface with, instead of bypassing band_vel and effective_mass

            dp1_dt =  0.*p1*q1 # Because v_p_theta is zero for a circular fermi surface

            if params.dispersion == 'linear' : 
                dp2_dt = (params.fermi_velocity/params.l_c) * (p_F/p_r) + 0.*p1*q1

            elif params.dispersion == 'quadratic' :
                dp2_dt =  params.fermi_velocity/params.l_c + 0.*p1*q1

        else : 
            raise NotImplementedError('Unsupported shape of fermi surface for magnetotansport')
            # TODO : Magnetotransport for arbitrary Fermi surface shapes.
            # This problem is the same as aligning the grid in real space to fit the device geometry

        dp3_dt =  0.*p1*q1

    elif (params.p_space_grid == 'cartesian'):

        # The equation being solved is 
        # df/dt + v_x df/dx + v_y df/dy + F_x df/dp_x + F_y df/dp_y + F_z df/dp_z = C[f]
        
        # Considering an external magnetic field in the z-place : B = B_z
        # F_x = -e v_y B_z
        # F_y =  e v_x B_z
        # F_z = 0
        
        # Substituting e B_z/m = omega_c, and using the relation omega_c = 1/tau_c = v_F/l_c, we get
        # F_x = -omega_c m v_y = -v_F m v_y/l_c = -p_F v_y/l_c (Using p_F = m v_F)
        # F_y =  omega_c m v_x =  v_F m v_x/l_c =  p_F v_x/l_c
        
        # Thus the equation being solved can be written as : 
        # df/dt + v_x df/dx + v_y df/dy - (p_F v_y/l_c) df/dp_x + (p_F v_x/l_c) df/dp_y = C[f]

        p_x = p1
        p_y = p2

        if params.dispersion == 'quadratic':

            dp1_dt = -p_y*params.fermi_velocity/params.l_c + 0.*q1*p1 # p1 = hcross * k1
            dp2_dt =  p_x*params.fermi_velocity/params.l_c + 0.*q1*p1 # p2 = hcross * k2
            dp3_dt =  0.*p1*q1
        
        else : 
            raise NotImplementedError('Cartesian coordinates in momentum space cannot be used with linear dispersion for magnetotransport')


    return (dp1_dt, dp2_dt, dp3_dt)
