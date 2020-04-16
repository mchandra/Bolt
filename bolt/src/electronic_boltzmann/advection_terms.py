import numpy as np
import arrayfire as af
import coords

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

    jac = coords.jacobian_dq_dx(q1, q2)
    #jac = params.jacobian_dq_dx(q1, q2)

    dq1_dx = jac[0][0]; dq1_dy = jac[0][1]
    dq2_dx = jac[1][0]; dq2_dy = jac[1][1]

    C_q1 = C_x*dq1_dx + C_y*dq1_dy
    C_q2 = C_x*dq2_dx + C_y*dq2_dy

    g = coords.sqrt_det_g(q1, q2)
    #g = params.sqrt_det_g(q1, q2)

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
    e = params.charge_electron
    c = params.speed_of_light
    B3_mean = params.B3_mean

    v1, v2 = params.vel_band

    dp1_dt = e*(E1 + v2*B3_mean/c) # p1 = hcross * k1
    dp2_dt = e*(E2 - v1*B3_mean/c) # p2 = hcross * k2
    dp3_dt = 0.*p1

    return (dp1_dt, dp2_dt, dp3_dt)
