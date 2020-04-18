import numpy as np
import arrayfire as af

import domain

def get_theta(q1, q2, boundary):

    [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]] = jacobian_dx_dq(q1, q2)
    
    dq1 = (domain.q1_end - domain.q1_start)/domain.N_q1
    dq2 = (domain.q2_end - domain.q2_start)/domain.N_q2

    if ((boundary == "left") or (boundary == "right")):
        dq1 = 0
    if ((boundary == "top") or (boundary == "bottom")):
        dq2 = 0

    dy = dy_dq1*dq1 + dy_dq2*dq2
    dx = dx_dq1*dq1 + dx_dq2*dq2
    dy_dx = dy/dx

    return (af.atan(dy_dx))

def get_cartesian_coords_for_post(q1, q2):

    X = q1; Y = q2

    x = X; y = Y

    indices_1     =  X < -4.62
    indices_2     = (X < 0    ) * (X > -4.62)
    indices_4     = (X > 26.3 ) * (X < 29.46) * (Y > 12)
    indices_5_top = (X > 29.46) * (X < 32.98) * (Y > 12)
    indices_5_bot = (X > 29.46) * (X < 32.98) * (Y < 12)
    indices_6_top = (X > 32.98)               * (Y > 12) 
    indices_6_bot = (X > 32.98)               * (Y < 12)

    if (af.any_true(indices_1)):
        y[indices_1]     = 0.816*Y[indices_1]
    
    if (af.any_true(indices_2)):
        y[indices_2]     = (Y *(1 + 0.04*(X)))[indices_2]
    
    if (af.any_true(indices_4)):
        y[indices_4]     = ((Y-12) *(1 - 2*0.0451*(X-26.3)))[indices_4] + 12
    
    if (af.any_true(indices_5_top)):
        y[indices_5_top] = ((Y-12) *(1 - 2*0.0451*(X-26.3)))[indices_5_top] + 12
    
    if (af.any_true(indices_5_bot)):
        y[indices_5_bot] = ((Y-12) *(1 - 0.1193*(X-29.46)))[indices_5_bot] + 12
    
    if (af.any_true(indices_6_top)):
        y[indices_6_top] = 0.40*(Y[indices_6_top]-12) + 12
    
    if (af.any_true(indices_6_bot)):
        y[indices_6_bot] = 0.58*(Y[indices_6_bot]-12) + 12
    
    return(x, y)


def get_cartesian_coords(q1, q2):

    q1_midpoint = 0.5*(af.max(q1) + af.min(q1))
    q2_midpoint = 0.5*(af.max(q2) + af.min(q2))

    x = q1

    if (q1_midpoint < -4.62): # Domain 1 and 7
        y = 0.816*q2

    elif ((q1_midpoint > -4.62) and (q1_midpoint < 0)): # Domain 2 and 8
        y = (q2 *(1 + 0.04*(q1)))

    elif ((q1_midpoint > 0) and (q1_midpoint < 26.3)): # Domain 3 and 9
        y = q2

    elif ((q1_midpoint > 26.3) and (q1_midpoint < 29.46) and (q2_midpoint < 12)): # Domain 4
        y = q2

    elif ((q1_midpoint > 29.46) and (q1_midpoint < 32.98) and (q2_midpoint < 12)): # Domain 5
        y = ((q2-12) *(1 - 0.1193*(q1-29.46))) + 12

    elif ((q1_midpoint > 32.98) and (q2_midpoint < 12)): # Domain 6
        y = 0.58*(q2-12) + 12

    elif ((q1_midpoint > 26.3) and (q1_midpoint < 29.46) and (q2_midpoint > 12)): # Domain 10
        y = ((q2-12) *(1 - 2*0.0451*(q1-26.3))) + 12

    elif ((q1_midpoint > 29.46) and (q1_midpoint < 32.98) and (q2_midpoint > 12)): # Domain 11
        y = ((q2-12) *(1 - 2*0.0451*(q1-26.3))) + 12

    elif ((q1_midpoint > 32.98) and (q2_midpoint > 12)):  # Domain 12
        y = 0.40*(q2-12) + 12

    else:
        raise NotImplementedError('q1_center/q2_center out of bounds!') 
        

    return(x, y)

def jacobian_dx_dq(q1, q2):
    
    eps = 1e-7 # small parameter needed for numerical differentiation. Can't be too small though!
    x, y                         = get_cartesian_coords(q1,     q2    )
    x_plus_eps_q1, y_plus_eps_q1 = get_cartesian_coords(q1+eps, q2    )
    x_plus_eps_q2, y_plus_eps_q2 = get_cartesian_coords(q1,     q2+eps)

    dx_dq1 = (x_plus_eps_q1 - x)/eps; dy_dq1 = (y_plus_eps_q1 - y)/eps
    dx_dq2 = (x_plus_eps_q2 - x)/eps; dy_dq2 = (y_plus_eps_q2 - y)/eps

    return([[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]])

def jacobian_dq_dx(q1, q2):

    """ Returns dq/dx in a 2x2 array

    Usage :         
    [[dq1_dx, dq1_dy], [dq2_dx, dq2_dy]] = jacobian_dq_dx(q1, q2)

    """

    A = jacobian_dx_dq(q1, q2)

    a = A[0][0]; b = A[0][1]
    c = A[1][0]; d = A[1][1]

    det_A = a*d - b*c

    inv_A = [[0, 0], [0, 0]]

    inv_A[0][0] =  d/det_A; inv_A[0][1] = -b/det_A
    inv_A[1][0] = -c/det_A; inv_A[1][1] =  a/det_A

    return(inv_A)

def sqrt_det_g(q1, q2):

    jac = jacobian_dx_dq(q1, q2)
    
    dx_dq1 = jac[0][0]; dx_dq2 = jac[0][1]
    dy_dq1 = jac[1][0]; dy_dq2 = jac[1][1]

    g_11 = (dx_dq1)**2. + (dy_dq1)**2.

    g_12 = dx_dq1*dx_dq2 + dy_dq1*dy_dq2

    g_21 = g_12

    g_22 = (dx_dq2)**2. + (dy_dq2)**2.

    det_g = g_11*g_22 - g_12*g_21

    return(af.sqrt(det_g))
