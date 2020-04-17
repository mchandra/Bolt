import numpy as np
import arrayfire as af

import domain_2 as domain

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

def get_cartesian_coords(q1, q2):
    
    #x = q1
    #y = q2*(1 + q1)
    weight = 0.5*(1+af.tanh(q2))

    #x = (1. - weight)*q1 + weight*q1*(1 - 0.5*q2)
    x = q1*(1 - 0.5*q2)
    #x = q1*(2 + q2)
    y = q2

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
