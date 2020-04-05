import numpy as np
import arrayfire as af

def get_cartesian_coords(q1, q2):
    
    # g = 1 for the following transformation
    #a = 0.3; k = np.pi
    #x = q1
    #y = q2 - a*af.sin(k*q1)

    a = 0.1; k = np.pi
    x = q1 + a*af.cos(k*q2)
    y = q2 - a*af.sin(k*q1)

    #a = 2; r = 1.1
    #x = q1*(1 +  a*af.sqrt(r**2. - q2**2.))
    #y = q2

    return(x, y)

def jacobian_dx_dq(q1, q2):
    
    # TODO: evaluate this numerically using get_cartesian_coords

    a = 0.1; k = np.pi
    dx_dq1 = 1.;                dx_dq2 = -a*k*af.sin(k*q2)
    dy_dq1 = -a*k*af.cos(k*q1); dy_dq2 = 1.
#    a = 2; r = 1.1
#    dx_dq1 = 1. + a*af.sqrt(r**2. - q2**2.)
#
#    dx_dq2 = -a*q1*q2/af.sqrt(r**2. - q2**2.)
#
#    dy_dq1 = 0.
#
#    dy_dq2 = 1.

    return([[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]])

def jacobian_dq_dx(q1, q2):

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
