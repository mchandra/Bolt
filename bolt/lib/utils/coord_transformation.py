import numpy as np
import arrayfire as af

import domain
import coords
import params


def get_theta(q1, q2, boundary,  q1_start_local_left=None, q2_start_local_bottom=None):

    [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]] = jacobian_dx_dq(q1, q2,
                                                          q1_start_local_left, 
                                                          q2_start_local_bottom)
    
    dq1 = (domain.q1_end - domain.q1_start)/domain.N_q1
    dq2 = (domain.q2_end - domain.q2_start)/domain.N_q2

    if ((boundary == "left") or (boundary == "right")):
        dq1 = 0
    if ((boundary == "top") or (boundary == "bottom")):
        dq2 = 0

    dy = dy_dq1*dq1 + dy_dq2*dq2
    dx = dx_dq1*dq1 + dx_dq2*dq2

    left_edge = 0
    #print ("Rank : ", params.rank, ", coordinate_transformation.py, any dx is zero : ", af.any_true(af.iszero(dx[0, 0, left_edge, :])))
    #print ("Rank : ", params.rank, ", coordinate_transformation.py, all dx is zero : ", af.all_true(af.iszero(dx[0, 0, left_edge, :])))
    #print ("Rank : ", params.rank, ", coordinate_transformation.py, all dy is zero : ", af.all_true(af.iszero(dy)))

    # TODO : Testing



    dy_dx = dy/dx
    #print ("Rank = ", params.rank, ", dy_dx : ", dy_dx.dims())

    return (af.atan(dy_dx))

def jacobian_dx_dq(q1, q2, q1_start_local_left=None, q2_start_local_bottom=None):
   
    x, y, jacobian = coords.get_cartesian_coords(q1, q2, q1_start_local_left, q2_start_local_bottom, return_jacobian=True)
    
    if (jacobian==None):
        # No analytic jacobian. Proceed to compute it numerically: 
        eps = 1e-7 # small parameter needed for numerical differentiation. Can't be too small though!
        x, y                         = coords.get_cartesian_coords(q1,     q2    , q1_start_local_left, q2_start_local_bottom)
        x_plus_eps_q1, y_plus_eps_q1 = coords.get_cartesian_coords(q1+eps, q2    , q1_start_local_left, q2_start_local_bottom)
        x_plus_eps_q2, y_plus_eps_q2 = coords.get_cartesian_coords(q1,     q2+eps, q1_start_local_left, q2_start_local_bottom)

        dx_dq1 = (x_plus_eps_q1 - x)/eps; dy_dq1 = (y_plus_eps_q1 - y)/eps
        dx_dq2 = (x_plus_eps_q2 - x)/eps; dy_dq2 = (y_plus_eps_q2 - y)/eps

#    print ("coordinate_transformation.py, dx_dq1 : ", dx_dq1)
#    print ("coordinate_transformation.py, dy_dq1 : ", dy_dq1)
#    print ("coordinate_transformation.py, dx_dq2 : ", dx_dq2)
#    print ("coordinate_transformation.py, dy_dq2 : ", dy_dq2)
        
        jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]

    return(jacobian)

def jacobian_dq_dx(q1, q2, q1_start_local_left=None, q2_start_local_bottom=None):

    """ Returns dq/dx in a 2x2 array

    Usage :         
    [[dq1_dx, dq1_dy], [dq2_dx, dq2_dy]] = jacobian_dq_dx(q1, q2)

    """

    A = jacobian_dx_dq(q1, q2, q1_start_local_left, q2_start_local_bottom)

    a = A[0][0]; b = A[0][1]
    c = A[1][0]; d = A[1][1]

    det_A = a*d - b*c

    inv_A = [[0, 0], [0, 0]]

    inv_A[0][0] =  d/det_A; inv_A[0][1] = -b/det_A
    inv_A[1][0] = -c/det_A; inv_A[1][1] =  a/det_A

    return(inv_A)

def sqrt_det_g(q1, q2, q1_start_local_left=None, q2_start_local_bottom=None):

    jac = jacobian_dx_dq(q1, q2, q1_start_local_left, q2_start_local_bottom)
    
    dx_dq1 = jac[0][0]; dx_dq2 = jac[0][1]
    dy_dq1 = jac[1][0]; dy_dq2 = jac[1][1]

    g_11 = (dx_dq1)**2. + (dy_dq1)**2.

    g_12 = dx_dq1*dx_dq2 + dy_dq1*dy_dq2

    g_21 = g_12

    g_22 = (dx_dq2)**2. + (dy_dq2)**2.

    det_g = g_11*g_22 - g_12*g_21

    return(af.sqrt(det_g))

def compute_shift_indices(q1, q2, p1, p2, p3, params):
    """
    Inject the shift_indices corresponding to a shift operation of -2*theta for the left boundary,
    where theta is the angular variation of the left boundary.
    """
    N_theta       = domain.N_p2  # TODO run this and check
    N_g           = domain.N_ghost

    # Define edge indices
    left_edge   = N_g; right_edge = -N_g-1
    bottom_edge = N_g; top_edge   = -N_g-1
    
    temp = af.range(N_theta) # Indices with no shift. Shape : N_theta


    # Left boundary

    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, 0, :]
    N_q2_local    = shift_indices.dims()[3]

    # Get the angular variation of the left boundary.
    theta_left = get_theta(q1, q2, "left", \
                           q1_start_local_left=params.q1_start_local_left, \
                           q2_start_local_bottom=params.q2_start_local_bottom)[0, 0, left_edge, :]
    print ("Rank = ", params.rank, ", theta_left : ", af.any_true(af.isnan(theta_left)))

    theta_left = af.moddims(theta_left, N_q2_local) # Convert to 1D array

    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_left
    shifts  = -((2*theta_left)/(2*np.pi))*N_theta 
#    print ("coordinate_transformation, shifts : ", shifts, " Rank : ", params.rank)

    # Populate shift_indices 2D array using shifts.
    for index, value in enumerate(shifts):
        shift_indices[:, 0, 0, index] = af.shift(N_theta*index+temp, int(value.scalar()))

    # Convert into a 1D array
    shift_indices_left = af.moddims(shift_indices, N_theta*N_q2_local)
   
    # Right boundary 

    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, 0, :] # Shape : N_theta x 1 x 1 x N_q2+2*N_g

    # Get the angular variation of the right boundary.
    theta_right = get_theta(q1, q2, "right", \
                            q1_start_local_left=params.q1_start_local_left, \
                            q2_start_local_bottom=params.q2_start_local_bottom)[0, 0, right_edge, :]
    theta_right = af.moddims(theta_right, N_q2_local) # Convert to 1D array
    print ("Rank = ", params.rank, ", theta_right : ", af.any_true(af.isnan(theta_right)))

    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_right
    shifts  = -(theta_right/np.pi)*N_theta


    # Populate shift_indices 2D array using shifts.
    for index, value in enumerate(shifts):
        shift_indices[:, 0, 0, index] = af.shift(N_theta*index+temp, int(value.scalar()))

    # Convert into a 1D array
    shift_indices_right = af.moddims(shift_indices, N_theta*N_q2_local)

    
    # Bottom boundary
    
    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, :, 0] # Shape : N_theta x 1 x  N_q1+2*N_g x 1
    N_q1_local    = shift_indices.dims()[2]

    # Get the angular variation of the bottom boundary.
    theta_bottom = get_theta(q1, q2, "bottom", \
                             q1_start_local_left=params.q1_start_local_left, \
                             q2_start_local_bottom=params.q2_start_local_bottom)[0, 0, :, bottom_edge]
    theta_bottom = af.moddims(theta_bottom, N_q1_local) # Convert to 1D array
    print ("Rank = ", params.rank, ", theta_bottom : ", af.any_true(af.isnan(theta_bottom)))

    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_bottom
    shifts  = -(theta_bottom/np.pi)*N_theta

    # Populate shift_indices 2D array using shifts.
    for index, value in enumerate(shifts):
        shift_indices[:, 0, index, 0] = af.shift(N_theta*index+temp, int(value.scalar()))

    # Convert into a 1D array
    shift_indices_bottom = af.moddims(shift_indices, N_theta*N_q1_local)


    # Top Boundary

    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, :, 0] # Shape : N_theta x 1 x N_q1+2*N_g x 1
    
    # Get the angular variation of the top boundary.
    theta_top = get_theta(q1, q2, "top", \
                          q1_start_local_left=params.q1_start_local_left, \
                          q2_start_local_bottom=params.q2_start_local_bottom)[0, 0, :, top_edge]
    theta_top = af.moddims(theta_top, N_q1_local) # Convert to 1D array
    print ("Rank = ", params.rank, ", theta_top : ", af.any_true(af.isnan(theta_top)))
 
    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_top
    shifts  = -(theta_top/np.pi)*N_theta


    # Populate shift_indices 2D array using shifts.
    for index, value in enumerate(shifts):
        shift_indices[:, 0, index, 0] = af.shift(N_theta*index+temp, int(value.scalar()))

    #Convert to a 1D array
    shift_indices_top = af.moddims(shift_indices, N_theta*N_q1_local)
    
    return(shift_indices_left, shift_indices_right, shift_indices_bottom, shift_indices_top)


def affine(q1, q2,
           x_y_bottom_left, x_y_bottom_right, 
           x_y_top_right, x_y_top_left,
           X_Y_bottom_left, X_Y_bottom_right, 
           X_Y_top_right, X_Y_top_left,
          ):

    '''
        Inputs :
            - Grid in q1 and q2
            - coordinates of 4 points on the original grid (x, y)
            - coordinates of the corresponding 4 points on the desired transformed grid (X, Y)
        Output : Transformed grid
    '''
    
    x0, y0 = x_y_bottom_left;  X0, Y0 = X_Y_bottom_left
    x1, y1 = x_y_bottom_right; X1, Y1 = X_Y_bottom_right
    x2, y2 = x_y_top_right;    X2, Y2 = X_Y_top_right
    x3, y3 = x_y_top_left;     X3, Y3 = X_Y_top_left

    
    #x = a0 + a1*X + a2*Y + a3*X*Y
    #y = b0 + b1*X + b2*Y + b3*X*Y
    
    #x0 = a0 + a1*X0 + a2*Y0 + a3*X0*Y0
    #x1 = a0 + a1*X1 + a2*Y1 + a3*X1*Y1
    #x2 = a0 + a1*X2 + a2*Y2 + a3*X2*Y2
    #x3 = a0 + a1*X3 + a2*Y3 + a3*X3*Y3
    
    # A x = b
    A = np.array([[1, X0, Y0, X0*Y0],
                  [1, X1, Y1, X1*Y1],
                  [1, X2, Y2, X2*Y2],
                  [1, X3, Y3, X3*Y3],
                 ])
    b = np.array([[x0],
                  [x1],
                  [x2],
                  [x3]
                 ])

    a0, a1, a2, a3 = np.linalg.solve(A, b)
    
    #y0 = b0 + b1*X0 + b2*Y0 + b3*X0*Y0
    #y1 = b0 + b1*X1 + b2*Y1 + b3*X1*Y1
    #y2 = b0 + b1*X2 + b2*Y2 + b3*X2*Y2
    #y3 = b0 + b1*X3 + b2*Y3 + b3*X3*Y3

    b = np.array([[y0],
                  [y1],
                  [y2],
                  [y3]
                 ])

    b0, b1, b2, b3 = np.linalg.solve(A, b)
    
    x = a0 + a1*q1 + a2*q2 + a3*q1*q2
    y = b0 + b1*q1 + b2*q2 + b3*q1*q2
    
    return(x, y)
