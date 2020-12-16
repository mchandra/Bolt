import numpy as np
import arrayfire as af

import domain
import params


def get_theta(q1, q2, boundary,  q1_start_local_left=None, q2_start_local_bottom=None):
 
    dq1 = domain.dq1
    dq2 = domain.dq2

    q1_tmp = q1.copy()
    q2_tmp = q2.copy()

    # Assuming q1, q2 are at zone centers. Need to calculate thetas using the face-centers on which the boundary lies
    if (boundary == "left"):
        q1_tmp  = q1_tmp - 0.5*dq1
        dq1 = 0        

    if (boundary == "right"):
        q1_tmp  = q1_tmp + 0.5*dq1
        dq1 = 0

    if (boundary == "top"):
        q2_tmp  = q2_tmp + 0.5*dq2
        dq2 = 0

    if (boundary == "bottom"):
        q2_tmp  = q2_tmp - 0.5*dq2
        dq2 = 0

    [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]] = jacobian_dx_dq(q1_tmp, q2_tmp, q1_start_local_left, q2_start_local_bottom)

    dy = dy_dq1*dq1 + dy_dq2*dq2
    dx = dx_dq1*dq1 + dx_dq2*dq2

    dy_dx = dy/dx

    return (af.atan(dy_dx))

def jacobian_dx_dq(q1, q2, q1_start_local_left=None, q2_start_local_bottom=None):
   
    x, y, jacobian = params.get_cartesian_coords(q1, q2, q1_start_local_left, q2_start_local_bottom, return_jacobian=True)
    
    if (jacobian==None):

        raise NotImplementedError('Jacobian must be given, can not calculate numerically!')
        # TODO : Errors in numerical calculation of Jacobian are higher than the error floor of the solver.
        
        # No analytic jacobian. Proceed to compute it numerically: 
        eps = 1e-7 # small parameter needed for numerical differentiation. Can't be too small though!
        x, y                         = params.get_cartesian_coords(q1,     q2    , q1_start_local_left, q2_start_local_bottom)
        x_plus_eps_q1, y_plus_eps_q1 = params.get_cartesian_coords(q1+eps, q2    , q1_start_local_left, q2_start_local_bottom)
        x_plus_eps_q2, y_plus_eps_q2 = params.get_cartesian_coords(q1,     q2+eps, q1_start_local_left, q2_start_local_bottom)

        dx_dq1 = (x_plus_eps_q1 - x)/eps; dy_dq1 = (y_plus_eps_q1 - y)/eps
        dx_dq2 = (x_plus_eps_q2 - x)/eps; dy_dq2 = (y_plus_eps_q2 - y)/eps

        #pl.plot(af.moddims(dx[0, 0, :, 0], dx.dims()[2]).to_ndarray())
        #pl.plot(af.moddims(dy[0, 0, :, 0], dy.dims()[2]).to_ndarray())
        
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

    d_q1          = (q1[0, 0, 1, 0] - q1[0, 0, 0, 0]).scalar()
    d_q2          = (q2[0, 0, 0, 1] - q2[0, 0, 0, 0]).scalar()

    N_q1_local    = q1.dims()[2]
    N_q2_local    = q2.dims()[3]

    # Define edge indices
    left_edge   = N_g; right_edge = -N_g-1
    bottom_edge = N_g; top_edge   = -N_g-1
    
    temp = af.range(N_theta) # Indices with no shift. Shape : N_theta


    # Left boundary

    # Initialize to zero
    shift_indices = (0.*q1*p1)[:, 0, 0, :]


    # Get the angular variation of the left boundary.
    theta_left = get_theta(q1, q2, "left", \
                           q1_start_local_left=params.q1_start_local_left, \
                           q2_start_local_bottom=params.q2_start_local_bottom)[0, 0, left_edge, :]

    theta_left = af.moddims(theta_left, N_q2_local) # Convert to 1D array

    # Apply manually specified mirror angles only if left boundary of the zone is the left boundary of the device
    is_left_most_domain = abs(params.q1_start_local_left - domain.q1_start) < 1e-13

    if (params.enable_manual_mirror and is_left_most_domain):
        theta_left = 0.*theta_left + params.mirror_angles[3]


    # TODO : Dumping theta for testing
#    if (params.rank == 5):
#        np.savetxt("/home/quazartech/bolt/example_problems/electronic_boltzmann/circular_domain_method_2/theta_right.txt", theta_left[N_g:-N_g])


    # Calculate the number of shifts of the array along the p_theta axis
    # required for an angular shift of -2*theta_left
    shifts  = -((2*theta_left)/(2*np.pi))*N_theta 

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

    q1_end_local_right = params.q1_start_local_left + (N_q1_local - 2*N_g)*d_q1

    # Apply manually specified mirror angles only if right boundary of the zone is the right boundary of the device
    is_right_most_domain = abs(q1_end_local_right - domain.q1_end) < 1e-13

    if (params.enable_manual_mirror and is_right_most_domain):
        theta_right = 0.*theta_right + params.mirror_angles[1]

    # TODO : Dumping theta for testing
#    if (params.rank == 3):
#        np.savetxt("/home/quazartech/bolt/example_problems/electronic_boltzmann/circular_domain_method_2/theta_left.txt", theta_right[N_g:-N_g])

#    if (params.rank == 1):
#        np.savetxt("/home/mchandra/gitansh/merge_to_master/example_problems/electronic_boltzmann/test_quadratic/theta_right.txt", theta_bottom[N_g:-N_g])


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

    # Get the angular variation of the bottom boundary.
    theta_bottom = get_theta(q1, q2, "bottom", \
                             q1_start_local_left=params.q1_start_local_left, \
                             q2_start_local_bottom=params.q2_start_local_bottom)[0, 0, :, bottom_edge]
    theta_bottom = af.moddims(theta_bottom, N_q1_local) # Convert to 1D array

    # TODO : Dumping theta for testing
#    if (params.rank == 7):
#        np.savetxt("/home/quazartech/bolt/example_problems/electronic_boltzmann/circular_domain_method_2/theta_top.txt", theta_bottom[N_g:-N_g])

    # Apply manually specified mirror angles only if bottom boundary of the zone is the bottom boundary of the device
    is_bottom_most_domain = abs(params.q2_start_local_bottom - domain.q2_start) < 1e-13


    if (params.enable_manual_mirror and is_bottom_most_domain):
        theta_bottom = 0.*theta_bottom + params.mirror_angles[0]


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

    q2_end_local_top = params.q2_start_local_bottom + (N_q2_local - 2*N_g)*d_q2

    # Apply manually specified mirror angles only if top boundary of the zone is the top boundary of the device
    is_top_most_domain = abs((q2_end_local_top - domain.q2_end)) < 1e-13

    if (params.enable_manual_mirror and is_top_most_domain):
        theta_top = 0.*theta_top + params.mirror_angles[2]

    # TODO : Dumping theta for testing
#    if (params.rank == 1):
#        np.savetxt("/home/quazartech/bolt/example_problems/electronic_boltzmann/circular_domain_method_2/theta_bottom.txt", theta_top[N_g:-N_g])


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
#    a = [a0, a1, a2, a3, a4, a5, a6, a7]

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

    a0 = a0[0]
    a1 = a1[0]
    a2 = a2[0]
    a3 = a3[0]
    
    
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

    b0 = b0[0]
    b1 = b1[0]
    b2 = b2[0]
    b3 = b3[0]
    
    
    x = a0 + a1*q1 + a2*q2 + a3*q1*q2
    y = b0 + b1*q1 + b2*q2 + b3*q1*q2

    # Calculate and return analytical jacobian
    dx_dq1 = a1 + a3*q2
    dx_dq2 = a2 + a3*q1

    dy_dq1 = b1 + b3*q2
    dy_dq2 = b2 + b3*q1

    jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]
    
    return(x, y, jacobian)

def quadratic(q1, q2,
              x_y_bottom_left, x_y_bottom_right,
              x_y_top_right, x_y_top_left,
              x_y_bottom_center, x_y_right_center,
              x_y_top_center, x_y_left_center,
              q1_start_local_left,
              q2_start_local_bottom,
             ):
    # Maps from ref element [-1, 1] x [-1, 1] to physical domain


    # Nodes on the reference element
    q1_q2_bottom_left   = [-1, -1]
    q1_q2_bottom_center = [ 0, -1]
    q1_q2_bottom_right  = [ 1, -1]

    q1_q2_left_center   = [-1, 0]
    q1_q2_right_center  = [ 1, 0]

    q1_q2_top_left      = [-1, 1]
    q1_q2_top_center    = [ 0, 1]
    q1_q2_top_right     = [ 1, 1]

    x0, y0 = x_y_bottom_left;  X0, Y0 = q1_q2_bottom_left
    x1, y1 = x_y_bottom_right; X1, Y1 = q1_q2_bottom_right
    x2, y2 = x_y_top_right;    X2, Y2 = q1_q2_top_right
    x3, y3 = x_y_top_left;     X3, Y3 = q1_q2_top_left

    x4, y4 = x_y_bottom_center; X4, Y4 = q1_q2_bottom_center
    x5, y5 = x_y_right_center;  X5, Y5 = q1_q2_right_center
    x6, y6 = x_y_top_center;    X6, Y6 = q1_q2_top_center
    x7, y7 = x_y_left_center;   X7, Y7 = q1_q2_left_center


    # Input (q1, q2) may not be [-1, 1] x [-1, 1].
    # First we convert (q1, q2) -> (q1_ref, q2_ref), which are indeed [-1, 1] x [-1, 1]
#    dq1 = (q1[0, 0, 1, 0] - q1[0, 0, 0, 0]).scalar()
#    dq2 = (q2[0, 0, 0, 1] - q2[0, 0, 0, 0]).scalar()
    dq1 = domain.dq1
    dq2 = domain.dq2

    N_g = domain.N_ghost
    q1_start_local = q1[0, 0,  N_g  ,   0     ].scalar()
    q1_end_local   = q1[0, 0, -N_g-1,   0     ].scalar()
    q2_start_local = q2[0, 0,  0,       N_g   ].scalar()
    q2_end_local   = q2[0, 0,  0,      -N_g -1].scalar()

    N_q1_local = q1.dims()[2] - 2*N_g
    N_q2_local = q2.dims()[3] - 2*N_g

    # All of the code below could be done away with by simply:
    # q1_ref = a*q1_input + b
    # q2_ref = c*q2_input + d
    # where a = 2./(q1_end_local_left   - q1_start_local_left)
    #       c = 2./(q2_end_local_bottom - q2_start_local_bottom)
    # (b, d) are irrelevant
    # Note: q1_end_local_left, q2_end_local_bottom are not currently
    #       being passed into the args.

    # Longer code below to avoid passing
    # q1_end_local_left, q2_end_local_bottom into args.
    # Will take a call on what is more elegant later.

    # Need dq1 and dq2 in the ref element
    # End points of ref element are [-1, 1]
    # If N_q1 = 3, i.e., 3 zones:
    #  |  |  |  |
    # -1        1
    # Therefore, we have:
    dq1_ref = 2./N_q1_local
    dq2_ref = 2./N_q2_local

    epsilon = 1e-10 # Book keeping parameter

    if (np.abs(q1_start_local - q1_start_local_left - 0.5*dq1) < epsilon):
        # q1_start_local is at zone center and so q1 is q1_center
        # Get zone centers for the reference element
        q1_ref_start_local = -1. + 0.5*dq1_ref
        q1_ref_end_local   =  1. - 0.5*dq1_ref

    if (np.abs(q1_start_local - q1_start_local_left - dq1) < epsilon):
        # q1_start_local is at right edge and so q1 is q1_right
        # Get right edges for the reference element
        q1_ref_start_local = -1. + dq1_ref
        q1_ref_end_local   =  1.

    if (np.abs(q1_start_local - q1_start_local_left) < epsilon):
        # q1_start_local is at the left edge and so q1 is q1_left
        # Get left edges for the reference element
        q1_ref_start_local = -1.
        q1_ref_end_local   =  1. - dq1_ref


    if (np.abs(q2_start_local - q2_start_local_bottom - 0.5*dq2) < epsilon):
        # q2_start_local is at zone center and so q2 is q2_center
        # Get zone centers for the reference element
        q2_ref_start_local = -1. + 0.5*dq2_ref
        q2_ref_end_local   =  1. - 0.5*dq2_ref

    if (np.abs(q2_start_local - q2_start_local_bottom - dq2) < epsilon):
        # q2_start_local is at top edge and so q2 is q2_top
        # Get top edges for the reference element
        q2_ref_start_local = -1. + dq2_ref
        q2_ref_end_local   =  1.

    if (np.abs(q2_start_local - q2_start_local_bottom) <= epsilon):
        # q2_start_local is at the bottom edge and so q2 is q2_bottom
        # Get bottom edges for the reference element

        q2_ref_start_local = -1.
        q2_ref_end_local   =  1. - dq2_ref

    # Now q1_ref = a*q1 + b, and we need to solve for (a, b)
    # Use the end points:
    #      q1_ref_start_local = a*q1_start_local + b
    #      q1_ref_end_local   = a*q1_end_local   + b

    a =   (q1_ref_start_local - q1_ref_end_local) \
        / (q1_start_local     - q1_end_local)

    b = q1_ref_start_local - a*q1_start_local

    # Similarly, q2_ref = c*q2 + d
    c =   (q2_ref_start_local - q2_ref_end_local) \
        / (q2_start_local     - q2_end_local)

    d = q2_ref_start_local - c*q2_start_local

    # Finally,
    q1_tmp = a*q1 + b
    q2_tmp = c*q2 + d

    dq1_tmp_dq1 = a ; dq1_tmp_dq2 = 0.
    dq2_tmp_dq1 = 0.; dq2_tmp_dq2 = c

#    print ('coordinate_transformation, a :', a)
#    print ('coordinate_transformation, b :', b)
#    print ('coordinate_transformation, c :', c)
#    print ('coordinate_transformation, d :', d)


    #x = a0 + a1*X + a2*Y + a3*X*Y + a4*X**2 + a5*Y**2 + a6*X**2*Y + a7*X*Y**2
    #y = b0 + b1*X + b2*Y + b3*X*Y + b4*X**2 + b5*Y**2 + b6*X**2*Y + b7*X*Y**2

    #x0 = a0 + a1*X0 + a2*Y0 + a3*X0*Y0 + ...
    #x1 = a0 + a1*X1 + a2*Y1 + a3*X1*Y1 + ...
    #x2 = a0 + a1*X2 + a2*Y2 + a3*X2*Y2 + ...
    #x3 = a0 + a1*X3 + a2*Y3 + a3*X3*Y3 + ...

    # A x = b
    A = np.array([[1, X0, Y0, X0*Y0, X0**2., Y0**2., X0**2.*Y0, X0*Y0**2.],
                  [1, X1, Y1, X1*Y1, X1**2., Y1**2., X1**2.*Y1, X1*Y1**2.],
                  [1, X2, Y2, X2*Y2, X2**2., Y2**2., X2**2.*Y2, X2*Y2**2.],
                  [1, X3, Y3, X3*Y3, X3**2., Y3**2., X3**2.*Y3, X3*Y3**2.],
                  [1, X4, Y4, X4*Y4, X4**2., Y4**2., X4**2.*Y4, X4*Y4**2.],
                  [1, X5, Y5, X5*Y5, X5**2., Y5**2., X5**2.*Y5, X5*Y5**2.],
                  [1, X6, Y6, X6*Y6, X6**2., Y6**2., X6**2.*Y6, X6*Y6**2.],
                  [1, X7, Y7, X7*Y7, X7**2., Y7**2., X7**2.*Y7, X7*Y7**2.]
                 ])
    b = np.array([[x0],
                  [x1],
                  [x2],
                  [x3],
                  [x4],
                  [x5],
                  [x6],
                  [x7]
                 ])

    a0, a1, a2, a3, a4, a5, a6, a7 = np.linalg.solve(A, b)

    a0 = a0[0]
    a1 = a1[0]
    a2 = a2[0]
    a3 = a3[0]
    a4 = a4[0]
    a5 = a5[0]
    a6 = a6[0]
    a7 = a7[0]

    #y0 = b0 + b1*X0 + b2*Y0 + b3*X0*Y0 + ...
    #y1 = b0 + b1*X1 + b2*Y1 + b3*X1*Y1 + ...
    #y2 = b0 + b1*X2 + b2*Y2 + b3*X2*Y2 + ...
    #y3 = b0 + b1*X3 + b2*Y3 + b3*X3*Y3 + ...

    b = np.array([[y0],
                  [y1],
                  [y2],
                  [y3],
                  [y4],
                  [y5],
                  [y6],
                  [y7]
                 ])

    b0, b1, b2, b3, b4, b5, b6, b7 = np.linalg.solve(A, b)

    b0 = b0[0]
    b1 = b1[0]
    b2 = b2[0]
    b3 = b3[0]
    b4 = b4[0]
    b5 = b5[0]
    b6 = b6[0]
    b7 = b7[0]

    X = q1_tmp; Y = q2_tmp # renaming (q1, q2) -> (X, Y) for ease of reading the eqns below

    x     = a0 + a1*X + a2*Y + a3*X*Y + a4*X**2 + a5*Y**2 + a6*X**2*Y + a7*X*Y**2
    y     = b0 + b1*X + b2*Y + b3*X*Y + b4*X**2 + b5*Y**2 + b6*X**2*Y + b7*X*Y**2

    dx_dX =      a1          + a3*Y   + 2*a4*X            + 2*a6*X*Y  + a7*Y**2
    dx_dY =             a2   + a3*X             + 2*a5*Y  +   a6*X**2 + 2*a7*X*Y

    dy_dX =      b1          + b3*Y   + 2*b4*X            + 2*b6*X*Y  + b7*Y**2
    dy_dY =             b2   + b3*X             + 2*b5*Y  + b6*X**2   + 2*b7*X*Y


    dx_dq1_tmp = dx_dX; dx_dq2_tmp = dx_dY
    dy_dq1_tmp = dy_dX; dy_dq2_tmp = dy_dY

    dx_dq1 = (dx_dq1_tmp * dq1_tmp_dq1) + (dx_dq2_tmp * dq2_tmp_dq1)
    dx_dq2 = (dx_dq1_tmp * dq1_tmp_dq2) + (dx_dq2_tmp * dq2_tmp_dq2)

    dy_dq1 = (dy_dq1_tmp * dq1_tmp_dq1) + (dy_dq2_tmp * dq2_tmp_dq1)
    dy_dq2 = (dy_dq1_tmp * dq1_tmp_dq2) + (dy_dq2_tmp * dq2_tmp_dq2)

    jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]

    return (x, y, jacobian)



def quadratic_test(q1, q2, q1_slice, q2_slice,
              x_y_bottom_left, x_y_bottom_right,
              x_y_top_right, x_y_top_left,
              x_y_bottom_center, x_y_right_center,
              x_y_top_center, x_y_left_center,
              q1_start_local_left,
              q2_start_local_bottom,
             ):
    # Here q1, q2 is a point in q-space
    # q1_slice, q2_slice define the slice in q1, q2 space over which the transformation is to be applied
    # Maps from ref element [-1, 1] x [-1, 1] to physical domain
    # Returns x, y and jacobian at a point after applying transformation


    # Define the reference element
    q1_q2_bottom_left   = [-1, -1]
    q1_q2_bottom_center = [ 0, -1]
    q1_q2_bottom_right  = [ 1, -1]

    q1_q2_left_center   = [-1, 0]
    q1_q2_right_center  = [ 1, 0]

    q1_q2_top_left      = [-1, 1]
    q1_q2_top_center    = [ 0, 1]
    q1_q2_top_right     = [ 1, 1]

    # Store points to be used for transformation
    x0, y0 = x_y_bottom_left;  X0, Y0 = q1_q2_bottom_left
    x1, y1 = x_y_bottom_right; X1, Y1 = q1_q2_bottom_right
    x2, y2 = x_y_top_right;    X2, Y2 = q1_q2_top_right
    x3, y3 = x_y_top_left;     X3, Y3 = q1_q2_top_left

    x4, y4 = x_y_bottom_center; X4, Y4 = q1_q2_bottom_center
    x5, y5 = x_y_right_center;  X5, Y5 = q1_q2_right_center
    x6, y6 = x_y_top_center;    X6, Y6 = q1_q2_top_center
    x7, y7 = x_y_left_center;   X7, Y7 = q1_q2_left_center



    # Input (q1, q2) may not be [-1, 1] x [-1, 1].
    # First we convert (q1, q2) -> (q1_ref, q2_ref), which are indeed [-1, 1] x [-1, 1]
    dq1 = domain.dq1
    dq2 = domain.dq2

    N_g = domain.N_ghost
    q1_start_local = q1_slice[0, 0,  0,  0 ].scalar()
    q2_start_local = q2_slice[0, 0,  0,  0 ].scalar()


    try :
        N_q1_local = q1_slice.dims()[2] # Does not include any ghost zones
    except(IndexError):
        N_q1_local = 1
    try :
        N_q2_local = q2_slice.dims()[3] 
    except(IndexError):
        N_q2_local = 1

    # All of the code below could be done away with by simply:
    # q1_ref = a*q1_input + b
    # q2_ref = c*q2_input + d
    # where a = 2./(q1_end_local_left   - q1_start_local_left)
    #       c = 2./(q2_end_local_bottom - q2_start_local_bottom)
    # (b, d) are irrelevant
    # Note: q1_end_local_left, q2_end_local_bottom are not currently
    #       being passed into the args.

    # Longer code below to avoid passing
    # q1_end_local_left, q2_end_local_bottom into args.
    # Will take a call on what is more elegant later.

    # Need dq1 and dq2 in the ref element
    # End points of ref element are [-1, 1]
    # If N_q1 = 3, i.e., 3 zones:
    #  |  |  |  |
    # -1        1
    # Therefore, we have:
    dq1_ref = 2./N_q1_local
    dq2_ref = 2./N_q2_local


    epsilon = 1e-10 # Book keeping parameter

    if (np.abs(q1_start_local - q1_start_local_left - 0.5*dq1) < epsilon):
        # q1_start_local is at zone center and so q1 is q1_center
        # Get zone centers for the reference element
        q1_ref_start_local = -1. + 0.5*dq1_ref
        q1_ref_end_local   =  1. - 0.5*dq1_ref

    if (np.abs(q1_start_local - q1_start_local_left - dq1) < epsilon):
        # q1_start_local is at right edge and so q1 is q1_right
        # Get right edges for the reference element
        q1_ref_start_local = -1. + dq1_ref
        q1_ref_end_local   =  1.

    if (np.abs(q1_start_local - q1_start_local_left) < epsilon):
        # q1_start_local is at the left edge and so q1 is q1_left
        # Get left edges for the reference element
        q1_ref_start_local = -1.
        q1_ref_end_local   =  1. - dq1_ref


    if (np.abs(q2_start_local - q2_start_local_bottom - 0.5*dq2) < epsilon):
        # q2_start_local is at zone center and so q2 is q2_center
        # Get zone centers for the reference element
        q2_ref_start_local = -1. + 0.5*dq2_ref
        q2_ref_end_local   =  1. - 0.5*dq2_ref

    if (np.abs(q2_start_local - q2_start_local_bottom - dq2) < epsilon):
        # q2_start_local is at top edge and so q2 is q2_top
        # Get top edges for the reference element
        q2_ref_start_local = -1. + dq2_ref
        q2_ref_end_local   =  1.

    if (np.abs(q2_start_local - q2_start_local_bottom) <= epsilon):
        # q2_start_local is at the bottom edge and so q2 is q2_bottom
        # Get bottom edges for the reference element

        q2_ref_start_local = -1.
        q2_ref_end_local   =  1. - dq2_ref


    # Now q1_ref = a*q1 + b, and we need to solve for (a, b)
    # Use the end points:
    #      q1_ref_start_local = a*q1_start_local + b
    #      q1_ref_end_local   = a*q1_end_local   + b

    a = dq1_ref/dq1 # Scaling factor in q1
    b = q1_ref_start_local - a*q1_start_local

    # Similarly, q2_ref = c*q2 + d

    c = dq2_ref/dq2 # Scaling factor in q2
    d = q2_ref_start_local - c*q2_start_local


    # Finally,
    q1_tmp = a*q1 + b
    q2_tmp = c*q2 + d

    dq1_tmp_dq1 = a ; dq1_tmp_dq2 = 0.
    dq2_tmp_dq1 = 0.; dq2_tmp_dq2 = c


    #x = a0 + a1*X + a2*Y + a3*X*Y + a4*X**2 + a5*Y**2 + a6*X**2*Y + a7*X*Y**2
    #y = b0 + b1*X + b2*Y + b3*X*Y + b4*X**2 + b5*Y**2 + b6*X**2*Y + b7*X*Y**2

    #x0 = a0 + a1*X0 + a2*Y0 + a3*X0*Y0 + ...
    #x1 = a0 + a1*X1 + a2*Y1 + a3*X1*Y1 + ...
    #x2 = a0 + a1*X2 + a2*Y2 + a3*X2*Y2 + ...
    #x3 = a0 + a1*X3 + a2*Y3 + a3*X3*Y3 + ...


    # A x = b
    A = np.array([[1, X0, Y0, X0*Y0, X0**2., Y0**2., X0**2.*Y0, X0*Y0**2.],
                  [1, X1, Y1, X1*Y1, X1**2., Y1**2., X1**2.*Y1, X1*Y1**2.],
                  [1, X2, Y2, X2*Y2, X2**2., Y2**2., X2**2.*Y2, X2*Y2**2.],
                  [1, X3, Y3, X3*Y3, X3**2., Y3**2., X3**2.*Y3, X3*Y3**2.],
                  [1, X4, Y4, X4*Y4, X4**2., Y4**2., X4**2.*Y4, X4*Y4**2.],
                  [1, X5, Y5, X5*Y5, X5**2., Y5**2., X5**2.*Y5, X5*Y5**2.],
                  [1, X6, Y6, X6*Y6, X6**2., Y6**2., X6**2.*Y6, X6*Y6**2.],
                  [1, X7, Y7, X7*Y7, X7**2., Y7**2., X7**2.*Y7, X7*Y7**2.]
                 ])
    b = np.array([[x0],
                  [x1],
                  [x2],
                  [x3],
                  [x4],
                  [x5],
                  [x6],
                  [x7]
                 ])

    a0, a1, a2, a3, a4, a5, a6, a7 = np.linalg.solve(A, b)

    # Convert to floats for multiplication with arrayfire arrays
    a0 = a0[0]
    a1 = a1[0]
    a2 = a2[0]
    a3 = a3[0]
    a4 = a4[0]
    a5 = a5[0]
    a6 = a6[0]
    a7 = a7[0]

    #y0 = b0 + b1*X0 + b2*Y0 + b3*X0*Y0 + ...
    #y1 = b0 + b1*X1 + b2*Y1 + b3*X1*Y1 + ...
    #y2 = b0 + b1*X2 + b2*Y2 + b3*X2*Y2 + ...
    #y3 = b0 + b1*X3 + b2*Y3 + b3*X3*Y3 + ...

    b = np.array([[y0],
                  [y1],
                  [y2],
                  [y3],
                  [y4],
                  [y5],
                  [y6],
                  [y7]
                 ])

    b0, b1, b2, b3, b4, b5, b6, b7 = np.linalg.solve(A, b)

    # Convert to floats for multiplication with arrayfire arrays
    b0 = b0[0]
    b1 = b1[0]
    b2 = b2[0]
    b3 = b3[0]
    b4 = b4[0]
    b5 = b5[0]
    b6 = b6[0]
    b7 = b7[0]


    X = q1_tmp; Y = q2_tmp # renaming (q1, q2) -> (X, Y) for ease of reading the eqns below


    x     = a0 + a1*X + a2*Y + a3*X*Y + a4*X**2 + a5*Y**2 + a6*X**2*Y + a7*X*Y**2
    y     = b0 + b1*X + b2*Y + b3*X*Y + b4*X**2 + b5*Y**2 + b6*X**2*Y + b7*X*Y**2


    dx_dX =      a1          + a3*Y   + 2*a4*X            + 2*a6*X*Y  + a7*Y**2
    dx_dY =             a2   + a3*X             + 2*a5*Y  +   a6*X**2 + 2*a7*X*Y

    dy_dX =      b1          + b3*Y   + 2*b4*X            + 2*b6*X*Y  + b7*Y**2
    dy_dY =             b2   + b3*X             + 2*b5*Y  + b6*X**2   + 2*b7*X*Y


    dx_dq1_tmp = dx_dX; dx_dq2_tmp = dx_dY
    dy_dq1_tmp = dy_dX; dy_dq2_tmp = dy_dY

    dx_dq1 = (dx_dq1_tmp * dq1_tmp_dq1) + (dx_dq2_tmp * dq2_tmp_dq1)    
    dx_dq2 = (dx_dq1_tmp * dq1_tmp_dq2) + (dx_dq2_tmp * dq2_tmp_dq2)

    dy_dq1 = (dy_dq1_tmp * dq1_tmp_dq1) + (dy_dq2_tmp * dq2_tmp_dq1)
    dy_dq2 = (dy_dq1_tmp * dq1_tmp_dq2) + (dy_dq2_tmp * dq2_tmp_dq2)

    jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]

    return (x, y, jacobian)

