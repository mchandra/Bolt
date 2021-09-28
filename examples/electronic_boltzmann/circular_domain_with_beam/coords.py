import numpy as np
import arrayfire as af
import domain

from bolt.lib.utils.coord_transformation import quadratic

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
    dq1 = (q1[0, 0, 1, 0] - q1[0, 0, 0, 0]).scalar()
    dq2 = (q2[0, 0, 0, 1] - q2[0, 0, 0, 0]).scalar()

    N_g = domain.N_ghost
    q1_start_local = q1[0, 0,  N_g  ,   0    ].scalar()
    q1_end_local   = q1[0, 0, -N_g-1,   0    ].scalar()
    q2_start_local = q2[0, 0,      0,  N_g   ].scalar()
    q2_end_local   = q2[0, 0,      0, -N_g -1].scalar()

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
    
    if (q1_start_local - q1_start_local_left > 0):
        # q1_start_local is at zone center and so q1 is q1_center
        # Get zone centers for the reference element
        q1_ref_start_local = -1. + 0.5*dq1_ref
        q1_ref_end_local   =  1. - 0.5*dq1_ref
        
    if (np.abs(q1_start_local - q1_start_local_left) < 1e-10):
        # q1_start_local is at the left edge and so q1 is q1_left
        # Get left edges for the reference element

        q1_ref_start_local = -1.
        q1_ref_end_local   =  1. - dq1_ref
        
    if (q2_start_local - q2_start_local_bottom > 0):
        # q2_start_local is at zone center and so q2 is q2_center
        # Get zone centers for the reference element
        q2_ref_start_local = -1. + 0.5*dq2_ref
        q2_ref_end_local   =  1. - 0.5*dq2_ref
        
    if (np.abs(q2_start_local - q2_start_local_bottom) < 1e-10):
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


def affine(q1, q2,
           x_y_bottom_left, x_y_bottom_right, 
           x_y_top_right, x_y_top_left,
           X_Y_bottom_left, X_Y_bottom_right, 
           X_Y_top_right, X_Y_top_left,
          ):
    a = [a0, a1, a2, a3, a4, a5, a6, a7]

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
    
    return(x, y)

def get_cartesian_coords(q1, q2, 
                         q1_start_local_left=None, 
                         q2_start_local_bottom=None,
                         return_jacobian = False
                        ):

    q1_midpoint = 0.5*(af.max(q1) + af.min(q1))
    q2_midpoint = 0.5*(af.max(q2) + af.min(q2))

    # Default initializsation to rectangular grid
    x = q1
    y = q2
    jacobian = [[1. + 0.*q1,      0.*q1],
                [     0.*q1, 1. + 0.*q1]
               ]

    # Radius and center of circular region
    radius          = 0.5
    center          = [0, 0]


    if (q1_start_local_left != None and q2_start_local_bottom != None):

        # Bottom-center domain
#        if ((q2_midpoint < -0.33) and (q1_midpoint > -0.33) and (q1_midpoint < 0.33)):
#
#            # Note : Never specify the x, y coordinates below in terms of q1 and q2 coordinates. Specify only in
#            # physical x, y values.
#    
#            x_y_bottom_left   = [-radius/np.sqrt(2), -1]
#            x_y_bottom_center = [0                 , -1]
#            x_y_bottom_right  = [radius/np.sqrt(2) , -1]
#            
#            x_y_left_center  = [-radius/np.sqrt(2), (-1-radius/np.sqrt(2))/2]
#            x_y_right_center = [ radius/np.sqrt(2), (-1-radius/np.sqrt(2))/2]
#            
#            x_y_top_left     = [-radius/np.sqrt(2), -radius/np.sqrt(2)]
#            x_y_top_center   = [0                 , -radius           ]
#            x_y_top_right    = [radius/np.sqrt(2) , -radius/np.sqrt(2)]     
#
#            x, y, jacobian = quadratic(q1, q2,
#                                       x_y_bottom_left,   x_y_bottom_right,
#                                       x_y_top_right,     x_y_top_left,
#                                       x_y_bottom_center, x_y_right_center,
#                                       x_y_top_center,    x_y_left_center,
#                                       q1_start_local_left,
#                                       q2_start_local_bottom,
#                                      )
#            
#    
#        # Bottom-left domain
#        elif ((q2_midpoint < -0.33) and (q1_midpoint > -1) and (q1_midpoint < -0.33)):
#    
#            x_y_bottom_left   = [-1,                 -1]
#            x_y_bottom_center = [(-1-radius/np.sqrt(2))/2,    -1]
#            x_y_bottom_right  = [-radius/np.sqrt(2), -1]
#    
#            x_y_left_center  = [-1,                 (-1-radius/np.sqrt(2))/2]
#            x_y_right_center = [-radius/np.sqrt(2), (-1-radius/np.sqrt(2))/2]        
#            
#            x_y_top_left     = [-1,                        -radius/np.sqrt(2)]
#            x_y_top_center   = [-(1.+radius/np.sqrt(2))/2, -radius/np.sqrt(2)]
#            x_y_top_right    = [-radius/np.sqrt(2),        -radius/np.sqrt(2)] 
#    
#            x, y, jacobian = quadratic(q1, q2,
#                                       x_y_bottom_left,   x_y_bottom_right,
#                                       x_y_top_right,     x_y_top_left,
#                                       x_y_bottom_center, x_y_right_center,
#                                       x_y_top_center,    x_y_left_center,
#                                       q1_start_local_left,
#                                       q2_start_local_bottom,
#                                      )
#
#    
#        # Bottom-right domain
#        elif ((q2_midpoint < -0.33) and (q1_midpoint > 0.33) and (q1_midpoint < 1.)):
#    
#            x_y_bottom_left   = [radius/np.sqrt(2),          -1]
#            x_y_bottom_center = [(1+radius/np.sqrt(2))/2,   -1]
#            x_y_bottom_right  = [1,                          -1]
#    
#            x_y_left_center   = [ radius/np.sqrt(2), (-1-radius/np.sqrt(2))/2]
#            x_y_right_center  = [1,                  (-1-radius/np.sqrt(2))/2]
#            
#            x_y_top_left     = [radius/np.sqrt(2),         -radius/np.sqrt(2)]
#            x_y_top_center   = [(1.+radius/np.sqrt(2))/2,  -radius/np.sqrt(2)]
#            x_y_top_right    = [1,                         -radius/np.sqrt(2)] 
#    
#            x, y, jacobian = quadratic(q1, q2,
#                                       x_y_bottom_left,   x_y_bottom_right,
#                                       x_y_top_right,     x_y_top_left,
#                                       x_y_bottom_center, x_y_right_center,
#                                       x_y_top_center,    x_y_left_center,
#                                       q1_start_local_left,
#                                       q2_start_local_bottom,
#                                      )
#    
#
#        # Top-center domain
#        elif ((q2_midpoint > 0.33) and (q1_midpoint > -0.33) and (q1_midpoint < 0.33)):
#
#            x_y_bottom_left   = [-radius/np.sqrt(2), radius/np.sqrt(2)]
#            x_y_bottom_center = [0,                  radius]
#            x_y_bottom_right  = [radius/np.sqrt(2),  radius/np.sqrt(2)]
#            
#            x_y_left_center   = [-radius/np.sqrt(2), (1+radius/np.sqrt(2))/2]
#            x_y_right_center  = [ radius/np.sqrt(2), (1+radius/np.sqrt(2))/2]
#            
#            x_y_top_left      = [-radius/np.sqrt(2), 1]
#            x_y_top_center    = [0,                  1]
#            x_y_top_right     = [radius/np.sqrt(2),  1]
#            
#            x, y, jacobian = quadratic(q1, q2,
#                                       x_y_bottom_left,   x_y_bottom_right,
#                                       x_y_top_right,     x_y_top_left,
#                                       x_y_bottom_center, x_y_right_center,
#                                       x_y_top_center,    x_y_left_center,
#                                       q1_start_local_left,
#                                       q2_start_local_bottom,
#                                      )
#
#
#        # Top-left domain
#        elif ((q2_midpoint > 0.33) and (q1_midpoint > -1) and (q1_midpoint < -0.33)):
#
#            x_y_bottom_left   = [-1,                         radius/np.sqrt(2)]
#            x_y_bottom_center = [-(1.+radius/np.sqrt(2))/2,  radius/np.sqrt(2)]
#            x_y_bottom_right  = [-radius/np.sqrt(2),         radius/np.sqrt(2)]
#            
#            x_y_left_center  = [-1,                 (1+radius/np.sqrt(2))/2]
#            x_y_right_center = [-radius/np.sqrt(2), (1+radius/np.sqrt(2))/2]
#    
#            x_y_top_left      = [-1,                          1]
#            x_y_top_center    = [-(1+radius/np.sqrt(2))/2,    1] 
#            x_y_top_right     = [-radius/np.sqrt(2), 1]
#            
#            x, y, jacobian = quadratic(q1, q2,
#                                       x_y_bottom_left,   x_y_bottom_right,
#                                       x_y_top_right,     x_y_top_left,
#                                       x_y_bottom_center, x_y_right_center,
#                                       x_y_top_center,    x_y_left_center,
#                                       q1_start_local_left,
#                                       q2_start_local_bottom,
#                                      )
#
#        
#        # Top-right domain
#        elif ((q2_midpoint > 0.33) and (q1_midpoint > 0.33) and (q1_midpoint < 1)):
#
#            x_y_bottom_left   = [radius/np.sqrt(2),         radius/np.sqrt(2)]
#            x_y_bottom_center = [(1.+radius/np.sqrt(2))/2,  radius/np.sqrt(2)]
#            x_y_bottom_right  = [1,                         radius/np.sqrt(2)]
#            
#            x_y_right_center = [1.,                 (1+radius/np.sqrt(2))/2]
#            x_y_left_center  = [ radius/np.sqrt(2), (1+radius/np.sqrt(2))/2]
#    
#            x_y_top_left      = [radius/np.sqrt(2), 1]
#            x_y_top_center    = [(1+radius/np.sqrt(2))/2,   1]
#            x_y_top_right     = [1,                 1]
#            
#            x, y, jacobian = quadratic(q1, q2,
#                                       x_y_bottom_left,   x_y_bottom_right,
#                                       x_y_top_right,     x_y_top_left,
#                                       x_y_bottom_center, x_y_right_center,
#                                       x_y_top_center,    x_y_left_center,
#                                       q1_start_local_left,
#                                       q2_start_local_bottom,
#                                      )
#
#
#        # Center-Right domain
#        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint > 0.33)):
#
#            x_y_bottom_left   = [radius/np.sqrt(2),           -radius/np.sqrt(2)]
#            x_y_bottom_center = [(1.+radius/np.sqrt(2))/2,    -radius/np.sqrt(2)]
#            x_y_bottom_right  = [1.,                          -radius/np.sqrt(2)]
#            
#            x_y_left_center  = [radius, 0.]
#            x_y_right_center = [1.,     0.]
#            
#            x_y_top_left     = [radius/np.sqrt(2),           radius/np.sqrt(2) ]
#            x_y_top_center   = [(1.+radius/np.sqrt(2))/2,    radius/np.sqrt(2) ]
#            x_y_top_right    = [1.,                          radius/np.sqrt(2) ]
#            
#            x, y, jacobian = quadratic(q1, q2,
#                                       x_y_bottom_left,   x_y_bottom_right,
#                                       x_y_top_right,     x_y_top_left,
#                                       x_y_bottom_center, x_y_right_center,
#                                       x_y_top_center,    x_y_left_center,
#                                       q1_start_local_left,
#                                       q2_start_local_bottom,
#                                      )
#    
#
#        # Center-Left domain
#        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint < -0.33)):
#
#            x_y_bottom_left   = [-1.,                          -radius/np.sqrt(2)]
#            x_y_bottom_center = [-(1.+radius/np.sqrt(2))/2,    -radius/np.sqrt(2)]
#            x_y_bottom_right  = [-radius/np.sqrt(2),           -radius/np.sqrt(2)  ]
#            
#            x_y_left_center  = [-1.,     0.]
#            x_y_right_center = [-radius, 0.]
#            
#            x_y_top_left     = [-1.,                          radius/np.sqrt(2)]
#            x_y_top_center   = [-(1.+radius/np.sqrt(2))/2,    radius/np.sqrt(2)]
#            x_y_top_right    = [-radius/np.sqrt(2),           radius/np.sqrt(2)  ]
#            
#            x, y, jacobian = quadratic(q1, q2,
#                                       x_y_bottom_left,   x_y_bottom_right,
#                                       x_y_top_right,     x_y_top_left,
#                                       x_y_bottom_center, x_y_right_center,
#                                       x_y_top_center,    x_y_left_center,
#                                       q1_start_local_left,
#                                       q2_start_local_bottom,
#                                      )
#    
#
#        # Center domain
#        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint > -0.33) and (q1_midpoint < 0.33)):
#
#            x_y_bottom_left   = [-radius/np.sqrt(2),  -radius/np.sqrt(2)]
#            x_y_bottom_center = [0.,                  -radius]
#            x_y_bottom_right  = [radius/np.sqrt(2),   -radius/np.sqrt(2)]
#            
#            x_y_left_center  = [-radius, 0]
#            x_y_right_center = [ radius, 0]
#            
#            x_y_top_left     = [-radius/np.sqrt(2),  radius/np.sqrt(2)]
#            x_y_top_center   = [0.,                  radius]
#            x_y_top_right    = [radius/np.sqrt(2),   radius/np.sqrt(2)]
#            
#            x, y, jacobian = quadratic(q1, q2,
#                                       x_y_bottom_left,   x_y_bottom_right,
#                                       x_y_top_right,     x_y_top_left,
#                                       x_y_bottom_center, x_y_right_center,
#                                       x_y_top_center,    x_y_left_center,
#                                       q1_start_local_left,
#                                       q2_start_local_bottom,
#                                      )

        if (return_jacobian):
            return (x, y, jacobian)
        else: 
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")


