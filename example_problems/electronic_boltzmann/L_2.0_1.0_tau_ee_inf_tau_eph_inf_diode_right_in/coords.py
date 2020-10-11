import numpy as np
import arrayfire as af

#TODO : Not able to import affine from utils. Fix this.
#from bolt.lib.utils.coord_transformation import affine

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
    
    jacobian = None # Numerically compute the Jacobian
    
    if (q1_start_local_left != None and q2_start_local_bottom != None):

        # Bottom center patch
        if ((q2_midpoint < 0.3) and (q1_midpoint > 1.) and (q1_midpoint < 2.)):
            x0 = 1.;  y0 = 0  # Bottom-left
            x1 = 2;   y1 = 0  # Bottom-right
            x2 = 2;   y2 = 0.45  # Top-right
            x3 = 1.;  y3 = 0.1  # Top-left
            x, y = \
              affine(q1, q2, 
               [x0, y0], [x1, y1],
               [x2, y2], [x3, y3],
               [1., 0], [2, 0],
               [2, 0.3], [1., 0.3]
                    )

        # Bottom left patch
        elif ((q2_midpoint < 0.3) and (q1_midpoint > 0.) and (q1_midpoint < 1.)):
            x0 = 0.;  y0 = 0  # Bottom-left
            x1 = 1;   y1 = 0  # Bottom-right
            x2 = 1;   y2 = 0.1  # Top-right
            x3 = 0.;  y3 = 0.1  # Top-left
            x, y = \
              affine(q1, q2, 
               [x0, y0], [x1, y1],
               [x2, y2], [x3, y3],
               [0., 0], [1, 0],
               [1, 0.3], [0., 0.3]
                    )
        
        # Bottom right patch
        elif ((q2_midpoint < 0.3) and (q1_midpoint > 2.) and (q1_midpoint < 3.)):
            x0 = 2.;  y0 = 0  # Bottom-left
            x1 = 3;   y1 = 0  # Bottom-right
            x2 = 3;   y2 = 0.1  # Top-right
            x3 = 2.;  y3 = 0.45  # Top-left
            x, y = \
              affine(q1, q2, 
               [x0, y0], [x1, y1],
               [x2, y2], [x3, y3],
               [2., 0], [3, 0],
               [3, 0.3], [2., 0.3]
                    )
        
        # Top right patch
        elif ((q2_midpoint > 0.7) and (q1_midpoint > 2.) and (q1_midpoint < 3.)):
            x0 = 2.;  y0 = 0.55  # Bottom-left
            x1 = 3;   y1 = 0.9  # Bottom-right
            x2 = 3;   y2 = 1.  # Top-right
            x3 = 2.;  y3 = 1.  # Top-left
            x, y = \
              affine(q1, q2, 
               [x0, y0], [x1, y1],
               [x2, y2], [x3, y3],
               [2., 0.7], [3, 0.7],
               [3, 1.], [2., 1.]
                    )
        # Top left patch
        elif ((q2_midpoint > 0.7) and (q1_midpoint > 0.) and (q1_midpoint < 1.)):
            x0 = 0.;  y0 = 0.9  # Bottom-left
            x1 = 1;   y1 = 0.9  # Bottom-right
            x2 = 1;   y2 = 1.  # Top-right
            x3 = 0.;  y3 = 1.  # Top-left
            x, y = \
              affine(q1, q2, 
               [x0, y0], [x1, y1],
               [x2, y2], [x3, y3],
               [0., 0.7], [1, 0.7],
               [1, 1.], [0., 1.]
                    )
        
        # Top center patch
        elif ((q2_midpoint > 0.7) and (q1_midpoint > 1.) and (q1_midpoint < 2.)):
            x0 = 1.;  y0 = 0.9  # Bottom-left
            x1 = 2;   y1 = 0.55  # Bottom-right
            x2 = 2;   y2 = 1.  # Top-right
            x3 = 1.;  y3 = 1.  # Top-left
            x, y = \
              affine(q1, q2, 
               [x0, y0], [x1, y1],
               [x2, y2], [x3, y3],
               [1., 0.7], [2, 0.7],
               [2, 1.], [1., 1.]
                    )
        
        # Center center patch
        elif ((q2_midpoint > 0.3) and (q2_midpoint < 0.7) and (q1_midpoint > 1.) and (q1_midpoint < 2.)):
            x0 = 1.;  y0 = 0.1  # Bottom-left
            x1 = 2;   y1 = 0.45  # Bottom-right
            x2 = 2;   y2 = 0.55  # Top-right
            x3 = 1.;  y3 = 0.9  # Top-left
            x, y = \
              affine(q1, q2, 
               [x0, y0], [x1, y1],
               [x2, y2], [x3, y3],
               [1., 0.3], [2, 0.3],
               [2, 0.7], [1., 0.7]
                    )
        
        # Left center patch
        elif ((q2_midpoint > 0.3) and (q2_midpoint < 0.7) and (q1_midpoint > 0.) and (q1_midpoint < 1.)):
            x0 = 0.;  y0 = 0.1  # Bottom-left
            x1 = 1;   y1 = 0.1  # Bottom-right
            x2 = 1;   y2 = 0.9  # Top-right
            x3 = 0.;  y3 = 0.9  # Top-left
            x, y = \
              affine(q1, q2, 
               [x0, y0], [x1, y1],
               [x2, y2], [x3, y3],
               [0., 0.3], [1, 0.3],
               [1, 0.7], [0., 0.7]
                    )
        
        # Right center patch
        elif ((q2_midpoint > 0.3) and (q2_midpoint < 0.7) and (q1_midpoint > 2.) and (q1_midpoint < 3.)):
            x0 = 2.;  y0 = 0.45  # Bottom-left
            x1 = 3;   y1 = 0.1  # Bottom-right
            x2 = 3;   y2 = 0.9  # Top-right
            x3 = 2.;  y3 = 0.55  # Top-left
            x, y = \
              affine(q1, q2, 
               [x0, y0], [x1, y1],
               [x2, y2], [x3, y3],
               [2., 0.3], [3, 0.3],
               [3, 0.7], [2., 0.7]
                    )


        if (return_jacobian):
            return (x, y, jacobian)
        else: 
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")

