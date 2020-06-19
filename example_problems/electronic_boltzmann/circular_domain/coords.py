import numpy as np
import arrayfire as af

#TODO : Not able to import affine from utils. Fix this.
#from bolt.lib.utils.coord_transformation import affine

def quadratic(X, Y,
              x_y_bottom_left, x_y_bottom_right, 
              x_y_top_right, x_y_top_left,
              x_y_bottom_center, x_y_right_center,
              x_y_top_center, x_y_left_center,
              X_Y_bottom_left, X_Y_bottom_right, 
              X_Y_top_right, X_Y_top_left,
              X_Y_bottom_center, X_Y_right_center,
              X_Y_top_center, X_Y_left_center,
          ):
    
    x0, y0 = x_y_bottom_left;  X0, Y0 = X_Y_bottom_left
    x1, y1 = x_y_bottom_right; X1, Y1 = X_Y_bottom_right
    x2, y2 = x_y_top_right;    X2, Y2 = X_Y_top_right
    x3, y3 = x_y_top_left;     X3, Y3 = X_Y_top_left

    x4, y4 = x_y_bottom_center; X4, Y4 = X_Y_bottom_center
    x5, y5 = x_y_right_center;  X5, Y5 = X_Y_right_center
    x6, y6 = x_y_top_center;    X6, Y6 = X_Y_top_center
    x7, y7 = x_y_left_center;   X7, Y7 = X_Y_left_center

    
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

#    print ('coords.py, A : ', A.shape, A.dtype)
#    print ('coords.py, b : ', b.shape, b.dtype)

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

#    print ('coords.py, A : ', A.shape, A.dtype)
#    print ('coords.py, b : ', b.shape, b.dtype)


    b0, b1, b2, b3, b4, b5, b6, b7 = np.linalg.solve(A, b)

    b0 = b0[0]
    b1 = b1[0]
    b2 = b2[0]
    b3 = b3[0]
    b4 = b4[0]
    b5 = b5[0]
    b6 = b6[0]
    b7 = b7[0]
    
    x = a0 + a1*X + a2*Y + a3*X*Y + a4*X**2 + a5*Y**2 + a6*X**2*Y + a7*X*Y**2
    y = b0 + b1*X + b2*Y + b3*X*Y + b4*X**2 + b5*Y**2 + b6*X**2*Y + b7*X*Y**2
    
    return(x, y)


#def affine(q1, q2,
#           x_y_bottom_left, x_y_bottom_right, 
#           x_y_top_right, x_y_top_left,
#           X_Y_bottom_left, X_Y_bottom_right, 
#           X_Y_top_right, X_Y_top_left,
#          ):
def affine(q1, q2,
              x_y_bottom_left, x_y_bottom_right, 
              x_y_top_right, x_y_top_left,
              x_y_bottom_center, x_y_right_center,
              x_y_top_center, x_y_left_center,
              X_Y_bottom_left, X_Y_bottom_right, 
              X_Y_top_right, X_Y_top_left,
              X_Y_bottom_center, X_Y_right_center,
              X_Y_top_center, X_Y_left_center,
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

def get_cartesian_coords(q1, q2):

    q1_midpoint = 0.5*(af.max(q1) + af.min(q1))
    q2_midpoint = 0.5*(af.max(q2) + af.min(q2))

    x = q1
    y = q2

    radius = 0.5
    center = [0, 0]

    dq1 = (q1[0, 0, 1, 0] - q1[0, 0, 0, 0]).scalar()
    dq2 = (q2[0, 0, 0, 1] - q2[0, 0, 0, 0]).scalar()

    q1_start_local = q1[0, 0, 0, 0].scalar()
    q2_start_local = q2[0, 0, 0, 0].scalar()

    q1_end_local = q1[0, 0, -1, 0].scalar()
    q2_end_local = q2[0, 0, 0, -1].scalar()

    q1_center_local = q1_midpoint
    q2_center_local = q2_midpoint

    print ("coords.py, start : ", q1_start_local, q2_start_local)
    print ("coords.py, end : ", q1_end_local, q2_end_local)
    print ("coords.py, center : ", q1_center_local, q2_center_local)
        
#    X_Y_bottom_left   = [q1_start_local, q2_start_local]
#    X_Y_bottom_center = [q1_center_local, q2_start_local]
#    X_Y_bottom_right  = [q1_end_local, q2_start_local]
#    
#    X_Y_left_center   = [q1_start_local, q2_center_local]
#    X_Y_right_center  = [q1_end_local, q2_center_local]
#    
#    X_Y_top_left      = [q1_start_local, q2_end_local]
#    X_Y_top_center    = [q1_center_local, q2_end_local]
#    X_Y_top_right     = [q1_end_local, q2_end_local] 

    X_Y_bottom_left   = [-1, -1]
    X_Y_bottom_center = [0, -1]
    X_Y_bottom_right  = [1, -1]
    
    X_Y_left_center   = [-1, 0]
    X_Y_right_center  = [1, 0]
    
    X_Y_top_left      = [-1, 1]
    X_Y_top_center    = [0, 1]
    X_Y_top_right     = [1, 1]

    q1_scale_factor = 2./(q1_end_local - q1_start_local)
    q2_scale_factor = 2./(q2_end_local - q2_start_local)

    q1_temp = (q1 - q1_midpoint)*q1_scale_factor
    q2_temp = (q2 - q2_midpoint)*q2_scale_factor
    
    # Bottom domain
    if ((q2_midpoint < -0.66) and (q1_midpoint > -0.66) and (q1_midpoint < 0.33)):

        x_y_bottom_left   = [-radius/np.sqrt(2) - .5*dq1, radius/np.sqrt(2) - .5*dq2]
        x_y_bottom_center = [0,                           radius                    ]
        x_y_bottom_right  = [radius/np.sqrt(2) + .5*dq1,  radius/np.sqrt(2) - .5*dq2]
        
        x_y_left_center   = [-radius/np.sqrt(2) - .5*dq1, (1+radius/np.sqrt(2))/2]
        x_y_right_center  = [ radius/np.sqrt(2) + .5*dq1, (1+radius/np.sqrt(2))/2]
        
        x_y_top_left      = [-radius/np.sqrt(2) - .5*dq1, 1]
        x_y_top_center    = [0,                           1]
        x_y_top_right     = [radius/np.sqrt(2)  + .5*dq1, 1]        
    
#        x_y_bottom_left   = [q1_start_local, q2_start_local]
#        x_y_bottom_center = [q1_center_local, q2_start_local]
#        x_y_bottom_right  = [q1_end_local, q2_start_local]
#        
#        x_y_left_center  = [-radius/np.sqrt(2) - .0*dq1, (-1-radius/np.sqrt(2))/2]
#        x_y_right_center = [radius/np.sqrt(2)  + .0*dq1, (-1-radius/np.sqrt(2))/2]
#        
#        x_y_top_left     = [-radius/np.sqrt(2) - .0*dq1, -radius/np.sqrt(2) + .0*dq2]
#        x_y_top_center   = [q1_center_local, -radius]
#        x_y_top_right    = [radius/np.sqrt(2) + .0*dq1,  -radius/np.sqrt(2) + .0*dq2]
        print ('x_y_bottom_left = ', x_y_bottom_left)
        print ('x_y_bottom_center = ', x_y_bottom_center)
        print ('x_y_bottom_right = ', x_y_bottom_right)
        print ('x_y_left_center = ', x_y_left_center)
        print ('x_y_right_center = ', x_y_right_center)
        print ('x_y_top_left = ', x_y_top_left)
        print ('x_y_top_center = ', x_y_top_center)
        print ('x_y_top_right = ', x_y_top_right)
        
        x, y = quadratic(q1_temp, q2_temp,
                         x_y_bottom_left, x_y_bottom_right, 
                         x_y_top_right, x_y_top_left,
                         x_y_bottom_center, x_y_right_center,
                         x_y_top_center, x_y_left_center,
                         X_Y_bottom_left, X_Y_bottom_right, 
                         X_Y_top_right, X_Y_top_left,
                         X_Y_bottom_center, X_Y_right_center,
                         X_Y_top_center, X_Y_left_center,
                        )

        x = (x/q1_scale_factor) + q1_midpoint
        y = (y/q2_scale_factor) + q2_midpoint

#
#    # Top domain
#    elif ((q2_midpoint > 0.33) and (q1_midpoint > -0.66) and (q1_midpoint < 0.33)):
#        x_y_bottom_left   = [-radius/np.sqrt(2) - .5*dq1, radius/np.sqrt(2) - .5*dq2]
#        x_y_bottom_center = [0,                           radius                    ]
#        x_y_bottom_right  = [radius/np.sqrt(2) + .5*dq1,  radius/np.sqrt(2) - .5*dq2]
#        
#        x_y_left_center   = [-radius/np.sqrt(2) - .5*dq1, (1+radius/np.sqrt(2))/2]
#        x_y_right_center  = [ radius/np.sqrt(2) + .5*dq1, (1+radius/np.sqrt(2))/2]
#        
#        x_y_top_left      = [-radius/np.sqrt(2) - .5*dq1, 1]
#        x_y_top_center    = [0,                           1]
#        x_y_top_right     = [radius/np.sqrt(2)  + .5*dq1, 1]
#        
#        x, y = affine(q1, q2,
#                         x_y_bottom_left, x_y_bottom_right, 
#                         x_y_top_right, x_y_top_left,
#                         x_y_bottom_center, x_y_right_center,
#                         x_y_top_center, x_y_left_center,
#                         X_Y_bottom_left, X_Y_bottom_right, 
#                         X_Y_top_right, X_Y_top_left,
#                         X_Y_bottom_center, X_Y_right_center,
#                         X_Y_top_center, X_Y_left_center,
#                        )
#    
#    
#    # Right domain
#    elif ((q2_midpoint > -0.66) and (q2_midpoint < 0.33) and (q1_midpoint > 0.33)):
#        x_y_bottom_left   = [radius/np.sqrt(2) - .5*dq1, -radius/np.sqrt(2) - .5*dq2]
#        x_y_bottom_center = [(1.+radius/np.sqrt(2))/2,    -radius/np.sqrt(2) - .5*dq2]
#        x_y_bottom_right  = [1.,                          -radius/np.sqrt(2) - .5*dq2]
#        
#        x_y_left_center  = [radius, 0.]
#        x_y_right_center = [1., 0.]
#        
#        x_y_top_left     = [radius/np.sqrt(2) - .5*dq1, radius/np.sqrt(2) + .5*dq2]
#        x_y_top_center   = [(1.+radius/np.sqrt(2))/2,    radius/np.sqrt(2) + .5*dq2]
#        x_y_top_right    = [1.,                          radius/np.sqrt(2) + .5*dq2]
#        
#        x, y = affine(q1, q2,
#                         x_y_bottom_left, x_y_bottom_right, 
#                         x_y_top_right, x_y_top_left,
#                         x_y_bottom_center, x_y_right_center,
#                         x_y_top_center, x_y_left_center,
#                         X_Y_bottom_left, X_Y_bottom_right, 
#                         X_Y_top_right, X_Y_top_left,
#                         X_Y_bottom_center, X_Y_right_center,
#                         X_Y_top_center, X_Y_left_center,
#                        )
#
#    # Left domain
#    elif ((q2_midpoint > -0.66) and (q2_midpoint < 0.33) and (q1_midpoint < -0.66)):
#        x_y_bottom_left   = [-1.,                          -radius/np.sqrt(2) - .5*dq2]
#        x_y_bottom_center = [-(1.+radius/np.sqrt(2))/2,    -radius/np.sqrt(2) - .5*dq2]
#        x_y_bottom_right  = [-radius/np.sqrt(2) + .5*dq1, -radius/np.sqrt(2) - .5*dq2]
#        
#        x_y_left_center  = [-1.,  0.]
#        x_y_right_center = [-radius, 0.]
#        
#        x_y_top_left     = [-1.,                          radius/np.sqrt(2) + .5*dq2]
#        x_y_top_center   = [-(1.+radius/np.sqrt(2))/2,    radius/np.sqrt(2) + .5*dq2]
#        x_y_top_right    = [-radius/np.sqrt(2) + .5*dq1, radius/np.sqrt(2) + .5*dq2]
#        
#        x, y = affine(q1, q2,
#                         x_y_bottom_left, x_y_bottom_right, 
#                         x_y_top_right, x_y_top_left,
#                         x_y_bottom_center, x_y_right_center,
#                         x_y_top_center, x_y_left_center,
#                         X_Y_bottom_left, X_Y_bottom_right, 
#                         X_Y_top_right, X_Y_top_left,
#                         X_Y_bottom_center, X_Y_right_center,
#                         X_Y_top_center, X_Y_left_center,
#                        )
#
#    # Center domain
#    elif ((q2_midpoint > -0.66) and (q2_midpoint < 0.33) and (q1_midpoint > -0.66) and (q1_midpoint < 0.33)):
#        x_y_bottom_left   = [-radius/np.sqrt(2), -radius/np.sqrt(2)]
#        x_y_bottom_center = [0.,                  -radius - .5*dq2  ]
#        x_y_bottom_right  = [radius/np.sqrt(2),  -radius/np.sqrt(2)]
#        
#        x_y_left_center  = [-radius - .5*dq1, 0]
#        x_y_right_center = [ radius + .5*dq1, 0]
#        
#        x_y_top_left     = [-radius/np.sqrt(2), radius/np.sqrt(2)]
#        x_y_top_center   = [0.,                  radius + .5*dq2  ]
#        x_y_top_right    = [radius/np.sqrt(2),  radius/np.sqrt(2)]
#        
#        x, y = affine(q1, q2,
#                         x_y_bottom_left, x_y_bottom_right, 
#                         x_y_top_right, x_y_top_left,
#                         x_y_bottom_center, x_y_right_center,
#                         x_y_top_center, x_y_left_center,
#                         X_Y_bottom_left, X_Y_bottom_right, 
#                         X_Y_top_right, X_Y_top_left,
#                         X_Y_bottom_center, X_Y_right_center,
#                         X_Y_top_center, X_Y_left_center,
#                        )


    return(x, y)

