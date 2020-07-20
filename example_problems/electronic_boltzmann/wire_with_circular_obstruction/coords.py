import numpy as np
import arrayfire as af
import domain

from bolt.lib.utils.coord_transformation import quadratic

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

    top    = domain.q2_end
    bottom = domain.q2_start
    left   = domain.q1_start
    right  = domain.q1_end


    if (q1_start_local_left != None and q2_start_local_bottom != None):

        # Bottom-center domain
        if ((q2_midpoint < -0.33) and (q1_midpoint > -0.33) and (q1_midpoint < 0.33)):

            # Note : Never specify the x, y coordinates below in terms of q1 and q2 coordinates. Specify only in
            # physical x, y values.
    
            x_y_bottom_left   = [-radius/np.sqrt(2), -1]
            x_y_bottom_center = [0                 , -1]
            x_y_bottom_right  = [radius/np.sqrt(2) , -1]
            
            x_y_left_center  = [-radius/np.sqrt(2), (-1-radius/np.sqrt(2))/2]
            x_y_right_center = [ radius/np.sqrt(2), (-1-radius/np.sqrt(2))/2]
            
            x_y_top_left     = [-radius/np.sqrt(2), -radius/np.sqrt(2)]
            x_y_top_center   = [0                 , -radius           ]
            x_y_top_right    = [radius/np.sqrt(2) , -radius/np.sqrt(2)]     

            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )
            
    
        # Bottom-left domain
        elif ((q2_midpoint < -0.33) and (q1_midpoint > -1) and (q1_midpoint < -0.33)):
    
            x_y_bottom_left   = [-1,                 -1]
            x_y_bottom_center = [(-1-radius/np.sqrt(2))/2,    -1]
            x_y_bottom_right  = [-radius/np.sqrt(2), -1]
    
            x_y_left_center  = [-1,                 (-1-radius/np.sqrt(2))/2]
            x_y_right_center = [-radius/np.sqrt(2), (-1-radius/np.sqrt(2))/2]        
            
            x_y_top_left     = [-1,                        -radius/np.sqrt(2)]
            x_y_top_center   = [-(1.+radius/np.sqrt(2))/2, -radius/np.sqrt(2)]
            x_y_top_right    = [-radius/np.sqrt(2),        -radius/np.sqrt(2)] 
    
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )

    
        # Bottom-right domain
        elif ((q2_midpoint < -0.33) and (q1_midpoint > 0.33) and (q1_midpoint < 1.)):
    
            x_y_bottom_left   = [radius/np.sqrt(2),          -1]
            x_y_bottom_center = [(1+radius/np.sqrt(2))/2,   -1]
            x_y_bottom_right  = [1,                          -1]
    
            x_y_left_center   = [ radius/np.sqrt(2), (-1-radius/np.sqrt(2))/2]
            x_y_right_center  = [1,                  (-1-radius/np.sqrt(2))/2]
            
            x_y_top_left     = [radius/np.sqrt(2),         -radius/np.sqrt(2)]
            x_y_top_center   = [(1.+radius/np.sqrt(2))/2,  -radius/np.sqrt(2)]
            x_y_top_right    = [1,                         -radius/np.sqrt(2)] 
    
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )
    

        # Top-center domain
        elif ((q2_midpoint > 0.33) and (q1_midpoint > -0.33) and (q1_midpoint < 0.33)):

            x_y_bottom_left   = [-radius/np.sqrt(2), radius/np.sqrt(2)]
            x_y_bottom_center = [0,                  radius]
            x_y_bottom_right  = [radius/np.sqrt(2),  radius/np.sqrt(2)]
            
            x_y_left_center   = [-radius/np.sqrt(2), (1+radius/np.sqrt(2))/2]
            x_y_right_center  = [ radius/np.sqrt(2), (1+radius/np.sqrt(2))/2]
            
            x_y_top_left      = [-radius/np.sqrt(2), 1]
            x_y_top_center    = [0,                  1]
            x_y_top_right     = [radius/np.sqrt(2),  1]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )


        # Top-left domain
        elif ((q2_midpoint > 0.33) and (q1_midpoint > -1) and (q1_midpoint < -0.33)):

            x_y_bottom_left   = [-1,                         radius/np.sqrt(2)]
            x_y_bottom_center = [-(1.+radius/np.sqrt(2))/2,  radius/np.sqrt(2)]
            x_y_bottom_right  = [-radius/np.sqrt(2),         radius/np.sqrt(2)]
            
            x_y_left_center  = [-1,                 (1+radius/np.sqrt(2))/2]
            x_y_right_center = [-radius/np.sqrt(2), (1+radius/np.sqrt(2))/2]
    
            x_y_top_left      = [-1,                          1]
            x_y_top_center    = [-(1+radius/np.sqrt(2))/2,    1] 
            x_y_top_right     = [-radius/np.sqrt(2), 1]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )

        
        # Top-right domain
        elif ((q2_midpoint > 0.33) and (q1_midpoint > 0.33) and (q1_midpoint < 1)):

            x_y_bottom_left   = [radius/np.sqrt(2),         radius/np.sqrt(2)]
            x_y_bottom_center = [(1.+radius/np.sqrt(2))/2,  radius/np.sqrt(2)]
            x_y_bottom_right  = [1,                         radius/np.sqrt(2)]
            
            x_y_right_center = [1.,                 (1+radius/np.sqrt(2))/2]
            x_y_left_center  = [ radius/np.sqrt(2), (1+radius/np.sqrt(2))/2]
    
            x_y_top_left      = [radius/np.sqrt(2), 1]
            x_y_top_center    = [(1+radius/np.sqrt(2))/2,   1]
            x_y_top_right     = [1,                 1]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )


        # Center-Right domain
        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint > 0.33) and (q1_midpoint < 1.)):

            x_y_bottom_left   = [radius/np.sqrt(2),           -radius/np.sqrt(2)]
            x_y_bottom_center = [(1.+radius/np.sqrt(2))/2,    -radius/np.sqrt(2)]
            x_y_bottom_right  = [1.,                          -radius/np.sqrt(2)]
            
            x_y_left_center  = [radius, 0.]
            x_y_right_center = [1.,     0.]
            
            x_y_top_left     = [radius/np.sqrt(2),           radius/np.sqrt(2) ]
            x_y_top_center   = [(1.+radius/np.sqrt(2))/2,    radius/np.sqrt(2) ]
            x_y_top_right    = [1.,                          radius/np.sqrt(2) ]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )
    

        # Center-Left domain
        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint < -0.33) and (q1_midpoint > -1)):

            x_y_bottom_left   = [-1.,                          -radius/np.sqrt(2)]
            x_y_bottom_center = [-(1.+radius/np.sqrt(2))/2,    -radius/np.sqrt(2)]
            x_y_bottom_right  = [-radius/np.sqrt(2),           -radius/np.sqrt(2)  ]
            
            x_y_left_center  = [-1.,     0.]
            x_y_right_center = [-radius, 0.]
            
            x_y_top_left     = [-1.,                          radius/np.sqrt(2)]
            x_y_top_center   = [-(1.+radius/np.sqrt(2))/2,    radius/np.sqrt(2)]
            x_y_top_right    = [-radius/np.sqrt(2),           radius/np.sqrt(2)  ]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )
    

        # Center domain
        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint > -0.33) and (q1_midpoint < 0.33)):

            x_y_bottom_left   = [-radius/np.sqrt(2),  -radius/np.sqrt(2)]
            x_y_bottom_center = [0.,                  -radius]
            x_y_bottom_right  = [radius/np.sqrt(2),   -radius/np.sqrt(2)]
            
            x_y_left_center  = [-radius, 0]
            x_y_right_center = [ radius, 0]
            
            x_y_top_left     = [-radius/np.sqrt(2),  radius/np.sqrt(2)]
            x_y_top_center   = [0.,                  radius]
            x_y_top_right    = [radius/np.sqrt(2),   radius/np.sqrt(2)]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )

        # Left-center extension 
        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint < -1) and (q1_midpoint > left)):

            x_y_bottom_left   = [left,         -radius/np.sqrt(2)]
            x_y_bottom_center = [(left-1)/2.,  -radius/np.sqrt(2)]
            x_y_bottom_right  = [-1,           -radius/np.sqrt(2)  ]
            
            x_y_left_center  = [left, 0.]
            x_y_right_center = [-1, 0.]
            
            x_y_top_left     = [left,        radius/np.sqrt(2)]
            x_y_top_center   = [(left-1)/2., radius/np.sqrt(2)]
            x_y_top_right    = [-1.,         radius/np.sqrt(2)  ]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )

        # Left-bottom extension 
        elif ((q2_midpoint > -1) and (q2_midpoint < -0.33) and (q1_midpoint < -1) and (q1_midpoint > left)):

            x_y_bottom_left   = [left,         -1]
            x_y_bottom_center = [(left-1)/2.,  -1]
            x_y_bottom_right  = [-1,           -1]
            
            x_y_left_center  = [left, (-1-radius/np.sqrt(2))/2]
            x_y_right_center = [-1,   (-1-radius/np.sqrt(2))/2]
            
            x_y_top_left     = [left,        -radius/np.sqrt(2)]
            x_y_top_center   = [(left-1)/2., -radius/np.sqrt(2)]
            x_y_top_right    = [-1.,         -radius/np.sqrt(2)]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )
        # Left-top extension 
        elif ((q2_midpoint > 0.33) and (q2_midpoint < 1) and (q1_midpoint < -1) and (q1_midpoint > left)):

            x_y_bottom_left   = [left,           radius/np.sqrt(2)]
            x_y_bottom_center = [(left-1)/2.,    radius/np.sqrt(2)]
            x_y_bottom_right  = [-1,              radius/np.sqrt(2)  ]
            
            x_y_left_center  = [left, (1+radius/np.sqrt(2))/2]
            x_y_right_center = [-1,   (1+radius/np.sqrt(2))/2]
            
            x_y_top_left     = [left,         1]
            x_y_top_center   = [(left-1)/2.,  1]
            x_y_top_right    = [-1.,          1]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )

    
        # Right-center extension 
        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint < right) and (q1_midpoint > 1)):

            x_y_bottom_left   = [1,             -radius/np.sqrt(2)]
            x_y_bottom_center = [(right+1)/2.,  -radius/np.sqrt(2)]
            x_y_bottom_right  = [right,         -radius/np.sqrt(2)]
            
            x_y_left_center  = [1,     0.]
            x_y_right_center = [right, 0.]
            
            x_y_top_left     = [1.,           radius/np.sqrt(2)]
            x_y_top_center   = [(right+1)/2., radius/np.sqrt(2)]
            x_y_top_right    = [right,        radius/np.sqrt(2)]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )
    
        # Right-bottom extension 
        elif ((q2_midpoint > -1) and (q2_midpoint < -0.33) and (q1_midpoint < right) and (q1_midpoint > 1)):

            x_y_bottom_left   = [1,             -1]
            x_y_bottom_center = [(right+1)/2.,  -1]
            x_y_bottom_right  = [right,         -1]
            
            x_y_left_center  = [1,     -(1+radius/np.sqrt(2))/2]
            x_y_right_center = [right, -(1+radius/np.sqrt(2))/2]
            
            x_y_top_left     = [1.,           -radius/np.sqrt(2)]
            x_y_top_center   = [(right+1)/2., -radius/np.sqrt(2)]
            x_y_top_right    = [right,        -radius/np.sqrt(2)  ]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )

        # Right-top extension 
        elif ((q2_midpoint > 0.33) and (q2_midpoint < 1) and (q1_midpoint < right) and (q1_midpoint > 1)):

            x_y_bottom_left   = [1,              radius/np.sqrt(2)]
            x_y_bottom_center = [(right+1)/2.,   radius/np.sqrt(2)]
            x_y_bottom_right  = [right,          radius/np.sqrt(2)]
            
            x_y_left_center  = [1,     (1+radius/np.sqrt(2))/2]
            x_y_right_center = [right, (1+radius/np.sqrt(2))/2]
            
            x_y_top_left     = [1.,            1]
            x_y_top_center   = [(right+1)/2.,  1]
            x_y_top_right    = [right,         1]
            
            x, y, jacobian = quadratic(q1, q2,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom,
                                      )
        if (return_jacobian):
            return (x, y, jacobian)
        else: 
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")


