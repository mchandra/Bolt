import numpy as np
import arrayfire as af
import domain
import params

from bolt.lib.utils.coord_transformation import quadratic

def get_cartesian_coords(q1, q2, 
                         q1_start_local_left=None, 
                         q2_start_local_bottom=None,
                         return_jacobian = False
                        ):

    q1_midpoint = 0.5*(af.max(q1) + af.min(q1))
    q2_midpoint = 0.5*(af.max(q2) + af.min(q2))

    d_q1          = (q1[0, 0, 1, 0] - q1[0, 0, 0, 0]).scalar()
    d_q2          = (q2[0, 0, 0, 1] - q2[0, 0, 0, 0]).scalar()

    # Default initializsation to rectangular grid
    x = q1
    y = q2

    # Radius and center of circular region
    radius          = 0.5
    center          = [0, 0]

    shift_x = 0.*d_q1/2
    shift_y = 0.*d_q2/2

    x_y_circle_top_left     = [-radius/np.sqrt(2) + shift_x, radius/np.sqrt(2)  - shift_y]
    x_y_circle_bottom_left  = [-radius/np.sqrt(2) + shift_x, -radius/np.sqrt(2) + shift_y]
    x_y_circle_top_right    = [radius/np.sqrt(2)  - shift_x, radius/np.sqrt(2)  - shift_y]
    x_y_circle_bottom_right = [radius/np.sqrt(2)  - shift_x, -radius/np.sqrt(2) + shift_y]

    print ('coords.py, q1_start_local_left, q1[0]', params.rank, q1_start_local_left, q1[0, 0, domain.N_ghost, 0].scalar())

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
            
            x_y_top_left     = x_y_circle_bottom_left
            x_y_top_center   = [0                 , -radius           ]
            x_y_top_right    = x_y_circle_bottom_right

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
            x_y_top_right    = x_y_circle_bottom_left
    
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
            
            x_y_top_left     = x_y_circle_bottom_right
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

            x_y_bottom_left   = x_y_circle_top_left
            x_y_bottom_center = [0,                  radius]
            x_y_bottom_right  = x_y_circle_top_right
            
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
            x_y_bottom_right  = x_y_circle_top_left
            
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

            x_y_bottom_left   = x_y_circle_top_right
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
        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint > 0.33)):

            x_y_bottom_left   = x_y_circle_bottom_right
            x_y_bottom_center = [(1.+radius/np.sqrt(2))/2,    -radius/np.sqrt(2)]
            x_y_bottom_right  = [1.,                          -radius/np.sqrt(2)]
            
            x_y_left_center  = [radius, 0.]
            x_y_right_center = [1.,     0.]
            
            x_y_top_left     = x_y_circle_top_right
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
        elif ((q2_midpoint > -0.33) and (q2_midpoint < 0.33) and (q1_midpoint < -0.33)):

            x_y_bottom_left   = [-1.,                          -radius/np.sqrt(2)]
            x_y_bottom_center = [-(1.+radius/np.sqrt(2))/2,    -radius/np.sqrt(2)]
            x_y_bottom_right  = x_y_circle_bottom_left
            
            x_y_left_center  = [-1.,     0.]
            x_y_right_center = [-radius, 0.]
            
            x_y_top_left     = [-1.,                          radius/np.sqrt(2)]
            x_y_top_center   = [-(1.+radius/np.sqrt(2))/2,    radius/np.sqrt(2)]
            x_y_top_right    = x_y_circle_top_left
            
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

            x_y_bottom_left   = x_y_circle_bottom_left
            x_y_bottom_center = [0.,                  -radius]
            x_y_bottom_right  = x_y_circle_bottom_right
            
            x_y_left_center  = [-radius, 0]
            x_y_right_center = [ radius, 0]
            
            x_y_top_left     = x_y_circle_top_left
            x_y_top_center   = [0.,                  radius]
            x_y_top_right    = x_y_circle_top_right
            
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


