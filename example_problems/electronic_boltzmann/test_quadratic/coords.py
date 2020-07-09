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

    x_y_bottom_left   = [-1, -1]
    x_y_bottom_center = [0. , -1.25]
    x_y_bottom_right  = [1 , -1]
    
    x_y_left_center  = [-1.25, 0.]
    x_y_right_center = [0.75 , 0]
    
    x_y_top_left     = [-1, 1.]
    x_y_top_center   = [0. , 0.75]
    x_y_top_right    = [1 , 1]

    if (q1_start_local_left != None and q2_start_local_bottom != None):
        
        x, y, jacobian = quadratic(q1, q2,
                                   x_y_bottom_left,   x_y_bottom_right, 
                                   x_y_top_right,     x_y_top_left,
                                   x_y_bottom_center, x_y_right_center,
                                   x_y_top_center,    x_y_left_center,
                                   q1_start_local_left, 
                                   q2_start_local_bottom,
                                  )
        if (return_jacobian):
            return(x, y, jacobian)
        else:
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")
