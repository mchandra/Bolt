import numpy as np
import arrayfire as af
import domain
import params

from bolt.lib.utils.coord_transformation import affine, jacobian_dx_dq

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
    #x = q1*(2+q2)
    #y = q2
    jacobian = None#[[1. + 0.*q1,      0.*q1],
                #[     0.*q1, 1. + 0.*q1]
               #]


    if (q1_start_local_left != None and q2_start_local_bottom != None):

#        X_Y_top_right    = [1,   1]
#        X_Y_top_left     = [-1,  1]
#        X_Y_bottom_left  = [-1, -1]
#        X_Y_bottom_right = [1,  -1]
#
#        x_y_top_right    = [1,   1]
#        x_y_top_left     = [-1,  1]
#        x_y_bottom_left  = [-1, -0.5]
#        x_y_bottom_right = [1,  -1]
#        
#        x, y =  affine(q1, q2,
#                       x_y_bottom_left, x_y_bottom_right, 
#                       x_y_top_right,   x_y_top_left,
#                       X_Y_bottom_left, X_Y_bottom_right, 
#                       X_Y_top_right,   X_Y_top_left,
#                      )
#        # Numerically calculate Jacobian
#        jacobian = None

        if (return_jacobian):
            return (x, y, jacobian)
        else: 
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")


