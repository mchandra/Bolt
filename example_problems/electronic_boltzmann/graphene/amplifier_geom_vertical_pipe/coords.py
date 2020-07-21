import numpy as np
import arrayfire as af

def get_cartesian_coords(q1, q2, 
                         q1_start_local_left=None, 
                         q2_start_local_bottom=None,
                         return_jacobian = False
                        ):

    q1_midpoint = 0.5*(af.max(q1) + af.min(q1))
    q2_midpoint = 0.5*(af.max(q2) + af.min(q2))

    x = q1
    y = q2
    jacobian = [[1. + 0.*q1,      0.*q1],
                [     0.*q1, 1. + 0.*q1]
               ]

#    if (q1_midpoint > 1.) and (q1_midpoint < 4.):
#        y = 0.5*q2+0.125

    if (q1_start_local_left != None and q2_start_local_bottom != None):

        if (return_jacobian):
            return(x, y, jacobian)
        else:
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")

    return(x, y)

