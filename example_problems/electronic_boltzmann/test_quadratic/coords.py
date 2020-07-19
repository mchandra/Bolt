import numpy as np
import arrayfire as af
import pylab as pl

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

    N_g = domain.N_ghost

    # Default initialisation to rectangular grid
    x = q1
    y = q2
    jacobian = [[1. + 0.*q1,      0.*q1],
                [     0.*q1, 1. + 0.*q1]
               ]
    [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]] = jacobian

    # Radius and center of circular region
    radius          = 0.5
    center          = [0, 0]

    if (q1_start_local_left != None and q2_start_local_bottom != None):

        x_0     = -0.33333333#-radius/np.sqrt(2)

        x_y_bottom_left   = [x_0,        0.]
        x_y_bottom_center = [0.,         0.]
        x_y_bottom_right  = [0.33333333, 0.]

        x_y_left_center   = [x_0,        (1.33333333)/2]
        x_y_right_center  = [0.33333333, (1.33333333)/2]

        x_y_top_left      = [x_0,        1.33333333]
        x_y_top_center    = [0.,         1.33333333]
        x_y_top_right     = [0.33333333, 1.33333333]


        x, y, jacobian = quadratic(q1, q2,
                           x_y_bottom_left,   x_y_bottom_right,
                           x_y_top_right,     x_y_top_left,
                           x_y_bottom_center, x_y_right_center,
                           x_y_top_center,    x_y_left_center,
                           q1_start_local_left,
                           q2_start_local_bottom,
                          )


        pl.plot(af.moddims(dx_dq1[0, 0, :, N_g], q1.dims()[2]).to_ndarray(), '-o', color = 'C0', alpha = 0.5, label = "dx_dq1")
        pl.plot(af.moddims(dy_dq1[0, 0, :, N_g], q1.dims()[2]).to_ndarray(), '-o', color = 'k', alpha = 0.5, label = "dy_dq1")
        pl.plot(af.moddims(dx_dq2[0, 0, :, N_g], q1.dims()[2]).to_ndarray(), '-o', color = 'C2', alpha = 0.5, label = "dx_dq2")
        pl.plot(af.moddims(dy_dq2[0, 0, :, N_g], q1.dims()[2]).to_ndarray(), '-o', color = 'C3', alpha = 0.5, label = "dy_dq2")

        pl.legend(loc='best')

        pl.savefig("/home/quazartech/bolt/example_problems/electronic_boltzmann/test_quadratic/iv.png")
        pl.clf()

        if (return_jacobian):
            return (x, y, jacobian)
        else: 
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")


