import numpy as np
import arrayfire as af
import pylab as pl

import domain
import params

from bolt.lib.utils.coord_transformation import quadratic_test

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

        N_q1     = q1.dims()[2] - 2*N_g # Manually apply quadratic transformation for each zone along q1
        N_q2     = q2.dims()[3] - 2*N_g # Manually apply quadratic transformation for each zone along q1
        N        = N_q1
        x_0     = -0.33333333#-radius/np.sqrt(2)

        # Initialize to zero
        x = 0*q1
        y = 0*q2


        # Loop over each zone in x
        for i in range(N_g, N_q1 + N_g):

            index = i - N_g # Index of the vertical slice, left-most being 0

            # q1, q2 grid slices to be passed into quadratic for transformation
            q1_slice = q1[0, 0, i, N_g:-N_g]
            q2_slice = q2[0, 0, i, N_g:-N_g]

            # Compute the x, y points using which the transformation will be defined
            # x, y nodes remain the same for each point on a vertical slice
            x_n           = x_0 + 0.66666666*index/N # Bottom-left
            y_n           = np.sqrt(radius**2 - x_n**2)


            x_n_plus_1    = x_0 + 0.66666666*(index+1)/N # Bottom-right
            y_n_plus_1    = np.sqrt(radius**2 - x_n_plus_1**2)

            x_n_plus_half = x_0 + 0.66666666*(index+0.5)/N # Bottom-center
            y_n_plus_half = np.sqrt(radius**2 - x_n_plus_half**2)


            x_y_bottom_left   = [x_n,           y_n]
            x_y_bottom_center = [x_n_plus_half, y_n_plus_half]
            x_y_bottom_right  = [x_n_plus_1,    y_n_plus_1]

            x_y_left_center   = [x_n,        (1.+2*y_n)/2]
            x_y_right_center  = [x_n_plus_1, (1.+2*y_n_plus_1)/2]

            x_y_top_left      = [x_n,           1.+y_n]
            x_y_top_center    = [x_n_plus_half, 1.+y_n_plus_half]
            x_y_top_right     = [x_n_plus_1,    1.+y_n_plus_1]


            for j in range(N_g, N_q2 + N_g):

                # Get the transformation (x_i, y_i) for each point (q1_i, q2_i)
                q1_i = q1[0, 0, i, j]
                q2_i = q2[0, 0, i, j]

                x_i, y_i, jacobian_i = quadratic_test(q1_i, q2_i, q1_slice, q2_slice,
                                   x_y_bottom_left,   x_y_bottom_right,
                                   x_y_top_right,     x_y_top_left,
                                   x_y_bottom_center, x_y_right_center,
                                   x_y_top_center,    x_y_left_center,
                                   q1_start_local_left + index*domain.dq1,
                                   q2_start_local_bottom,
                                  )

                # Reconstruct the x,y grid from the loop
                x[0, 0, i, j] = x_i
                y[0, 0, i, j] = y_i

                # TODO : Reconstruct jacobian
                [[dx_dq1_i, dx_dq2_i], [dy_dq1_i, dy_dq2_i]] = jacobian_i
                dx_dq1[0, 0, i, j] = dx_dq1_i.scalar()
                dx_dq2[0, 0, i, j] = dx_dq2_i.scalar()
                dy_dq1[0, 0, i, j] = dy_dq1_i.scalar()
                dy_dq2[0, 0, i, j] = dy_dq2_i.scalar()

            pl.plot(af.moddims(dx_dq1[0, 0, i, :], q1.dims()[2]).to_ndarray(), '-o', color = 'C0', alpha = 0.1, label = "dx_dq1")
            pl.plot(af.moddims(dy_dq1[0, 0, i, :], q1.dims()[2]).to_ndarray(), '-o', color = 'k',  alpha = 0.1, label = "dy_dq1")
            pl.plot(af.moddims(dx_dq2[0, 0, i, :], q1.dims()[2]).to_ndarray(), '-o', color = 'C2', alpha = 0.1, label = "dx_dq2")
            pl.plot(af.moddims(dy_dq2[0, 0, i, :], q1.dims()[2]).to_ndarray(), '-o', color = 'C3', alpha = 0.1, label = "dy_dq2")

        jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]

#        pl.plot(af.moddims(dx_dq1[0, 0, :, N_g], q1.dims()[2]).to_ndarray(), '-o', color = 'C0', alpha = 0.5, label = "dx_dq1")
#        pl.plot(af.moddims(dy_dq1[0, 0, :, N_g], q1.dims()[2]).to_ndarray(), '-o', color = 'k', alpha = 0.5, label = "dy_dq1")
#        pl.plot(af.moddims(dx_dq2[0, 0, :, N_g], q1.dims()[2]).to_ndarray(), '-o', color = 'C2', alpha = 0.5, label = "dx_dq2")
#        pl.plot(af.moddims(dy_dq2[0, 0, :, N_g], q1.dims()[2]).to_ndarray(), '-o', color = 'C3', alpha = 0.5, label = "dy_dq2")

        #pl.legend(loc='best')

        pl.savefig("/home/quazartech/bolt/example_problems/electronic_boltzmann/test_quadratic_2/iv.png")
        pl.clf()

        if (return_jacobian):
            return (x, y, jacobian)
        else: 
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")


