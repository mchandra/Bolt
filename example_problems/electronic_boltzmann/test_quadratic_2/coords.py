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
        N_i      = N_q1
        N_j      = N_q2

        # Bottom-left most point of the transformation
        x_0      = -radius/np.sqrt(2)
        y_0      = np.sqrt(radius**2 - x_0**2)

        # Initialize to zero
        x = 0*q1
        y = 0*q2


        # Loop over each zone in x
        #for i in range(N_g, N_q1 + N_g):
        for i in range(0, N_q1 + 2*N_g):

            #for j in range(N_g, N_q2 + N_g):
            for j in range(0, N_q2 + 2*N_g):
                index_i = i - N_g # Index of the vertical slice, left-most being 0
                index_j = j - N_g # Index of the horizontal slice, bottom-most being 0
    
                # q1, q2 grid slices to be passed into quadratic for transformation
                q1_slice = q1[0, 0, i, j]
                q2_slice = q2[0, 0, i, j]

                x_step = 0.66666666/N_i
                y_step = 1./N_j
    
                # Compute the x, y points using which the transformation will be defined
                # x, y nodes remain the same for each point on a vertical slice
 
#                y_n           = y_0 + y_step*index_j   # Bottom
#                y_n_plus_half = y_0 + y_step*(index_j+0.5) # y-center
#                y_n_plus_1    = y_0 + y_step*(index_j+1) # Top

                x_n           = x_0 + x_step*index_i       #+ x_step*(np.abs(index_j - N_j/2.))# Left
                x_n_plus_half = x_0 + x_step*(index_i+0.5) #+ x_step*(np.abs(index_j - N_j/2.))# x-center
                x_n_plus_1    = x_0 + x_step*(index_i+1)   #+ x_step*(np.abs(index_j - N_j/2.))# Right

                x_y_bottom_left   = [x_n,           np.sqrt(np.abs(radius**2 - x_n**2))           + index_j*y_step]
                x_y_bottom_center = [x_n_plus_half, np.sqrt(np.abs(radius**2 - x_n_plus_half**2)) + index_j*y_step]
                x_y_bottom_right  = [x_n_plus_1,    np.sqrt(np.abs(radius**2 - x_n_plus_1**2))    + index_j*y_step]
    
                x_y_left_center   = [x_n,           np.sqrt(np.abs(radius**2 - x_n**2))        + (index_j+0.5)*y_step]
                x_y_right_center  = [x_n_plus_1,    np.sqrt(np.abs(radius**2 - x_n_plus_1**2)) + (index_j+0.5)*y_step]
    
                x_y_top_left      = [x_n,           np.sqrt(np.abs(radius**2 - x_n**2))           + (index_j+1)*y_step]
                x_y_top_center    = [x_n_plus_half, np.sqrt(np.abs(radius**2 - x_n_plus_half**2)) + (index_j+1)*y_step]
                x_y_top_right     = [x_n_plus_1,    np.sqrt(np.abs(radius**2 - x_n_plus_1**2))    + (index_j+1)*y_step]

                # Get the transformation (x_i, y_i) for each point (q1_i, q2_i)
                q1_i = q1[0, 0, i, j]
                q2_i = q2[0, 0, i, j]

                x_i, y_i, jacobian_i = quadratic_test(q1_i, q2_i, q1_slice, q2_slice,
                                   x_y_bottom_left,   x_y_bottom_right,
                                   x_y_top_right,     x_y_top_left,
                                   x_y_bottom_center, x_y_right_center,
                                   x_y_top_center,    x_y_left_center,
                                   q1_start_local_left + index_i*domain.dq1,
                                   q2_start_local_bottom + index_j*domain.dq2
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


