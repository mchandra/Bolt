import numpy as np
import arrayfire as af
import domain
import params

from bolt.lib.utils.coord_transformation import quadratic_test, quadratic

def get_cartesian_coords(q1, q2, 
                         q1_start_local_left=None, 
                         q2_start_local_bottom=None,
                         return_jacobian = False
                        ):

    q1_midpoint = 0.5*(af.max(q1) + af.min(q1))
    q2_midpoint = 0.5*(af.max(q2) + af.min(q2))

    d_q1          = (q1[0, 0, 1, 0] - q1[0, 0, 0, 0]).scalar()
    d_q2          = (q2[0, 0, 0, 1] - q2[0, 0, 0, 0]).scalar()

    N_g      = domain.N_ghost
    N_q1     = q1.dims()[2] - 2*N_g # Manually apply quadratic transformation for each zone along q1
    N_q2     = q2.dims()[3] - 2*N_g # Manually apply quadratic transformation for each zone along q1

    # Default initializsation to rectangular grid
    x = q1
    y = q2
    jacobian = [[1. + 0.*q1,      0.*q1],
                [     0.*q1, 1. + 0.*q1]
               ]

    [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]] = jacobian

    # Radius and center of circular region
    radius          = 0.166666
    center          = [0, 0]

    shift_x = 0.*d_q1/2
    shift_y = 0.*d_q2/2

    x_y_circle_top_left     = [-radius/np.sqrt(2) + shift_x, radius/np.sqrt(2)  - shift_y]
    x_y_circle_bottom_left  = [-radius/np.sqrt(2) + shift_x, -radius/np.sqrt(2) + shift_y]
    x_y_circle_top_right    = [radius/np.sqrt(2)  - shift_x, radius/np.sqrt(2)  - shift_y]
    x_y_circle_bottom_right = [radius/np.sqrt(2)  - shift_x, -radius/np.sqrt(2) + shift_y]

    if (q1_start_local_left != None and q2_start_local_bottom != None):

        # Bottom-center domain
        if ((q2_midpoint < -0.1666) and (q1_midpoint > -0.1666) and (q1_midpoint < 0.1666)):

            # Note : Never specify the x, y coordinates below in terms of q1 and q2 coordinates. Specify only in
            # physical x, y values.
    
            N        = N_q1 
            x_0     = -radius/np.sqrt(2)
    
            # Initialize to zero
            x = 0*q1
            y = 0*q2
    
    
            # Loop over each zone in x
            for i in range(0, N_q1 + 2*N_g):
    
                index = i - N_g # Index of the vertical slice, left-most being 0
    
                # q1, q2 grid slices to be passed into quadratic for transformation
                q1_slice = q1[0, 0, i, N_g:-N_g]
                q2_slice = q2[0, 0, i, N_g:-N_g]
    
                # Compute the x, y points using which the transformation will be defined
                # x, y nodes remain the same for each point on a vertical slice
                x_n           = x_0 + np.sqrt(2)*radius*index/N # Top-left
                y_n           = -np.sqrt(radius**2 - x_n**2)
                
    
                x_n_plus_1    = x_0 + np.sqrt(2)*radius*(index+1)/N # Top-right
                y_n_plus_1    = -np.sqrt(radius**2 - x_n_plus_1**2)
    
                x_n_plus_half = x_0 + np.sqrt(2)*radius*(index+0.5)/N # Top-center
                y_n_plus_half = -np.sqrt(radius**2 - x_n_plus_half**2)
    
    
                x_y_bottom_left   = [x_n,           -0.75]
                x_y_bottom_center = [x_n_plus_half, -0.75]
                x_y_bottom_right  = [x_n_plus_1,    -0.75]
        
                x_y_left_center   = [x_n,        (-0.75+y_n)/2]
                x_y_right_center  = [x_n_plus_1, (-0.75+y_n_plus_1)/2]
        
                x_y_top_left      = [x_n,           y_n]
                x_y_top_center    = [x_n_plus_half, y_n_plus_half]
                x_y_top_right     = [x_n_plus_1,    y_n_plus_1]
    
    
                for j in range(0, N_q2 + 2*N_g):
    
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
    
            jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]
            
    
        # Bottom-left domain
        elif ((q2_midpoint < -0.1666) and (q1_midpoint > -0.5) and (q1_midpoint < -0.1666)):

            N        = N_q1 
            x_0      = -0.5
    
            # Initialize to zero
            x = 0*q1
            y = 0*q2
    
    
            # Loop over each zone in x
            for i in range(0, N_q1 + 2*N_g):
    
                index = i - N_g # Index of the vertical slice, left-most being 0
    
                # q1, q2 grid slices to be passed into quadratic for transformation
                q1_slice = q1[0, 0, i, N_g:-N_g]
                q2_slice = q2[0, 0, i, N_g:-N_g]
    
                # Compute the x, y points using which the transformation will be defined
                # x, y nodes remain the same for each point on a vertical slice
                x_n           = x_0 + (0.5-radius/np.sqrt(2))*index/N # Top-left
                y_n           = -radius/np.sqrt(2)
                
    
                x_n_plus_1    = x_0 + (0.5-radius/np.sqrt(2))*(index+1)/N # Top-right
                y_n_plus_1    = -radius/np.sqrt(2)
    
                x_n_plus_half = x_0 + (0.5-radius/np.sqrt(2))*(index+0.5)/N # Top-center
                y_n_plus_half = -radius/np.sqrt(2)
    
    
                x_y_bottom_left   = [x_n,           -0.75]
                x_y_bottom_center = [x_n_plus_half, -0.75]
                x_y_bottom_right  = [x_n_plus_1,    -0.75]
        
                x_y_left_center   = [x_n,        (-0.75+y_n)/2]
                x_y_right_center  = [x_n_plus_1, (-0.75+y_n_plus_1)/2]
        
                x_y_top_left      = [x_n,           y_n]
                x_y_top_center    = [x_n_plus_half, y_n_plus_half]
                x_y_top_right     = [x_n_plus_1,    y_n_plus_1]
    
    
                for j in range(0, N_q2 + 2*N_g):
    
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
    
            jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]
    
        # Bottom-right domain
        elif ((q2_midpoint < -0.1666) and (q1_midpoint > 0.1666) and (q1_midpoint < 0.5)):
    
            N       = N_q1 
            x_0     = radius/np.sqrt(2)
    
            # Initialize to zero
            x = 0*q1
            y = 0*q2
    
    
            # Loop over each zone in x
            for i in range(0, N_q1 + 2*N_g):
    
                index = i - N_g # Index of the vertical slice, left-most being 0
    
                # q1, q2 grid slices to be passed into quadratic for transformation
                q1_slice = q1[0, 0, i, N_g:-N_g]
                q2_slice = q2[0, 0, i, N_g:-N_g]
    
                # Compute the x, y points using which the transformation will be defined
                # x, y nodes remain the same for each point on a vertical slice
                x_n           = x_0 + (0.5-x_0)*index/N # Top-left
                y_n           = -radius/np.sqrt(2)
                
    
                x_n_plus_1    = x_0 + (0.5-x_0)*(index+1)/N # Top-right
                y_n_plus_1    = -radius/np.sqrt(2)
    
                x_n_plus_half = x_0 + (0.5-x_0)*(index+0.5)/N # Top-center
                y_n_plus_half = -radius/np.sqrt(2)
    
    
                x_y_bottom_left   = [x_n,           -0.75]
                x_y_bottom_center = [x_n_plus_half, -0.75]
                x_y_bottom_right  = [x_n_plus_1,    -0.75]
        
                x_y_left_center   = [x_n,        (-0.75+y_n)/2]
                x_y_right_center  = [x_n_plus_1, (-0.75+y_n_plus_1)/2]
        
                x_y_top_left      = [x_n,           y_n]
                x_y_top_center    = [x_n_plus_half, y_n_plus_half]
                x_y_top_right     = [x_n_plus_1,    y_n_plus_1]
    
    
                for j in range(0, N_q2 + 2*N_g):
    
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
    
            jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]


        # Top-center domain
        if ((q2_midpoint > 0.1666) and (q1_midpoint > -0.1666) and (q1_midpoint < 0.1666)):

            N        = N_q1 
            x_0     = -radius/np.sqrt(2)
    
            # Initialize to zero
            x = 0*q1
            y = 0*q2
    
    
            # Loop over each zone in x
            for i in range(0, N_q1 + 2*N_g):
    
                index = i - N_g # Index of the vertical slice, left-most being 0
    
                # q1, q2 grid slices to be passed into quadratic for transformation
                q1_slice = q1[0, 0, i, N_g:-N_g]
                q2_slice = q2[0, 0, i, N_g:-N_g]
    
                # Compute the x, y points using which the transformation will be defined
                # x, y nodes remain the same for each point on a vertical slice
                x_n           = x_0 + np.sqrt(2)*radius*index/N # Bottom-left
                y_n           = np.sqrt(radius**2 - x_n**2)
                
    
                x_n_plus_1    = x_0 + np.sqrt(2)*radius*(index+1)/N # Bottom-right
                y_n_plus_1    = np.sqrt(radius**2 - x_n_plus_1**2)
    
                x_n_plus_half = x_0 + np.sqrt(2)*radius*(index+0.5)/N # Bottom-center
                y_n_plus_half = np.sqrt(radius**2 - x_n_plus_half**2)
    
    
                x_y_bottom_left   = [x_n,           y_n]
                x_y_bottom_center = [x_n_plus_half, y_n_plus_half]
    
                x_y_bottom_right  = [x_n_plus_1,    y_n_plus_1]
        
                x_y_left_center   = [x_n,        (0.5+y_n)/2]
                x_y_right_center  = [x_n_plus_1, (0.5+y_n_plus_1)/2]
        
                x_y_top_left      = [x_n,           0.5]
                x_y_top_center    = [x_n_plus_half, 0.5]
                x_y_top_right     = [x_n_plus_1,    0.5]
    
    
                for j in range(0, N_q2 + 2*N_g):
    
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
    
            jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]


        # Top-left domain
        elif ((q2_midpoint > 0.1666) and (q1_midpoint > -0.5) and (q1_midpoint < 0.1666)):

            N        = N_q1 
            x_0      = -0.5#-radius/np.sqrt(2)
    
            # Initialize to zero
            x = 0*q1
            y = 0*q2
    
    
            # Loop over each zone in x
            for i in range(0, N_q1 + 2*N_g):
    
                index = i - N_g # Index of the vertical slice, left-most being 0
    
                # q1, q2 grid slices to be passed into quadratic for transformation
                q1_slice = q1[0, 0, i, N_g:-N_g]
                q2_slice = q2[0, 0, i, N_g:-N_g]
    
                # Compute the x, y points using which the transformation will be defined
                # x, y nodes remain the same for each point on a vertical slice
                x_n           = x_0 + (0.5-radius/np.sqrt(2))*index/N # Bottom-left
                y_n           = radius/np.sqrt(2)
                
    
                x_n_plus_1    = x_0 + (0.5-radius/np.sqrt(2))*(index+1)/N # Bottom-right
                y_n_plus_1    = radius/np.sqrt(2)
    
                x_n_plus_half = x_0 + (0.5-radius/np.sqrt(2))*(index+0.5)/N # Bottom-center
                y_n_plus_half = radius/np.sqrt(2)
    
    
                x_y_bottom_left   = [x_n,           y_n]
                x_y_bottom_center = [x_n_plus_half, y_n_plus_half]
    
                x_y_bottom_right  = [x_n_plus_1,    y_n_plus_1]
        
                x_y_left_center   = [x_n,        (0.5+y_n)/2]
                x_y_right_center  = [x_n_plus_1, (0.5+y_n_plus_1)/2]
        
                x_y_top_left      = [x_n,           0.5]
                x_y_top_center    = [x_n_plus_half, 0.5]
                x_y_top_right     = [x_n_plus_1,    0.5]
    
    
                for j in range(0, N_q2 + 2*N_g):
    
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
    
            jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]

        
        # Top-right domain
        elif ((q2_midpoint > 0.1666) and (q1_midpoint > 0.1666) and (q1_midpoint < 0.5)):

            N        = N_q1 
            x_0     = radius/np.sqrt(2)
    
            # Initialize to zero
            x = 0*q1
            y = 0*q2
    
    
            # Loop over each zone in x
            for i in range(0, N_q1 + 2*N_g):
    
                index = i - N_g # Index of the vertical slice, left-most being 0
    
                # q1, q2 grid slices to be passed into quadratic for transformation
                q1_slice = q1[0, 0, i, N_g:-N_g]
                q2_slice = q2[0, 0, i, N_g:-N_g]
    
                # Compute the x, y points using which the transformation will be defined
                # x, y nodes remain the same for each point on a vertical slice
                x_n           = x_0 + (0.5-x_0)*index/N # Bottom-left
                y_n           = radius/np.sqrt(2)
                
    
                x_n_plus_1    = x_0 + (0.5-x_0)*(index+1)/N # Bottom-right
                y_n_plus_1    = radius/np.sqrt(2)
    
                x_n_plus_half = x_0 + (0.5-x_0)*(index+0.5)/N # Bottom-center
                y_n_plus_half = radius/np.sqrt(2)
    
    
                x_y_bottom_left   = [x_n,           y_n]
                x_y_bottom_center = [x_n_plus_half, y_n_plus_half]
    
                x_y_bottom_right  = [x_n_plus_1,    y_n_plus_1]
        
                x_y_left_center   = [x_n,        (0.5+y_n)/2]
                x_y_right_center  = [x_n_plus_1, (0.5+y_n_plus_1)/2]
        
                x_y_top_left      = [x_n,           0.5]
                x_y_top_center    = [x_n_plus_half, 0.5]
                x_y_top_right     = [x_n_plus_1,    0.5]
    
    
                for j in range(0, N_q2 + 2*N_g):
    
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
    
            jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]


        # Center-Right domain
        elif ((q2_midpoint > -0.1666) and (q2_midpoint < 0.1666) and (q1_midpoint > 0.1666)):

            N        = N_q1 
            y_0      = -radius/np.sqrt(2)
    
            # Initialize to zero
            x = 0*q1
            y = 0*q2
    
    
            # Loop over each zone in y
            for j in range(0, N_q2 + 2*N_g):
                print (j)
                index = j - N_g # Index of the vertical slice, left-most being 0
    
                # q1, q2 grid slices to be passed into quadratic for transformation
                q1_slice = q1[0, 0, N_g:-N_g, j]
                q2_slice = q2[0, 0, N_g:-N_g, j]
    
                # Compute the x, y points using which the transformation will be defined
                # x, y nodes remain the same for each point on a vertical slice
                y_n           = y_0 + np.sqrt(2)*radius*index/N
                x_n           = np.sqrt(radius**2 - y_n**2) # Bottom-left
                
    
                y_n_plus_1    = y_0 + np.sqrt(2)*radius*(index+1)/N # top-left
                x_n_plus_1    = np.sqrt(radius**2 - y_n_plus_1**2)
    
                y_n_plus_half = y_0 + np.sqrt(2)*radius*(index+0.5)/N # Center-left
                x_n_plus_half = np.sqrt(radius**2 - y_n_plus_half**2)
    
    
                x_y_bottom_left   = [x_n,         y_n]
                x_y_bottom_center = [(0.5 + x_n)/2, y_n]
                x_y_bottom_right  = [0.5,           y_n]
        
                x_y_left_center   = [x_n_plus_half, y_n_plus_half]
                x_y_right_center  = [0.5,             y_n_plus_half]
        
                x_y_top_left      = [x_n_plus_1,         y_n_plus_1]
                x_y_top_center    = [(0.5 + x_n_plus_1)/2, y_n_plus_1]
                x_y_top_right     = [0.5,                  y_n_plus_1]
    
    
                for i in range(0, N_q1 + 2*N_g):
    
                    # Get the transformation (x_i, y_i) for each point (q1_i, q2_i) 
                    q1_i = q1[0, 0, i, j]
                    q2_i = q2[0, 0, i, j]
        
                    x_i, y_i, jacobian_i = quadratic_test(q1_i, q2_i, q1_slice, q2_slice,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom + index*domain.dq2,
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
    
            jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]



        # Center-Left domain
        elif ((q2_midpoint > -0.1666) and (q2_midpoint < 0.1666) and (q1_midpoint < -0.1666)):

            N        = N_q1 
            #x_0      = -radius/np.sqrt(2)
            y_0      = -radius/np.sqrt(2)
    
            # Initialize to zero
            x = 0*q1
            y = 0*q2
    
    
            # Loop over each zone in y
            for j in range(0, N_q2 + 2*N_g):
                print (j)
                index = j - N_g # Index of the vertical slice, left-most being 0
    
                # q1, q2 grid slices to be passed into quadratic for transformation
                q1_slice = q1[0, 0, N_g:-N_g, j]
                q2_slice = q2[0, 0, N_g:-N_g, j]
    
                # Compute the x, y points using which the transformation will be defined
                # x, y nodes remain the same for each point on a vertical slice
                y_n           = y_0 + np.sqrt(2)*radius*index/N
                x_n           = -np.sqrt(radius**2 - y_n**2) # Bottom-right
                
    
                y_n_plus_1    = y_0 + np.sqrt(2)*radius*(index+1)/N # top-right
                x_n_plus_1    = -np.sqrt(radius**2 - y_n_plus_1**2)
    
                y_n_plus_half = y_0 + np.sqrt(2)*radius*(index+0.5)/N # Center-right
                x_n_plus_half = -np.sqrt(radius**2 - y_n_plus_half**2)
    
    
                x_y_bottom_left   = [-0.5,           y_n]
                x_y_bottom_center = [(-0.5 + x_n)/2, y_n]
                x_y_bottom_right  = [x_n,          y_n]
        
                x_y_left_center   = [-0.5,            y_n_plus_half]
                x_y_right_center  = [x_n_plus_half, y_n_plus_half]
        
                x_y_top_left      = [-0.5,                  y_n_plus_1]
                x_y_top_center    = [(-0.5 + x_n_plus_1)/2, y_n_plus_1]
                x_y_top_right     = [x_n_plus_1,          y_n_plus_1]
    
    
                for i in range(0, N_q1 + 2*N_g):
    
                    # Get the transformation (x_i, y_i) for each point (q1_i, q2_i) 
                    q1_i = q1[0, 0, i, j]
                    q2_i = q2[0, 0, i, j]
        
                    x_i, y_i, jacobian_i = quadratic_test(q1_i, q2_i, q1_slice, q2_slice,
                                       x_y_bottom_left,   x_y_bottom_right,
                                       x_y_top_right,     x_y_top_left,
                                       x_y_bottom_center, x_y_right_center,
                                       x_y_top_center,    x_y_left_center,
                                       q1_start_local_left,
                                       q2_start_local_bottom + index*domain.dq2,
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
    
            jacobian = [[dx_dq1, dx_dq2], [dy_dq1, dy_dq2]]

        # Center domain
        elif ((q2_midpoint > -0.1666) and (q2_midpoint < 0.1666) and (q1_midpoint > -0.1666) and (q1_midpoint < 0.1666)):

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


