import numpy as np
import arrayfire as af
import domain
import params

from bolt.lib.utils.coord_transformation import affine

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
    jacobian = None


    if (q1_start_local_left != None and q2_start_local_bottom != None):

        # Patch-0
        if ((q2_midpoint < -12.25) and (q1_midpoint > -59.85) and (q1_midpoint < -16.65)):
            X_Y_top_right    = [-16.65,   -12.25]
            X_Y_top_left     = [-59.85,   -12.25]
            X_Y_bottom_left  = [-59.85,   -15.65]
            X_Y_bottom_right = [-16.65,   -15.65]
    
            x_y_top_right    = [-16.65,   -12.25]  
            x_y_top_left     = [-59.85,   -12.25]
            x_y_bottom_left  = [-59.85,   -14.00]
            x_y_bottom_right = [-16.65,   -14.00]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )

        # Patch-1
        elif ((q2_midpoint < -12.25) and (q1_midpoint > -16.65) and (q1_midpoint < -14.6)):
            X_Y_top_right    = [-14.6,    -12.25]
            X_Y_top_left     = [-16.65,   -12.25]
            X_Y_bottom_left  = [-16.65,   -15.65]
            X_Y_bottom_right = [-14.6,    -15.65]
    
            x_y_top_right    = [-14.6,    -12.25]  
            x_y_top_left     = [-16.65,   -12.25]
            x_y_bottom_left  = [-16.65,   -14.00]
            x_y_bottom_right = [-14.6,    -15.65]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )
        

        # Patch-2
        elif ((q2_midpoint < -12.25) and (q1_midpoint > -14.6) and (q1_midpoint < -1.25)):
            X_Y_top_right    = [-1.25,   -12.25]
            X_Y_top_left     = [-14.6,   -12.25]
            X_Y_bottom_left  = [-14.6,   -15.65]
            X_Y_bottom_right = [-1.25,   -15.65]
    
            x_y_top_right    = [-1.25,   -12.25]  
            x_y_top_left     = [-14.6,   -12.25]
            x_y_bottom_left  = [-14.6,   -15.65]
            x_y_bottom_right = [-4.3,   -15.65]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )

        # Patch-3
        elif ((q2_midpoint < -12.25) and (q1_midpoint > -1.25) and (q1_midpoint < 25.47)):
            X_Y_top_right    = [25.47,   -12.25]
            X_Y_top_left     = [-1.25,     -12.25]
            X_Y_bottom_left  = [-1.25,     -15.65]
            X_Y_bottom_right = [25.47,   -15.65]
    
            x_y_top_right    = [25.47,   -12.25]  
            x_y_top_left     = [-1.25,     -12.25]
            x_y_bottom_left  = [-4.3,     -15.65]
            x_y_bottom_right = [25.47,   -15.65]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )
        
        # Patch-4
        elif ((q2_midpoint > -12.25) and (q2_midpoint < -9.85) and (q1_midpoint > -59.85) and (q1_midpoint < -16.65)):
            X_Y_top_right    = [-16.65,   -9.85]
            X_Y_top_left     = [-59.85,   -9.85]
            X_Y_bottom_left  = [-59.85,   -12.25]
            X_Y_bottom_right = [-16.65,   -12.25]
    
            x_y_top_right    = [-16.65,   -9.85]  
            x_y_top_left     = [-59.85,   -9.85]
            x_y_bottom_left  = [-59.85,   -12.25]
            x_y_bottom_right = [-16.65,   -12.25]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )
        
        # Patch-5
        elif ((q2_midpoint > -12.25) and (q2_midpoint < -9.85) and (q1_midpoint > -16.65) and (q1_midpoint < -14.6)):
            X_Y_top_right    = [-14.6,    -9.85]
            X_Y_top_left     = [-16.65,   -9.85]
            X_Y_bottom_left  = [-16.65,   -12.25]
            X_Y_bottom_right = [-14.6,    -12.25]
    
            x_y_top_right    = [-14.6,    -9.85]  
            x_y_top_left     = [-16.65,   -9.85]
            x_y_bottom_left  = [-16.65,   -12.25]
            x_y_bottom_right = [-14.6,    -12.25]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )
        

        # Patch-6
        elif ((q2_midpoint > -12.25) and (q2_midpoint < -9.85) and (q1_midpoint > -14.6) and (q1_midpoint < -1.25)):
            X_Y_top_right    = [-1.25,   -9.85]
            X_Y_top_left     = [-14.6,   -9.85]
            X_Y_bottom_left  = [-14.6,   -12.25]
            X_Y_bottom_right = [-1.25,   -12.25]
    
            x_y_top_right    = [0.9 ,   -9.85]  
            x_y_top_left     = [-14.6,   -9.85]
            x_y_bottom_left  = [-14.6,   -12.25]
            x_y_bottom_right = [-1.25,   -12.25]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )

        # Patch-7
        elif ((q2_midpoint > -12.25) and (q2_midpoint < -9.85) and (q1_midpoint > -1.25) and (q1_midpoint < 25.47)):
            X_Y_top_right    = [25.47,   -9.85]
            X_Y_top_left     = [-1.25,    -9.85]
            X_Y_bottom_left  = [-1.25,    -12.25]
            X_Y_bottom_right = [25.47,   -12.25]
    
            x_y_top_right    = [25.47,   -9.85]  
            x_y_top_left     = [0.9,     -9.85]
            x_y_bottom_left  = [-1.25,    -12.25]
            x_y_bottom_right = [25.47,   -12.25]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )

        # Patch-8
        if ((q2_midpoint > -9.85) and (q2_midpoint < 11.25) and (q1_midpoint > -59.85) and (q1_midpoint < -16.65)):
            X_Y_top_right    = [-16.65,   11.25]
            X_Y_top_left     = [-59.85,   11.25]
            X_Y_bottom_left  = [-59.85,   -9.85]
            X_Y_bottom_right = [-16.65,   -9.85]
    
            x_y_top_right    = [-16.65,   11.25]  
            x_y_top_left     = [-59.85,   11.25]
            x_y_bottom_left  = [-59.85,   -9.85]
            x_y_bottom_right = [-16.65,   -9.85]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )
        
        # Patch-9
        elif ((q2_midpoint > -9.85) and (q2_midpoint < 11.25) and (q1_midpoint > -16.65) and (q1_midpoint < -14.6)):
            X_Y_top_right    = [-14.6,    11.25]
            X_Y_top_left     = [-16.65,   11.25]
            X_Y_bottom_left  = [-16.65,   -9.85]
            X_Y_bottom_right = [-14.6,    -9.85]
    
            x_y_top_right    = [-14.6,    11.25]  
            x_y_top_left     = [-16.65,   11.25]
            x_y_bottom_left  = [-16.65,   -9.85]
            x_y_bottom_right = [-14.6,    -9.85]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )
        

        # Patch-10
        elif ((q2_midpoint > -9.85) and (q2_midpoint < 11.25) and (q1_midpoint > -14.6) and (q1_midpoint < -1.25)):
            X_Y_top_right    = [-1.25,   11.25]
            X_Y_top_left     = [-14.6,   11.25]
            X_Y_bottom_left  = [-14.6,   -9.85]
            X_Y_bottom_right = [-1.25,   -9.85]
    
            x_y_top_right    = [0.9,   11.25]  
            x_y_top_left     = [-14.6,   11.25]
            x_y_bottom_left  = [-14.6,   -9.85]
            x_y_bottom_right = [0.9,   -9.85]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )

        # Patch-11
        elif ((q2_midpoint > -9.85) and (q2_midpoint < 11.25) and (q1_midpoint > -1.25) and (q1_midpoint < 25.47)):
            X_Y_top_right    = [25.47,   11.25]
            X_Y_top_left     = [-1.25,    11.25]
            X_Y_bottom_left  = [-1.25,    -9.85]
            X_Y_bottom_right = [25.47,   -9.85]
    
            x_y_top_right    = [25.47,   11.25]  
            x_y_top_left     = [0.9,    11.25]
            x_y_bottom_left  = [0.9,    -9.85]
            x_y_bottom_right = [25.47,   -9.85]
            
            x, y =  affine(q1, q2,
                           x_y_bottom_left, x_y_bottom_right, 
                           x_y_top_right,   x_y_top_left,
                           X_Y_bottom_left, X_Y_bottom_right, 
                           X_Y_top_right,   X_Y_top_left,
                          )
        # Numerically calculate Jacobian
        jacobian = None

        if (return_jacobian):
            return (x, y, jacobian)
        else: 
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")


