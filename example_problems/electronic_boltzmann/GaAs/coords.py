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
    jacobian = None


    if (q1_start_local_left != None and q2_start_local_bottom != None):

        if (q1_midpoint < -4.62): # Domain 1 and 7
            y = 0.816*q2
    
        elif ((q1_midpoint > -4.62) and (q1_midpoint < 0)): # Domain 2 and 8
            y = (q2 *(1 + 0.04*(q1)))
    
        elif ((q1_midpoint > 29.46) and (q1_midpoint < 32.98) and (q2_midpoint < 12)): # Domain 5
            y = ((q2-12) *(1 - 0.1193*(q1-29.46))) + 12
    
        elif ((q1_midpoint > 32.98) and (q2_midpoint < 12)): # Domain 6
            y = 0.58*(q2-12) + 12
    
        elif ((q1_midpoint > 26.3) and (q1_midpoint < 29.46) and (q2_midpoint > 12)): # Domain 10
            y = ((q2-12) *(1 - 2*0.0451*(q1-26.3))) + 12
    
        elif ((q1_midpoint > 29.46) and (q1_midpoint < 32.98) and (q2_midpoint > 12)): # Domain 11
            y = ((q2-12) *(1 - 2*0.0451*(q1-26.3))) + 12
    
        elif ((q1_midpoint > 32.98) and (q2_midpoint > 12)):  # Domain 12
            y = 0.40*(q2-12) + 12

        # Numerically calculate Jacobian
        jacobian = None

        if (return_jacobian):
            return (x, y, jacobian)
        else: 
            return(x, y)

    else:
        print("Error in get_cartesian_coords(): q1_start_local_left or q2_start_local_bottom not provided")

