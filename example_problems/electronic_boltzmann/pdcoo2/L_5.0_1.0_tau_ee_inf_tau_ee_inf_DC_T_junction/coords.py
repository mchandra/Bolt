import numpy as np
import arrayfire as af

def get_cartesian_coords(q1, q2):

    q1_midpoint = 0.5*(af.max(q1) + af.min(q1))
    q2_midpoint = 0.5*(af.max(q2) + af.min(q2))

    x = q1
    y = q2

#    if (q1_midpoint > 4.25):
#        y = 3*q2-1

    return(x, y)

