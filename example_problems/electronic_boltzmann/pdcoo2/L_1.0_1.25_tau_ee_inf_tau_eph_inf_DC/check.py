import numpy as np
import params
import arrayfire as af


p2_start = -np.pi
p2_end   = np.pi
N_p2     = 1024
theta = \
    p2_start + (0.5 + np.arange(N_p2))*(p2_end - p2_start)/N_p2

theta = af.from_ndarray(theta)

print (params.polygon (6, theta))
