"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc

def initialize_f(q1, q2, p1, p2, p3, params):
   
    params.p_x, params.p_y = params.get_p_x_and_p_y(p1, p2)

    PETSc.Sys.Print("Initializing f")
    k = params.boltzmann_constant
    
    params.mu          = 0.*q1 + params.initial_mu
    params.T           = 0.*q1 + params.initial_temperature
    params.vel_drift_x = 0.*q1
    params.vel_drift_y = 0.*q1

    params.mu_ee       = params.mu.copy()
    params.T_ee        = params.T.copy()
    params.vel_drift_x = 0.*q1 + 0e-3
    params.vel_drift_y = 0.*q1 + 0e-3
    params.j_x         = 0.*q1
    params.j_y         = 0.*q1

    params.E_band   = params.band_energy(p1, p2)
    params.vel_band = params.band_velocity(p1, p2)

    # Evaluating velocity space resolution for each species:
    self.dp1 = []
    self.dp2 = []
    self.dp3 = []

    for i in range(N_species):
        self.dp1.append((self.p1_end[i] - self.p1_start[i]) / self.N_p1)
        self.dp2.append((self.p2_end[i] - self.p2_start[i]) / self.N_p2)
        self.dp3.append((self.p3_end[i] - self.p3_start[i]) / self.N_p3)

    theta = af.atan(params.p_y / params.p_x)
    p_f   = params.fermi_momentum_magnitude(theta)

    params.integral_measure = \
      (4./(2.*np.pi*params.h_bar)**2) * self.dp3 * self.dp2 * self.dp1 


    f = (1./(af.exp( (params.E_band - params.vel_drift_x*params.p_x
                                    - params.vel_drift_y*params.p_y
                                    - params.mu
                    )/(k*params.T) 
                  ) + 1.
           ))

    af.eval(f)
    return(f)


def initialize_E(q1, q2, params):
    
    E1 = 0.*q1
    E2 = 0.*q1
    E3 = 0.*q1

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = 0.*q1
    B2 = 0.*q1
    B3 = 0.*q1

    return(B1, B2, B3)
