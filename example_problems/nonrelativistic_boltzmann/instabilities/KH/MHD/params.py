import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'user-defined'

# Can be defined as 'electrostatic' and 'fdtd'
fields_solver = 'fdtd'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

riemann_solver = 'upwind-flux'

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = -1

p_dim       = 3
num_devices = 4

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return(0 * q1**0 * p1**0)
