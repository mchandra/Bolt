"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

# Problem Parameters:
# n0_num  = 1
# B10_num = 1
# c_num   = 5
# mu_num  = 1
# e_num   = 1
# mi_num  = 1
# me_num  = 1 / 10
# L1_num  = 2 / 2.5 * pi
# k1_num  = 2 * pi / L1_num

# ('Eigenvalue   = ', -1.778678888935013e-15 - 3.9148649135154923*I)
# (delta_u2_e, ' = ', 5.263217094683519e-16 - 0.5704289802593318*I)
# (delta_u3_e, ' = ', 0.5704289802593329)
# (delta_u2_i, ' = ', 6.019317878676518e-17 - 0.07062528600894578*I)
# (delta_u3_i, ' = ', 0.0706252860089461 + 1.414129446374935e-17*I)
# (delta_B2, ' = ', -8.230407390233223e-17 + 0.22166393214105898*I)
# (delta_B3, ' = ', -0.2216639321410591 - 9.329104414028616e-17*I)
# (delta_E2, ' = ', -0.34711374021236485 + 1.2665609544910668e-16*I)
# (delta_E3, ' = ', 1.8537358402307726e-16 - 0.3471137402123639*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * -5.745404152435185e-15 * af.cos(params.k_q1 * q1) \
                - params.amplitude * - 0.07062528600894578 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * -5.88418203051333e-15 * af.cos(params.k_q1 * q1) \
                - params.amplitude * - 0.5704289802593318 * af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 0.0706252860089461 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 2.220446049250313e-16  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 0.5704289802593329 * af.cos(params.k_q1 * q1) \
                - params.amplitude * -2.7755575615628914e-16 * af.sin(params.k_q1 * q1)

    n = n_b + 0 * q1**0

    f_e = n * (m[0, 0] / (2 * np.pi * k * T_b)) \
            * af.exp(-m[0, 0] * (v2[:, 0] - v2_bulk_e)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v3[:, 0] - v3_bulk_e)**2 / (2 * k * T_b))

    f_i = n * (m[0, 1] / (2 * np.pi * k * T_b)) \
            * af.exp(-m[0, 1] * (v2[:, 1] - v2_bulk_i)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v3[:, 1] - v3_bulk_i)**2 / (2 * k * T_b))

    f = af.join(1, f_e, f_i)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    
    E2 =   params.amplitude * -0.34711374021236485 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -1.1102230246251565e-16 * af.sin(params.k_q1 * q1)
    
    E3 =   params.amplitude * -5.578870698741412e-15  * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.3471137402123639 * af.sin(params.k_q1 * q1)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    dt = params.dt
    B1 = params.B0 * q1**0

    omega = -1.778678888935013e-15 - 3.9148649135154923 * 1j

    B2 = (params.amplitude * (-8.230407390233223e-17 + 0.22166393214105898*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B3 = (params.amplitude * (-0.2216639321410591 - 9.329104414028616e-17*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B2 = af.moddims(af.to_array(B2), 1, 1, q1.shape[2], q1.shape[3])
    B3 = af.moddims(af.to_array(B3), 1, 1, q1.shape[2], q1.shape[3])

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
