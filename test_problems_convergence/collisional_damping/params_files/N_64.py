import numpy as np

mode = '1D1V'

constants = dict(
                  mass_particle      = 1.0,
                  boltzmann_constant = 1.0,
                )

background_electrons = dict(
                            rho         = 1.0, 
                            temperature = 1.0, 
                            vel_bulk_x  = 0,
                            vel_bulk_y  = 0,
                           )

background_ions = dict(
                       rho         = 1.0, 
                       temperature = 1.0, 
                       vel_bulk_x  = 0,
                       vel_bulk_y  = 0,
                      )

perturbation = dict(
                    pert_real = 1e-2, 
                    pert_imag = 0,
                    k_x       = 2*np.pi,
                    k_y       = 2*np.pi 
                   ) 

configuration_space = dict(N_x            = 64,
                           N_ghost_x      = 3,
                           left_boundary  = 0,
                           right_boundary = 1.0,

                           N_y            = 32,
                           N_ghost_y      = 3,
                           bot_boundary   = 0,
                           top_boundary   = 1.0,
                          )

boundary_conditions = dict(in_x = 'periodic',
                           in_y = 'periodic',

                           left_temperature = 1.0,
                           left_rho         = 1.0,
                           left_vel_bulk_x  = 0,
                           left_vel_bulk_y  = 0,

                           right_temperature = 1.0,
                           right_rho         = 1.0,
                           right_vel_bulk_x  = 0,
                           right_vel_bulk_y  = 0, 

                           bot_temperature = 1.0,
                           bot_rho         = 1.0,
                           bot_vel_bulk_x  = 0,
                           bot_vel_bulk_y  = 0, 

                           top_temperature = 1.0,
                           top_rho         = 1.0,
                           top_vel_bulk_x  = 0,
                           top_vel_bulk_y  = 0
                          )

velocity_space = dict(N_vel_x   = 401,
                      vel_x_max = 20.0, 

                      N_vel_y   = 101, 
                      vel_y_max = 10.0
                     )

time = dict(
            final_time   = 1.0,
            dt           = 0.01 * (32/configuration_space['N_x'])
           )

EM_fields = dict(
                 charge_particle = 0
                )

collisions = dict(
                  collision_operator = 'BGK',
                  tau                =  0.01
                 )