# This file contains the functions that are used to take care of the interzonal
# communications when the code is run in parallel across multiple nodes. 
# Additionally, these functions are also responsible for applying boundary conditions.

import numpy as np
import arrayfire as af

def communicate_distribution_function(da, args, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = args.config.N_ghost

  # Applying the boundary conditions:
  #args.f         = apply_BC_distribution_function(da, args)

  # Storing values of af.Array in PETSc.Vec:
  local_value[:] = np.array(args.f)
  
  # Global value is non-inclusive of the ghost-zones:
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost,\
                                   :
                                  ]

  # The following function takes care of the boundary conditions, 
  # and interzonal communications:
  da.globalToLocal(glob, local)

  # Converting back from PETSc.Vec to af.Array:
  f_updated = af.to_array(local_value[:])

  #args.f = f_updated

  # Applying the boundary conditions:
  #args.f         = apply_BC_distribution_function(da, args)

  config  = args.config
  N_ghost = config.N_ghost

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  (j_top, i_right) = (j_bottom + N_y_local - 1, i_left + N_x_local - 1)

#  if (i_left == 0):
#
#    # Reflecting boundaries on the left
#    for i in range(N_ghost):
#         # N_ghost = 3
#         # | o | o | o || o | o | o |...
#         #   0   1   2    3   4   5
#
#         f_updated[:, N_ghost-i-1] = f_updated[:, i+N_ghost]
#         # i = 0            2   =   3
#         # i = 1            1   =   4
#         # i = 2            0   =   5
#
#    import non_linear_solver.convert
#    f_updated = non_linear_solver.convert.to_velocitiesExpanded\
#                    (da, args.config, f_updated)  # (Ny * N_ghost, Nvy, Nvx)
#
#    f_flip_p_x = af.flip(f_updated, 2)
#
#    f_flip_p_x = non_linear_solver.convert.to_positionsExpanded\
#                    (da, args.config, f_flip_p_x)  # (Ny,  N_ghost, Nvy, Nvx)
#
#    f_updated = non_linear_solver.convert.to_positionsExpanded\
#                    (da, args.config, f_updated)  # (Ny,  N_ghost, Nvy, Nvx)
#
#    f_updated[:, :N_ghost] = f_flip_p_x[:, :N_ghost]
#
#    # Inflow between y = [y_center(j_inflow_start), y_center(j_inflow_end)]
#    size_of_inflow     = 1.
#    offset_from_center = 0.
#    length_y           = config.y_end - config.y_start
#    N_inflow_zones     = (int)(size_of_inflow/length_y*config.N_y)
#    N_offset           = (int)(abs(offset_from_center)/length_y*config.N_y)
#    j_inflow_start     =   N_ghost + config.N_y/2 - N_inflow_zones/2 \
#                         + np.sign(offset_from_center)*N_offset
#    j_inflow_end       = N_ghost + config.N_y/2 + N_inflow_zones/2 \
#                         + np.sign(offset_from_center)*N_offset
#
#    p_x = config.h_cross * args.vel_x # (Nx*Ny, Nvy, Nvx, Nvz)
#    p_y = config.h_cross * args.vel_y # (Nx*Ny, Nvy, Nvx, Nvz)
#
#    E_upper, E_lower = config.band_energy(p_x, p_y)
#    k                = config.boltzmann_constant
#    mu_0             = config.chemical_potential_background
#    T_0              = config.temperature_background
#
#    p_x     = non_linear_solver.convert.to_positionsExpanded \
#                (da, config, p_x)          # (Ny, Nx, Nvy*Nvx)
#    p_y     = non_linear_solver.convert.to_positionsExpanded \
#                (da, config, p_y)          # (Ny, Nx, Nvy*Nvx)
#    E_upper = non_linear_solver.convert.to_positionsExpanded \
#                (da, config, E_upper) # (Ny, Nx, Nvy*Nvx)
#
#    for i in range(N_ghost):
#         # N_ghost = 3
#         # | o | o | o || o | o | o |...
#         #   0   1   2    3   4   5
#
#         f_inflow = \
#            1./(af.exp( (E_upper - 1e-3*p_x - 0e-1*p_y - mu_0)/(k*T_0) ) + 1.)
#
#         f_updated[j_inflow_start:j_inflow_end, i] = \
#            f_inflow[j_inflow_start:j_inflow_end, i]
#
#         
#  if (i_right == config.N_x - 1):
#
#    # Reflecting boundaries on the right
#    for i in range(N_ghost):
#         # N_ghost = 3, N_x_local = 32
#         # ...| o | o | o || o | o | o |
#         #     32  33  34   35  36  37
#
#         f_updated[:, i + N_ghost + N_x_local] = \
#                 f_updated[:, N_ghost + N_x_local - i - 1]
#         # i = 0                      35    =   34
#         # i = 1                      36    =   33
#         # i = 2                      37    =   32
#
##    f_updated = non_linear_solver.convert.to_velocitiesExpanded\
##                    (da, args.config, f_updated)  # (Ny * N_ghost, Nvy, Nvx)
##
##    f_flip_p_x = af.flip(f_updated, 2)
##
##    f_flip_p_x = non_linear_solver.convert.to_positionsExpanded\
##                    (da, args.config, f_flip_p_x)  # (Ny,  N_ghost, Nvy, Nvx)
##
##    f_updated = non_linear_solver.convert.to_positionsExpanded\
##                    (da, args.config, f_updated)  # (Ny,  N_ghost, Nvy, Nvx)
##
##    f_updated[:, N_x_local + N_ghost:] = \
##            f_flip_p_x[:, N_x_local + N_ghost:]
##
##    # Outflow between y = [y_center(j_outflow_start), y_center(j_outflow_end)]
##    size_of_outflow = 0.2
##    length_y        = config.y_end - config.y_start
##    N_outflow_zones = (int)(size_of_outflow/length_y*config.N_y)
##    j_outflow_start = N_ghost + config.N_y/2 - N_outflow_zones/2
##    j_outflow_end   = N_ghost + config.N_y/2 + N_outflow_zones/2
##
##    for i in range(N_ghost):
##         # N_ghost = 3, N_x_local = 32
##         # ...| o | o | o || o | o | o |
##         #     32  33  34   35  36  37
##
##         f_updated[j_outflow_start:j_outflow_end, i + N_ghost + N_x_local] = \
##            f_updated[j_outflow_start:j_outflow_end, N_ghost + N_x_local - 1]
##
##         f_right = \
##            1./(af.exp( (E_upper - 0e-5*p_x - 0e-1*p_y - mu_0)/(k*T_0) ) + 1.)
##
##         f_updated[:, i+N_ghost+N_x_local] = \
##                 f_right[:, i+N_ghost+N_x_local]
#
#  if (j_bottom == 0):
#    for j in range(N_ghost):
#        # N_ghost = 3
#        # | o | o | o || o | o | o |...
#        #   0   1   2    3   4   5
#
#        f_updated[N_ghost-j-1, :] = f_updated[j+N_ghost, :]
#        # j = 0            2   =   3
#        # j = 1            1   =   4
#        # j = 2            0   =   5
#
#    f_updated = non_linear_solver.convert.to_velocitiesExpanded\
#                    (da, args.config, f_updated)  # (Ny * N_ghost, Nvy, Nvx)
#
#    f_flip_p_y = af.flip(f_updated, 1)
#
#    f_flip_p_y = non_linear_solver.convert.to_positionsExpanded\
#                    (da, args.config, f_flip_p_y)  # (Ny,  N_ghost, Nvy, Nvx)
#
#    f_updated = non_linear_solver.convert.to_positionsExpanded\
#                    (da, args.config, f_updated)  # (Ny,  N_ghost, Nvy, Nvx)
#
#    f_updated[:N_ghost, :] = f_flip_p_y[:N_ghost, :]
#
#  if (j_top == config.N_y - 1):
#    for j in range(N_ghost):
#        # N_ghost = 3, N_y_local = 32
#        # ...| o | o | o || o | o | o |
#        #     32  33  34   35  36  37
#
#        f_updated[j + N_ghost + N_y_local, :] = \
#                f_updated[N_ghost + N_y_local - j - 1, :]
#        # j = 0                      35    =   34
#        # j = 1                      36    =   33
#        # j = 2                      37    =   32
#
#    f_updated = non_linear_solver.convert.to_velocitiesExpanded\
#                    (da, args.config, f_updated)  # (Ny * N_ghost, Nvy, Nvx)
#
#    f_flip_p_y = af.flip(f_updated, 1)
#
#    f_flip_p_y = non_linear_solver.convert.to_positionsExpanded\
#                    (da, args.config, f_flip_p_y)  # (Ny,  N_ghost, Nvy, Nvx)
#
#    f_updated = non_linear_solver.convert.to_positionsExpanded\
#                    (da, args.config, f_updated)  # (Ny,  N_ghost, Nvy, Nvx)
#
#    f_updated[N_y_local + N_ghost:, :] = \
#            f_flip_p_y[N_y_local + N_ghost:, :]

  af.eval(f_updated)
  return(f_updated)
  #af.eval(args.f)
  #return(args.f)

def apply_BC_distribution_function(da, args):

  config  = args.config
  N_ghost = config.N_ghost

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  (j_top, i_right) = (j_bottom + 1, i_left + 1)

#  if(args.config.bc_in_x == 'dirichlet'):
#
#    if(i_left == 0):
#      args.f[:, :N_ghost] = args.f_left
#
#    if(i_right == config.N_x - 1):
#      args.f[:, -N_ghost:] = args.f_right
#
#  if(args.config.bc_in_y == 'dirichlet'):
#
#    if(j_bottom == 0):
#      args.f[:N_ghost, :] = args.f_bot
#
#    if(j_top == config.N_y - 1):
#      args.f[-N_ghost:, :] = args.f_top


#  if (i_left == 0):
#
#    for i in range(N_ghost):
#         # N_ghost = 3
#         # | o | o | o || o | o | o |...
#         #   0   1   2    3   4   5
#
#        args.f[:, N_ghost-i-1] = args.f[:, i+N_ghost]
#         # i = 0            2   =   3
#         # i = 1            1   =   4
#         # i = 2            0   =   5
#
#    f_copy = args.f[:, :N_ghost, :, :] # (Ny, N_ghost, Nvy*Nvx)
#    f_copy = af.moddims(f_copy, (N_y_local + 2*N_ghost)*N_ghost, \
#                                config.N_vel_y, \
#                                config.N_vel_x, \
#                                config.N_vel_z \
#                       ) # (Ny * N_ghost, Nvy, Nvx)
#
#    f_px_negative = f_copy[:, :, :config.N_vel_x/2]
#    f_px_positive = f_copy[:, :, config.N_vel_x/2:]
#
#    f_copy[:, :, :config.N_vel_x/2] = f_px_positive
#    f_copy[:, :, config.N_vel_x/2:] = f_px_negative
#
#    f_copy = af.moddims(f_copy, N_y_local + 2*N_ghost, N_ghost, \
#                                config.N_vel_y * \
#                                config.N_vel_x  * \
#                                config.N_vel_z \
#                       ) # (Ny,  N_ghost, Nvy, Nvx)
#
#    for i in range(N_ghost):
#
#        args.f[:, i] = f_copy[:, i]
#
#  if (i_right == config.N_x - 1):
#
#    for i in range(N_ghost):
#         # N_ghost = 3, N_x_local = 32
#         # ...| o | o | o || o | o | o |
#         #     32  33  34   35  36  37
#
#         args.f[:, i + N_ghost + N_x_local] = args.f[:, N_ghost + N_x_local - i - 1]
#         # i = 0                      35    =   34
#         # i = 1                      36    =   33
#         # i = 2                      37    =   32
#
#    f_copy = args.f[:, N_x_local + N_ghost:, :, :] # (Ny, N_ghost, Nvy*Nvx)
#    f_copy = af.moddims(f_copy, (N_y_local + 2*N_ghost)*N_ghost, \
#                                config.N_vel_y, \
#                                config.N_vel_x, \
#                                config.N_vel_z \
#                       ) # (Ny * N_ghost, Nvy, Nvx)
#
#    f_px_negative = f_copy[:, :, :config.N_vel_x/2]
#    f_px_positive = f_copy[:, :, config.N_vel_x/2:]
#
#    f_copy[:, :, :config.N_vel_x/2] = f_px_positive
#    f_copy[:, :, config.N_vel_x/2:] = f_px_negative
#
#    f_copy = af.moddims(f_copy, N_y_local + 2*N_ghost, N_ghost, \
#                                config.N_vel_y * \
#                                config.N_vel_x  * \
#                                config.N_vel_z \
#                       ) # (Ny,  N_ghost, Nvy, Nvx)
#
#    for i in range(N_ghost):
#
#        args.f[:, i + N_x_local] = f_copy[:, i]

#  if (j_bottom == 0):
#    for j in range(N_ghost):
#        # N_ghost = 3
#        # | o | o | o || o | o | o |...
#        #   0   1   2    3   4   5
#
#        args.f[N_ghost-j-1, :] = args.f[j+N_ghost, :]
#        # j = 0            2   =   3
#        # j = 1            1   =   4
#        # j = 2            0   =   5
#
# if (j_top == config.N_y - 1):
#    for j in range(N_ghost):
#        # N_ghost = 3, N_y_local = 32
#        # ...| o | o | o || o | o | o |
#        #     32  33  34   35  36  37
#
#        args.f[j + N_ghost + N_y_local, :] = args.f[N_ghost + N_y_local - j - 1, :]
#        # j = 0                      35    =   34
#        # j = 1                      36    =   33
#        # j = 2                      37    =   32

  

  af.eval(args.f)
  return(args.f)

def communicate_fields(da, args, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = args.config.N_ghost

  # Assigning the values of the af.Array fields quantities
  # to the PETSc.Vec:
  (local_value[:])[:, :, 0] = np.array(args.E_x)
  (local_value[:])[:, :, 1] = np.array(args.E_y)
  (local_value[:])[:, :, 2] = np.array(args.E_z)
  
  (local_value[:])[:, :, 3] = np.array(args.B_x)
  (local_value[:])[:, :, 4] = np.array(args.B_y)
  (local_value[:])[:, :, 5] = np.array(args.B_z)

  # Global value is non-inclusive of the ghost-zones:
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost,\
                                   :
                                  ]

  # Takes care of boundary conditions and interzonal communications:
  da.globalToLocal(glob, local)

  # Converting back to af.Array
  args.E_x = af.to_array((local_value[:])[:, :, 0])
  args.E_y = af.to_array((local_value[:])[:, :, 1])
  args.E_z = af.to_array((local_value[:])[:, :, 2])

  args.B_x = af.to_array((local_value[:])[:, :, 3])
  args.B_y = af.to_array((local_value[:])[:, :, 4])
  args.B_z = af.to_array((local_value[:])[:, :, 5])

  return(args)
