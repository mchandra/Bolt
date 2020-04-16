import os
import arrayfire as af
import numpy as np
import math
import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
MPI.WTIME_IS_GLOBAL=True

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear.nonlinear_solver \
    import nonlinear_solver
#from bolt.lib.nonlinear.fields.fields.fields \
#    import fields_solver.compute_electrostatic_fields
from bolt.lib.utils.restart_latest import latest_output, format_time


import bolt.src.electronic_boltzmann.advection_terms \
    as advection_terms
import bolt.src.electronic_boltzmann.collision_operator \
    as collision_operator
import bolt.src.electronic_boltzmann.moment_defs \
    as moment_defs


# Common params
import params

# Horizontal
import domain_1
import boundary_conditions_1
import params_1
import initialize_1

# Vertical
import domain_2
import boundary_conditions_2
import params_2
import initialize_2

# Defining the physical system to be solved:
system_1 = physical_system(domain_1,
                         boundary_conditions_1,
                         params_1,
                         initialize_1,
                         advection_terms,
                         collision_operator.RTA,
                         moment_defs
                        )

system_2 = physical_system(domain_2,
                         boundary_conditions_2,
                         params_2,
                         initialize_2,
                         advection_terms,
                         collision_operator.RTA,
                         moment_defs
                        )
# Time parameters:
dt_1      = params_1.dt
t_final_1 = params_1.t_final
params_1.current_time = time_elapsed_1 = 0.0
params_1.time_step    = time_step_1 = 0

dt_2      = params_2.dt
t_final_2 = params_2.t_final
params_2.current_time = time_elapsed_2 = 0.0
params_2.time_step    = time_step_2 = 0


dump_counter = 0
dump_time_array = []

N_g_1  = domain_1.N_ghost
N_g_2  = domain_2.N_ghost

# Declaring a nonlinear system object which will evolve the defined physical system:
nls_1 = nonlinear_solver(system_1)
params_1.rank = nls_1._comm.rank

nls_2 = nonlinear_solver(system_2)
params_2.rank = nls_2._comm.rank

params.f_1 = nls_1.f
params.f_2 = nls_2.f

if (params_1.restart):
    nls_1.load_distribution_function(params_1_.restart_file)
if (params_2.restart):
    nls_2.load_distribution_function(params_2_.restart_file)

# Checking that the file writing intervals are greater than dt:
assert(params_1.dt_dump_f > dt_1)
assert(params_1.dt_dump_moments > dt_1)
assert(params_1.dt_dump_fields > dt_1)
assert(params_2.dt_dump_f > dt_2)
assert(params_2.dt_dump_moments > dt_2)
assert(params_2.dt_dump_fields > dt_2)


density_1 = nls_1.compute_moments('density')
print("rank = ", params_1.rank, "\n",
      "     <mu>    = ", af.mean(params_1.mu[0, 0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
      "     max(mu) = ", af.max(params_1.mu[0, 0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
      "     <n>     = ", af.mean(density_1[0, 0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
      "     max(n)  = ", af.max(density_1[0, 0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n"
     )

density_2 = nls_2.compute_moments('density')
print("rank = ", params_2.rank, "\n",
      "     <mu>    = ", af.mean(params_2.mu[0, 0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
      "     max(mu) = ", af.max(params_2.mu[0, 0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
      "     <n>     = ", af.mean(density_2[0, 0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
      "     max(n)  = ", af.max(density_2[0, 0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n"
     )

while(time_elapsed_1 < t_final_1):

    # Refine to machine error
    if (time_step_1==0):
        params_1.collision_nonlinear_iters = 10
    else:
        params_1.collision_nonlinear_iters = params_1.collision_operator_nonlinear_iters
    
    if (time_step_2==0):
        params_2.collision_nonlinear_iters = 10
    else:
        params_2.collision_nonlinear_iters = params_2.collision_operator_nonlinear_iters

    # Store distribution functions in common params for coupling devices
    params.f_1 = nls_1.f
    params.f_2 = nls_2.f


    if(params_1.dt_dump_moments != 0):
        # We step by delta_dt to get the values at dt_dump
        delta_dt =   (1 - math.modf(time_elapsed_1/params_1.dt_dump_moments)[0]) \
                   * params_1.dt_dump_moments

        if((delta_dt-dt_1)<1e-5):
            nls_1.strang_timestep(delta_dt)
            nls_2.strang_timestep(delta_dt)
            time_elapsed_1 += delta_dt
            time_elapsed_2 += delta_dt
            formatted_time = format_time(time_elapsed_1)            
            nls_1.dump_moments('dump_moments_1/t=' + formatted_time)
            nls_2.dump_moments('dump_moments_2/t=' + formatted_time)
            nls_1.dump_aux_arrays([params_1.mu,
                             params_1.mu_ee,
                             params_1.T_ee,
                             params_1.vel_drift_x, params_1.vel_drift_y,
                             params_1.j_x, params_1.j_y],
                             'lagrange_multipliers',
                             'dump_lagrange_multipliers_1/t=' + formatted_time
                            )
            nls_2.dump_aux_arrays([params_2.mu,
                             params_2.mu_ee,
                             params_2.T_ee,
                             params_2.vel_drift_x, params_2.vel_drift_y,
                             params_2.j_x, params_2.j_y],
                             'lagrange_multipliers',
                             'dump_lagrange_multipliers_2/t=' + formatted_time
                            )
            dump_time_array.append(time_elapsed_1)
            if (params_1.rank==0):
                np.savetxt("dump_time_array.txt", dump_time_array)

    if(math.modf(time_elapsed_1/params_1.dt_dump_f)[0] < 1e-12):
        formatted_time = format_time(time_elapsed_1)
        nls_1.dump_distribution_function('dump_f_1/t=' + formatted_time)        
        nls_2.dump_distribution_function('dump_f_2/t=' + formatted_time)        

    PETSc.Sys.Print("Time step =", time_step_1, ", Time =", time_elapsed_1)

    nls_1.strang_timestep(dt_1)
    time_elapsed_1        = time_elapsed_1 + dt_1
    time_step_1           = time_step_1 + 1
    params_1.time_step    = time_step_1
    params_1.current_time = time_elapsed_1
    
    nls_2.strang_timestep(dt_2)
    time_elapsed_2        = time_elapsed_2 + dt_2
    time_step_2           = time_step_2 + 1
    params_2.time_step    = time_step_2
    params_2.current_time = time_elapsed_2

    density_1 = nls_1.compute_moments('density')
    PETSc.Sys.Print("------Domain-1------\n")
    print("rank = ", params_1.rank, "\n",
          "     <mu>    = ", af.mean(params_1.mu[0, 0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
          "     max(mu) = ", af.max(params_1.mu[0, 0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
          "     <n>     = ", af.mean(density_1[0, 0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n",
          "     max(n)  = ", af.max(density_1[0, 0, N_g_1:-N_g_1, N_g_1:-N_g_1]), "\n"
         )
    PETSc.Sys.Print("--------------------\n")
    
    density_2 = nls_2.compute_moments('density')
    PETSc.Sys.Print("------Domain-2------\n")
    print("rank = ", params_2.rank, "\n",
          "     <mu>    = ", af.mean(params_2.mu[0, 0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
          "     max(mu) = ", af.max(params_2.mu[0, 0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
          "     <n>     = ", af.mean(density_2[0, 0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n",
          "     max(n)  = ", af.max(density_2[0, 0, N_g_2:-N_g_2, N_g_2:-N_g_2]), "\n"
         )
    PETSc.Sys.Print("--------------------\n")

nls_1.dump_distribution_function('dump_f_1/t_laststep')
nls_2.dump_distribution_function('dump_f_2/t_laststep')
