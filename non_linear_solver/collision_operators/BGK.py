import numpy as np
import arrayfire as af
import non_linear_solver.compute_moments

def f_MB(da, args):
  # In order to compute the local Maxwell-Boltzmann distribution, the moments of
  # the distribution function need to be computed. For this purpose, all the functions
  # which are passed to the array need to be in velocitiesExpanded form

  config = args.config
  f      = args.f

  vel_x = args.vel_x
  vel_y = args.vel_y
  vel_z = args.vel_z
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  # NOTE: Here we are making the assumption that when mode == '2V'/'1V', N_vel_z = 1
  # If otherwise code will break here.
  if(config.mode == '3V'):
    n          = af.tile(non_linear_solver.compute_moments.calculate_density(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    T          = af.tile(non_linear_solver.compute_moments.calculate_temperature(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    vel_bulk_x = af.tile(non_linear_solver.compute_moments.calculate_vel_bulk_x(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    vel_bulk_y = af.tile(non_linear_solver.compute_moments.calculate_vel_bulk_y(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    vel_bulk_z = af.tile(non_linear_solver.compute_moments.calculate_vel_bulk_z(args),\
                         1, f.shape[1], f.shape[2], f.shape[3]
                        )
    
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T))**(3/2) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_z - vel_bulk_z)**2/(2*boltzmann_constant*T))

  elif(config.mode == '2V'):
    n          = af.tile(non_linear_solver.compute_moments.calculate_density(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    T          = af.tile(non_linear_solver.compute_moments.calculate_temperature(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    vel_bulk_x = af.tile(non_linear_solver.compute_moments.calculate_vel_bulk_x(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    vel_bulk_y = af.tile(non_linear_solver.compute_moments.calculate_vel_bulk_y(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T))

  else:
    n          = af.tile(non_linear_solver.compute_moments.calculate_density(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    T          = af.tile(non_linear_solver.compute_moments.calculate_temperature(args),\
                         1, f.shape[1], f.shape[2], 1
                        )
    vel_bulk_x = af.tile(non_linear_solver.compute_moments.calculate_vel_bulk_x(args),\
                         1, f.shape[1], f.shape[2], 1
                        )

    f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
             af.exp(-mass_particle*(vel_x-vel_bulk_x)**2/(2*boltzmann_constant*T))

  f_MB = f_MB/config.normalization

  af.eval(f_MB)
  return(f_MB)

def collision_step_BGK(da, args, dt):

  #tau = args.config.tau
  tau_defect = args.config.tau_defect
  tau_ee     = args.config.tau_ee

#  # Converting from positionsExpanded form to velocitiesExpanded form:
#  #args.f = non_linear_solver.convert.to_velocitiesExpanded(da, args.config, args.f)
#
#  # Performing the step of df/dt = C[f] = -(f - f_MB)/tau:
#  #f0             = f_MB(da, args)
#  #f0_defect      = f_defect_scattering(da, args)
  f0_ee          = f_ee_scattering(da, args)

  args.f = args.f \
                  - (dt)*(args.f - f0_ee    )/tau_ee
#
##  args.f = args.f - (dt)*(args.f - f0_defect)/tau_defect \
##                  - (dt)*(args.f - f0_ee    )/tau_ee
#
#  # Converting from velocitiesExpanded form to positionsExpanded form:
#  #args.f = non_linear_solver.convert.to_positionsExpanded(da, args.config, args.f)

  af.eval(args.f)
  return(args.f)

def compute_residual_ee(guess, da, args):
#    mu          = af.Array.as_type(af.to_array(guess[0]), af.Dtype.f64)
#    T           = af.Array.as_type(af.to_array(guess[1]), af.Dtype.f64)
#    vel_drift_x = af.Array.as_type(af.to_array(guess[2]), af.Dtype.f64)
#    vel_drift_y = af.Array.as_type(af.to_array(guess[3]), af.Dtype.f64)

    mu          = guess[0]
    T           = guess[1]
    vel_drift_x = guess[2]
    vel_drift_y = guess[3]

    config = args.config

    p_x = config.h_cross * args.vel_x # (Nx*Ny, Nvy, Nvx, Nvz)
    p_y = config.h_cross * args.vel_y # (Nx*Ny, Nvy, Nvx, Nvz)

    E_upper,   E_lower   = config.band_energy(p_x, p_y)
    vel_upper, vel_lower = config.band_velocity(p_x, p_y)
    k                    = config.boltzmann_constant

    E_upper       = non_linear_solver.convert.to_positionsExpanded \
                      (da, config, E_upper)      # (Ny, Nx, Nvy*Nvx)
    p_x           = non_linear_solver.convert.to_positionsExpanded \
                      (da, config, p_x)          # (Ny, Nx, Nvy*Nvx)
    p_y           = non_linear_solver.convert.to_positionsExpanded \
                      (da, config, p_y)          # (Ny, Nx, Nvy*Nvx)
    
    mu          = af.tile(mu,           1, 1, config.N_vel_x * config.N_vel_y)
    T           = af.tile(T,            1, 1, config.N_vel_x * config.N_vel_y)
    vel_drift_x = af.tile(vel_drift_x,  1, 1, config.N_vel_x * config.N_vel_y)
    vel_drift_y = af.tile(vel_drift_y,  1, 1, config.N_vel_x * config.N_vel_y)
    
    fermi_dirac = 1./(af.exp( (E_upper - vel_drift_x*p_x - vel_drift_y*p_y - mu)/(k*T) ) + 1.)

    zeroth_moment   =         (args.f - fermi_dirac)
    first_moment_vx =     p_x*(args.f - fermi_dirac)
    first_moment_vy =     p_y*(args.f - fermi_dirac)
    second_moment   = E_upper*(args.f - fermi_dirac)
    
    dp_x = config.h_cross * config.dv_x
    dp_y = config.h_cross * config.dv_y

    eqn_mass_conservation   = af.sum(zeroth_moment,   2) * dp_x * dp_y
    eqn_mom_x_conservation  = af.sum(first_moment_vx, 2) * dp_x * dp_y
    eqn_mom_y_conservation  = af.sum(first_moment_vy, 2) * dp_x * dp_y
    eqn_energy_conservation = af.sum(second_moment,   2) * dp_x * dp_y

#    return([np.array(eqn_mass_conservation), \
#            np.array(eqn_mom_x_conservation), \
#            np.array(eqn_mom_y_conservation), \
#            np.array(eqn_energy_conservation)])
    return([eqn_mass_conservation, \
            eqn_mom_x_conservation, \
            eqn_mom_y_conservation, \
            eqn_energy_conservation])

def compute_residual_defect(guess, da, args):
    mu = af.Array.as_type(af.to_array(guess[0]), af.Dtype.f64)
    T = af.Array.as_type(af.to_array(guess[1]), af.Dtype.f64)

    config = args.config

    #mu = guess[0] # (Ny, Nx)
    #T  = guess[1] # (Ny, Nx)

    p_x = config.h_cross * args.vel_x # (Nx*Ny, Nvy, Nvx, Nvz)
    p_y = config.h_cross * args.vel_y # (Nx*Ny, Nvy, Nvx, Nvz)

    E_upper, E_lower = config.band_energy(p_x, p_y)
    k                = config.boltzmann_constant

    E_upper = non_linear_solver.convert.to_positionsExpanded \
                (da, config, E_upper) # (Ny, Nx, Nvy*Nvx)
    
    mu = af.tile(mu, 1, 1, config.N_vel_x * config.N_vel_y)
    T  = af.tile(T,  1, 1, config.N_vel_x * config.N_vel_y)
    
    fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)

    zeroth_moment =          args.f - fermi_dirac
    second_moment = E_upper*(args.f - fermi_dirac)
    
    dp_x = config.h_cross * config.dv_x
    dp_y = config.h_cross * config.dv_y

    eqn_mass_conservation   = af.sum(zeroth_moment, 2) * dp_x * dp_y
    eqn_energy_conservation = af.sum(second_moment, 2) * dp_x * dp_y

    #return([eqn_mass_conservation, eqn_energy_conservation])
    return([np.array(eqn_mass_conservation), np.array(eqn_energy_conservation)])

def compute_jacobian(guess, compute_residual, da, args):
    
    # shape of guess is [num_vars](Ny, Nx)
    num_vars = len(guess)
    guess_plus_eps = args.guess_plus_eps
    jacobian       = args.jacobian

    # Initialize the correct shapes for guess_plus_eps and jacobian
    for row in range(num_vars):
        guess_plus_eps[row][:] = guess[row]

    epsilon = 4e-8
    residual = compute_residual(guess, da, args)
    for row in range(num_vars):
        
        small_var           = af.abs(guess[row]) < 0.5 * epsilon
        guess_plus_eps[row] = guess[row] + epsilon * guess[row]*(1. - small_var) + small_var*epsilon
        residual_plus_eps   = compute_residual(guess_plus_eps, da, args)

        for column in range(num_vars):

            jacobian[row][column] = \
                (residual_plus_eps[column] - residual[column]) \
              / (guess_plus_eps[row] - guess[row])

        guess_plus_eps[row][:] = guess[row]

    return(jacobian)

#def f_ee_scattering(da, args):
#
#    from scipy.optimize import root
#    guess = [np.array(args.mu),          np.array(args.T), \
#             np.array(args.vel_drift_x), np.array(args.vel_drift_y)]
#
#    soln  = root(compute_residual_ee, guess, args=(da, args), \
#                 method='krylov', options={'disp':True, 'fatol' : 1e-10})
#
#    args.mu           = af.Array.as_type(af.to_array(soln.x[0]), af.Dtype.f64)
#    args.T            = af.Array.as_type(af.to_array(soln.x[1]), af.Dtype.f64)
#    args.vel_drift_x  = af.Array.as_type(af.to_array(soln.x[2]), af.Dtype.f64)
#    args.vel_drift_y  = af.Array.as_type(af.to_array(soln.x[3]), af.Dtype.f64)
#
#    config = args.config
#
#    p_x = config.h_cross * args.vel_x # (Nx*Ny, Nvy, Nvx, Nvz)
#    p_y = config.h_cross * args.vel_y # (Nx*Ny, Nvy, Nvx, Nvz)
# 
#    E_upper, E_lower = config.band_energy(p_x, p_y)
#    k                = config.boltzmann_constant
#
#    E_upper = non_linear_solver.convert.to_positionsExpanded \
#                (da, config, E_upper) # (Ny, Nx, Nvy*Nvx)
#    p_x     = non_linear_solver.convert.to_positionsExpanded \
#                (da, config, p_x)     # (Ny, Nx, Nvy*Nvx)
#    p_y     = non_linear_solver.convert.to_positionsExpanded \
#                (da, config, p_y)     # (Ny, Nx, Nvy*Nvx)
#
#    mu           = af.tile(args.mu,           1, 1, config.N_vel_x * config.N_vel_y)
#    T            = af.tile(args.T,            1, 1, config.N_vel_x * config.N_vel_y)
#    vel_drift_x  = af.tile(args.vel_drift_x,  1, 1, config.N_vel_x * config.N_vel_y)
#    vel_drift_y  = af.tile(args.vel_drift_y,  1, 1, config.N_vel_x * config.N_vel_y)
#    
#    fermi_dirac = 1./(af.exp( (E_upper - vel_drift_x*p_x - vel_drift_y*p_y - mu)/(k*T) ) + 1.)
#
#    print("    max(v_drift_x) = ", af.max(args.vel_drift_x), "mean(v_drift_x) = ", af.mean(args.vel_drift_x))
#    print("")
#
#    return(fermi_dirac)

def f_ee_scattering(da, args):

    config = args.config

    nonlinear_iters = 3
    for n in range(nonlinear_iters):

        p_x = config.h_cross * args.vel_x # (Nx*Ny, Nvy, Nvx, Nvz)
        p_y = config.h_cross * args.vel_y # (Nx*Ny, Nvy, Nvx, Nvz)

        mu_ee       = af.tile(args.mu_ee, 1, 1, config.N_vel_x * config.N_vel_y)
        T_ee        = af.tile(args.T_ee,  1, 1, config.N_vel_x * config.N_vel_y)
        vel_drift_x = af.tile(args.vel_drift_x,  1, 1, config.N_vel_x * config.N_vel_y)
        vel_drift_y = af.tile(args.vel_drift_y,  1, 1, config.N_vel_x * config.N_vel_y)

        E_upper, E_lower = config.band_energy(p_x, p_y)
        k                = config.boltzmann_constant

        p_x     = non_linear_solver.convert.to_positionsExpanded \
                    (da, config, p_x)          # (Ny, Nx, Nvy*Nvx)
        p_y     = non_linear_solver.convert.to_positionsExpanded \
                    (da, config, p_y)          # (Ny, Nx, Nvy*Nvx)
        E_upper = non_linear_solver.convert.to_positionsExpanded \
                    (da, config, E_upper) # (Ny, Nx, Nvy*Nvx)

        tmp1        = (E_upper - mu_ee - p_x*vel_drift_x - p_y*vel_drift_y)
        tmp         = (tmp1/(k*T_ee))
        denominator = (k*T_ee**2.*(af.exp(tmp) + 2. + af.exp(-tmp)) )

        a_0 = T_ee       / denominator
        a_1 = tmp1       / denominator
        a_2 = T_ee * p_x / denominator
        a_3 = T_ee * p_y / denominator

        af.eval(a_0, a_1, a_2, a_3)

        dp_x = config.h_cross * config.dv_x
        dp_y = config.h_cross * config.dv_y

        a_00 = af.sum(a_0, 2) * dp_x * dp_y
        a_01 = af.sum(a_1, 2) * dp_x * dp_y
        a_02 = af.sum(a_2, 2) * dp_x * dp_y
        a_03 = af.sum(a_3, 2) * dp_x * dp_y

        a_10 = af.sum(E_upper * a_0, 2) * dp_x * dp_y
        a_11 = af.sum(E_upper * a_1, 2) * dp_x * dp_y
        a_12 = af.sum(E_upper * a_2, 2) * dp_x * dp_y
        a_13 = af.sum(E_upper * a_3, 2) * dp_x * dp_y

        a_20 = af.sum(p_x * a_0, 2) * dp_x * dp_y
        a_21 = af.sum(p_x * a_1, 2) * dp_x * dp_y
        a_22 = af.sum(p_x * a_2, 2) * dp_x * dp_y
        a_23 = af.sum(p_x * a_3, 2) * dp_x * dp_y

        a_30 = af.sum(p_y * a_0, 2) * dp_x * dp_y
        a_31 = af.sum(p_y * a_1, 2) * dp_x * dp_y
        a_32 = af.sum(p_y * a_2, 2) * dp_x * dp_y
        a_33 = af.sum(p_y * a_3, 2) * dp_x * dp_y

        A = [ [a_00, a_01, a_02, a_03], \
              [a_10, a_11, a_12, a_13], \
              [a_20, a_21, a_22, a_23], \
              [a_30, a_31, a_32, a_33]  \
            ]
        
        fermi_dirac = 1./(af.exp( (E_upper - vel_drift_x*p_x - vel_drift_y*p_y - mu_ee)/(k*T_ee) ) + 1.)
        af.eval(fermi_dirac)

        zeroth_moment  =         (args.f - fermi_dirac)
        second_moment  = E_upper*(args.f - fermi_dirac)
        first_moment_x =     p_x*(args.f - fermi_dirac)
        first_moment_y =     p_y*(args.f - fermi_dirac)

        eqn_mass_conservation   = af.sum(zeroth_moment,  2) * dp_x * dp_y
        eqn_energy_conservation = af.sum(second_moment,  2) * dp_x * dp_y
        eqn_mom_x_conservation  = af.sum(first_moment_x, 2) * dp_x * dp_y
        eqn_mom_y_conservation  = af.sum(first_moment_y, 2) * dp_x * dp_y

        residual = [eqn_mass_conservation, \
                    eqn_energy_conservation, \
                    eqn_mom_x_conservation, \
                    eqn_mom_y_conservation]
#        l2_norm  =  af.pow(residual[0], 2.) \
#                  + af.pow(residual[1], 2.) \
#                  + af.pow(residual[2], 2.) \
#                  + af.pow(residual[3], 2.)
#        af.eval(l2_norm)

        #error_norm = af.max(l2_norm)
        #error_norm = af.norm(af.flat(l2_norm))
        error_norm = np.max([af.max(af.abs(residual[0])),
                             af.max(af.abs(residual[1])),
                             af.max(af.abs(residual[2])),
                             af.max(af.abs(residual[3]))]
                           )
        print("    ||residual_ee|| = ", error_norm)

        if (error_norm < 1e-13):
            return(fermi_dirac)

        b_0 = eqn_mass_conservation  
        b_1 = eqn_energy_conservation
        b_2 = eqn_mom_x_conservation 
        b_3 = eqn_mom_y_conservation 
        b   = [b_0, b_1, b_2, b_3]

        # Solve Ax = b
        # where A == Jacobian,
        #       x == delta guess (correction to guess), 
        #       b = -residual

        A_inv = inverse_4x4_matrix(A)

        x_0 = A_inv[0][0]*b[0] + A_inv[0][1]*b[1] + A_inv[0][2]*b[2] + A_inv[0][3]*b[3]
        x_1 = A_inv[1][0]*b[0] + A_inv[1][1]*b[1] + A_inv[1][2]*b[2] + A_inv[1][3]*b[3]
        x_2 = A_inv[2][0]*b[0] + A_inv[2][1]*b[1] + A_inv[2][2]*b[2] + A_inv[2][3]*b[3]
        x_3 = A_inv[3][0]*b[0] + A_inv[3][1]*b[1] + A_inv[3][2]*b[2] + A_inv[3][3]*b[3]

        delta_mu = x_0
        delta_T  = x_1
        delta_vx = x_2
        delta_vy = x_3
    
        args.step_length = 1.
#        f0          = 0.5 * l2_norm
#        f_prime0    = -2. * f0
#        step_length_iters = 0
#        for s in range(step_length_iters):
#            mu_ee       = args.mu_ee       + args.step_length*delta_mu
#            T_ee        = args.T_ee        + args.step_length*delta_T
#            vel_drift_x = args.vel_drift_x + args.step_length*delta_vx
#            vel_drift_y = args.vel_drift_y + args.step_length*delta_vy
#
#            af.eval(mu_ee, T_ee, vel_drift_x, vel_drift_y)
#
#            guess    = [mu_ee, T_ee, vel_drift_x, vel_drift_y]
#            residual = compute_residual_ee(guess, da, args) 
#
#            l2_norm  =  af.pow(residual[0], 2.) \
#                      + af.pow(residual[1], 2.) \
#                      + af.pow(residual[2], 2.) \
#                      + af.pow(residual[3], 2.)
#
#            af.eval(l2_norm)
#            f1       = 0.5 * l2_norm
#
#            alpha             = 1e-4
#            line_search_floor = 1e-24
#            condition = f1 > (f0*(1. - alpha*args.step_length) + line_search_floor)
#            
#            denom            =   (f1-f0-f_prime0*args.step_length) * condition \
#                               + (1. - condition)
#            next_step_length = -f_prime0*args.step_length*args.step_length/denom/2.
#            args.step_length =   args.step_length* (1. - condition) \
#                               + condition  * next_step_length
#            af.eval(args.step_length)
#
#            condition_indices = af.where(condition > 0);
#            print("        Linesearch iter = ", s, \
#                  "min steplength = ", af.min(args.step_length), \
#                  "indices = ", condition_indices.elements())
#
#            if (condition_indices.elements() == 0):
#                break

        # step length has now been set
        args.mu_ee       = args.mu_ee       + args.step_length*delta_mu
        args.T_ee        = args.T_ee        + args.step_length*delta_T
        args.vel_drift_x = args.vel_drift_x + args.step_length*delta_vx
        args.vel_drift_y = args.vel_drift_y + args.step_length*delta_vy

        af.eval(args.mu_ee, args.T_ee, args.vel_drift_x, args.vel_drift_y)
        guess = [args.mu_ee, args.T_ee, args.vel_drift_x, args.vel_drift_y]

    mu_ee       = af.tile(args.mu_ee, 1, 1, config.N_vel_x * config.N_vel_y)
    T_ee        = af.tile(args.T_ee,  1, 1, config.N_vel_x * config.N_vel_y)
    vel_drift_x = af.tile(args.vel_drift_x,  1, 1, config.N_vel_x * config.N_vel_y)
    vel_drift_y = af.tile(args.vel_drift_y,  1, 1, config.N_vel_x * config.N_vel_y)

    fermi_dirac = 1./(af.exp( (E_upper - vel_drift_x*p_x - vel_drift_y*p_y - mu_ee)/(k*T_ee) ) + 1.)
    af.eval(fermi_dirac)

    zeroth_moment  =          args.f - fermi_dirac
    second_moment  = E_upper*(args.f - fermi_dirac)
    first_moment_x =     p_x*(args.f - fermi_dirac)
    first_moment_y =     p_y*(args.f - fermi_dirac)
    
    eqn_mass_conservation   = af.sum(zeroth_moment,  2) * dp_x * dp_y
    eqn_energy_conservation = af.sum(second_moment,  2) * dp_x * dp_y
    eqn_mom_x_conservation  = af.sum(first_moment_x, 2) * dp_x * dp_y
    eqn_mom_y_conservation  = af.sum(first_moment_y, 2) * dp_x * dp_y

    residual = [eqn_mass_conservation, \
                eqn_energy_conservation, \
                eqn_mom_x_conservation, \
                eqn_mom_y_conservation]
#    l2_norm  =  af.pow(residual[0], 2.) \
#              + af.pow(residual[1], 2.) \
#              + af.pow(residual[2], 2.) \
#              + af.pow(residual[3], 2.)
#    af.eval(l2_norm)

    #error_norm = af.norm(af.flat(l2_norm))
    #error_norm = af.max(l2_norm)
    error_norm = np.max([af.max(af.abs(residual[0])),
                         af.max(af.abs(residual[1])),
                         af.max(af.abs(residual[2])),
                         af.max(af.abs(residual[3]))]
                       )
    print("    ||residual_ee|| = ", error_norm)
    print("    ------------------")

    return(fermi_dirac)

def f_defect_scattering(da, args):

#    from scipy.optimize import root
#    guess = [np.array(args.mu), np.array(args.T)]
#    soln  = root(compute_residual_defect, guess, args=(da, args), \
#                 method='krylov', options={'disp':True, 'fatol' : 1e-10})
#
#    args.mu = af.Array.as_type(af.to_array(soln.x[0]), af.Dtype.f64)
#    args.T  = af.Array.as_type(af.to_array(soln.x[1]), af.Dtype.f64)
#
#    config = args.config
#
#    p_x = config.h_cross * args.vel_x # (Nx*Ny, Nvy, Nvx, Nvz)
#    p_y = config.h_cross * args.vel_y # (Nx*Ny, Nvy, Nvx, Nvz)
# 
#    E_upper, E_lower = config.band_energy(p_x, p_y)
#    k                = config.boltzmann_constant
#
#    E_upper = non_linear_solver.convert.to_positionsExpanded \
#                (da, config, E_upper) # (Ny, Nx, Nvy*Nvx)
#
#    mu = af.tile(args.mu, 1, 1, config.N_vel_x * config.N_vel_y)
#    T  = af.tile(args.T,  1, 1, config.N_vel_x * config.N_vel_y)
#    
#    fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)
#
#    return(fermi_dirac)


    # (1) Need to compute mu and T, for which we have to solve a set of coupled
    #     non-linear equations:
    #
    # \sum_{+, -} \int f_{+,-} d^2p = \int f_FD_{+, -}(mu, T) d^2p -- (1)
    # \sum_{+, -} \int E_{+, -} f_{+, -} d^2p
    #       = \sum_{+, -} \int E_{+, -} f_FD_{+, -}(mu, T)  d^2p   -- (2)

    # Need a guess. Use the values at the previous time step.
    # (Nx*Ny, Nvy, Nvx*Nvz)

    config = args.config

    nonlinear_iters = 3
    for n in range(nonlinear_iters):

        p_x = config.h_cross * args.vel_x # (Nx*Ny, Nvy, Nvx, Nvz)
        p_y = config.h_cross * args.vel_y # (Nx*Ny, Nvy, Nvx, Nvz)

        mu = af.tile(args.mu, 1, 1, config.N_vel_x * config.N_vel_y)
        T  = af.tile(args.T,  1, 1, config.N_vel_x * config.N_vel_y)

        E_upper, E_lower = config.band_energy(p_x, p_y)
        k                = config.boltzmann_constant

        E_upper = non_linear_solver.convert.to_positionsExpanded \
                    (da, config, E_upper) # (Ny, Nx, Nvy*Nvx)

        tmp         = ((E_upper - mu)/(k*T))
        denominator = (k*T**2.*(af.exp(tmp) + 2. + af.exp(-tmp)) )

        dp_x = config.h_cross * config.dv_x
        dp_y = config.h_cross * config.dv_y

        a00 = af.sum(T                      / denominator, 2) * dp_x * dp_y
        a01 = af.sum((E_upper - mu)         / denominator, 2) * dp_x * dp_y
        a10 = af.sum(E_upper*T              / denominator, 2) * dp_x * dp_y
        a11 = af.sum(E_upper*(E_upper - mu) / denominator, 2) * dp_x * dp_y

        # Solve Ax = b
        # where A == Jacobian,
        #       x == delta guess (correction to guess), 
        #       b = -residual

        fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)
        af.eval(fermi_dirac)

        zeroth_moment =          args.f - fermi_dirac
        second_moment = E_upper*(args.f - fermi_dirac)
    
        eqn_mass_conservation   = af.sum(zeroth_moment, 2) * dp_x * dp_y
        eqn_energy_conservation = af.sum(second_moment, 2) * dp_x * dp_y

        error_mass_conservation   = af.max(af.abs(eqn_mass_conservation))
        error_energy_conservation = af.max(af.abs(eqn_energy_conservation))

        residual   = [eqn_mass_conservation, eqn_energy_conservation]
        error_norm = np.max([af.max(af.abs(residual[0])), 
                             af.max(af.abs(residual[1]))]
                           )
        print("    ||residual_defect|| = ", error_norm)

        if (error_norm < 1e-9):
            return(fermi_dirac)

        b0  = eqn_mass_conservation
        b1  = eqn_energy_conservation

        det      =   a01*a10 - a00*a11
        delta_mu = -(a11*b0 - a01*b1)/det
        delta_T  =  (a10*b0 - a00*b1)/det

        args.mu = args.mu + delta_mu
        args.T  = args.T  + delta_T

        af.eval(args.mu, args.T)
        guess = [args.mu, args.T]

    mu = af.tile(args.mu, 1, 1, config.N_vel_x * config.N_vel_y)
    T  = af.tile(args.T,  1, 1, config.N_vel_x * config.N_vel_y)

    fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)
    af.eval(fermi_dirac)

    zeroth_moment =          args.f - fermi_dirac
    second_moment = E_upper*(args.f - fermi_dirac)
   
    eqn_mass_conservation   = af.sum(zeroth_moment, 2) * dp_x * dp_y
    eqn_energy_conservation = af.sum(second_moment, 2) * dp_x * dp_y

    residual   = [eqn_mass_conservation, eqn_energy_conservation]
    error_norm = np.max([af.max(af.abs(residual[0])), 
                         af.max(af.abs(residual[1]))]
                       )
    print("    ||residual_defect|| = ", error_norm)
    print("    ------------------")

    return(fermi_dirac)


def inverse_4x4_matrix(A):
# TO TEST:
#        A_test     = np.random.rand(4, 4)
#        A_inv_test = np.linalg.inv(A_test)
#        A_inv      = np.array(inverse_4x4_matrix(A_test))
#        print("err = ", np.max(np.abs(A_inv - A_inv_test)))


    det = \
        A[0][0]*A[1][1]*A[2][2]*A[3][3] \
      + A[0][0]*A[1][2]*A[2][3]*A[3][1] \
      + A[0][0]*A[1][3]*A[2][1]*A[3][2] \
      + A[0][1]*A[1][0]*A[2][3]*A[3][2] \
      + A[0][1]*A[1][2]*A[2][0]*A[3][3] \
      + A[0][1]*A[1][3]*A[2][2]*A[3][0] \
      + A[0][2]*A[1][0]*A[2][1]*A[3][3] \
      + A[0][2]*A[1][1]*A[2][3]*A[3][0] \
      + A[0][2]*A[1][3]*A[2][0]*A[3][1] \
      + A[0][3]*A[1][0]*A[2][2]*A[3][1] \
      + A[0][3]*A[1][1]*A[2][0]*A[3][2] \
      + A[0][3]*A[1][2]*A[2][1]*A[3][0] \
      - A[0][0]*A[1][1]*A[2][3]*A[3][2] \
      - A[0][0]*A[1][2]*A[2][1]*A[3][3] \
      - A[0][0]*A[1][3]*A[2][2]*A[3][1] \
      - A[0][1]*A[1][0]*A[2][2]*A[3][3] \
      - A[0][1]*A[1][2]*A[2][3]*A[3][0] \
      - A[0][1]*A[1][3]*A[2][0]*A[3][2] \
      - A[0][2]*A[1][0]*A[2][3]*A[3][1] \
      - A[0][2]*A[1][1]*A[2][0]*A[3][3] \
      - A[0][2]*A[1][3]*A[2][1]*A[3][0] \
      - A[0][3]*A[1][0]*A[2][1]*A[3][2] \
      - A[0][3]*A[1][1]*A[2][2]*A[3][0] \
      - A[0][3]*A[1][2]*A[2][0]*A[3][1]

    af.eval(det)

    A_inv = [[0, 0, 0, 0], 
             [0, 0, 0, 0], 
             [0, 0, 0, 0], 
             [0, 0, 0, 0]
            ]

    A_inv[0][0] = \
        (  A[1][1]*A[2][2]*A[3][3] 
         + A[1][2]*A[2][3]*A[3][1] 
         + A[1][3]*A[2][1]*A[3][2] 
         - A[1][1]*A[2][3]*A[3][2] 
         - A[1][2]*A[2][1]*A[3][3] 
         - A[1][3]*A[2][2]*A[3][1])/det
  
    A_inv[0][1] = \
        (  A[0][1]*A[2][3]*A[3][2] 
         + A[0][2]*A[2][1]*A[3][3] 
         + A[0][3]*A[2][2]*A[3][1] 
         - A[0][1]*A[2][2]*A[3][3] 
         - A[0][2]*A[2][3]*A[3][1] 
         - A[0][3]*A[2][1]*A[3][2])/det
  
    A_inv[0][2] = \
        (  A[0][1]*A[1][2]*A[3][3]
         + A[0][2]*A[1][3]*A[3][1]
         + A[0][3]*A[1][1]*A[3][2]
         - A[0][1]*A[1][3]*A[3][2]
         - A[0][2]*A[1][1]*A[3][3]
         - A[0][3]*A[1][2]*A[3][1])/det
  
    A_inv[0][3] = \
        (  A[0][1]*A[1][3]*A[2][2]
         + A[0][2]*A[1][1]*A[2][3]
         + A[0][3]*A[1][2]*A[2][1]
         - A[0][1]*A[1][2]*A[2][3]
         - A[0][2]*A[1][3]*A[2][1]
         - A[0][3]*A[1][1]*A[2][2])/det
  
    # b21
    A_inv[1][0] = \
        (  A[1][0]*A[2][3]*A[3][2]
         + A[1][2]*A[2][0]*A[3][3]
         + A[1][3]*A[2][2]*A[3][0]
         - A[1][0]*A[2][2]*A[3][3]
         - A[1][2]*A[2][3]*A[3][0]
         - A[1][3]*A[2][0]*A[3][2])/det
    
    A_inv[1][1] = \
        (  A[0][0]*A[2][2]*A[3][3]
         + A[0][2]*A[2][3]*A[3][0]
         + A[0][3]*A[2][0]*A[3][2]
         - A[0][0]*A[2][3]*A[3][2]
         - A[0][2]*A[2][0]*A[3][3]
         - A[0][3]*A[2][2]*A[3][0])/det
  
    A_inv[1][2] = \
        (  A[0][0]*A[1][3]*A[3][2]
         + A[0][2]*A[1][0]*A[3][3]
         + A[0][3]*A[1][2]*A[3][0]
         - A[0][0]*A[1][2]*A[3][3]
         - A[0][2]*A[1][3]*A[3][0]
         - A[0][3]*A[1][0]*A[3][2])/det
  
    A_inv[1][3] = \
        (  A[0][0]*A[1][2]*A[2][3]
         + A[0][2]*A[1][3]*A[2][0]
         + A[0][3]*A[1][0]*A[2][2]
         - A[0][0]*A[1][3]*A[2][2]
         - A[0][2]*A[1][0]*A[2][3]
         - A[0][3]*A[1][2]*A[2][0])/det
  
    # b31
    A_inv[2][0] = \
        (  A[1][0]*A[2][1]*A[3][3]
         + A[1][1]*A[2][3]*A[3][0]
         + A[1][3]*A[2][0]*A[3][1]
         - A[1][0]*A[2][3]*A[3][1]
         - A[1][1]*A[2][0]*A[3][3]
         - A[1][3]*A[2][1]*A[3][0])/det

    # b32
    A_inv[2][1] = \
        (  A[0][0]*A[2][3]*A[3][1]
         + A[0][1]*A[2][0]*A[3][3]
         + A[0][3]*A[2][1]*A[3][0]
         - A[0][0]*A[2][1]*A[3][3]
         - A[0][1]*A[2][3]*A[3][0]
         - A[0][3]*A[2][0]*A[3][1])/det
  
    A_inv[2][2] = \
        (  A[0][0]*A[1][1]*A[3][3]
         + A[0][1]*A[1][3]*A[3][0]
         + A[0][3]*A[1][0]*A[3][1]
         - A[0][0]*A[1][3]*A[3][1]
         - A[0][1]*A[1][0]*A[3][3]
         - A[0][3]*A[1][1]*A[3][0])/det
  
    A_inv[2][3] = \
        (  A[0][0]*A[1][3]*A[2][1]
         + A[0][1]*A[1][0]*A[2][3]
         + A[0][3]*A[1][1]*A[2][0]
         - A[0][0]*A[1][1]*A[2][3]
         - A[0][1]*A[1][3]*A[2][0]
         - A[0][3]*A[1][0]*A[2][1])/det
  
    # b41
    A_inv[3][0] = \
        (  A[1][0]*A[2][2]*A[3][1]  
         + A[1][1]*A[2][0]*A[3][2]  
         + A[1][2]*A[2][1]*A[3][0]  
         - A[1][0]*A[2][1]*A[3][2]  
         - A[1][1]*A[2][2]*A[3][0]  
         - A[1][2]*A[2][0]*A[3][1])/det

    # b42
    A_inv[3][1] = \
        (  A[0][0]*A[2][1]*A[3][2]  
         + A[0][1]*A[2][2]*A[3][0]  
         + A[0][2]*A[2][0]*A[3][1]  
         - A[0][0]*A[2][2]*A[3][1]  
         - A[0][1]*A[2][0]*A[3][2]  
         - A[0][2]*A[2][1]*A[3][0])/det

    # b43
    A_inv[3][2] = \
        (  A[0][0]*A[1][2]*A[3][1]  
         + A[0][1]*A[1][0]*A[3][2]  
         + A[0][2]*A[1][1]*A[3][0]  
         - A[0][0]*A[1][1]*A[3][2]  
         - A[0][1]*A[1][2]*A[3][0]  
         - A[0][2]*A[1][0]*A[3][1])/det
  
    A_inv[3][3] = \
        (  A[0][0]*A[1][1]*A[2][2]  
         + A[0][1]*A[1][2]*A[2][0]  
         + A[0][2]*A[1][0]*A[2][1]  
         - A[0][0]*A[1][2]*A[2][1]  
         - A[0][1]*A[1][0]*A[2][2]  
         - A[0][2]*A[1][1]*A[2][0])/det

    for i in range(4):
        for j in range(4):
            af.eval(A_inv[i][j])

    return(A_inv)
