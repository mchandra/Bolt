import numpy as np 

class config:
  pass

def set(params):
  """
  Used to set the parameters that are used in the simulation

  Parameters:
  -----------
    params : Name of the file that contains the parameters for the simulation run
             is passed to this function. 

  Output:
  -------
    config : Object whose attributes contain all the simulation parameters. This is
             passed to the remaining solver functions.
  """
  config.mode = params.mode

  config.mass_particle      = params.constants['mass_particle']
  config.boltzmann_constant = params.constants['boltzmann_constant']

  config.rho_background         = params.background_electrons['rho']
  config.temperature_background = params.background_electrons['temperature']
  config.vel_bulk_x_background  = params.background_electrons['vel_bulk_x']
  config.vel_bulk_y_background  = params.background_electrons['vel_bulk_y']

  config.pert_real = params.perturbation['pert_real']
  config.pert_imag = params.perturbation['pert_imag']
  config.k_x       = params.perturbation['k_x']
  config.k_y       = params.perturbation['k_y']

  config.N_x            = params.configuration_space['N_x']
  config.N_ghost_x      = params.configuration_space['N_ghost_x']
  config.left_boundary  = params.configuration_space['left_boundary']
  config.right_boundary = params.configuration_space['right_boundary']
  
  config.N_y          = params.configuration_space['N_y']
  config.N_ghost_y    = params.configuration_space['N_ghost_y']
  config.bot_boundary = params.configuration_space['bot_boundary']
  config.top_boundary = params.configuration_space['top_boundary']

  config.N_vel_x   = params.velocity_space['N_vel_x']
  config.vel_x_max = params.velocity_space['vel_x_max']
  config.N_vel_y   = params.velocity_space['N_vel_y']
  config.vel_y_max = params.velocity_space['vel_y_max']

  config.bc_in_x = params.boundary_conditions['in_x']
  config.bc_in_y = params.boundary_conditions['in_y']

  config.left_rho         = params.boundary_conditions['left_rho']  
  config.left_temperature = params.boundary_conditions['left_temperature']
  config.left_vel_bulk_x  = params.boundary_conditions['left_vel_bulk_x']
  config.left_vel_bulk_y  = params.boundary_conditions['left_vel_bulk_y']
  
  config.right_rho         = params.boundary_conditions['right_rho']  
  config.right_temperature = params.boundary_conditions['right_temperature']
  config.right_vel_bulk_x  = params.boundary_conditions['right_vel_bulk_x']
  config.right_vel_bulk_y  = params.boundary_conditions['right_vel_bulk_y']

  config.bot_rho         = params.boundary_conditions['bot_rho']  
  config.bot_temperature = params.boundary_conditions['bot_temperature']
  config.bot_vel_bulk_x  = params.boundary_conditions['bot_vel_bulk_x']
  config.bot_vel_bulk_y  = params.boundary_conditions['bot_vel_bulk_y']
  
  config.top_rho         = params.boundary_conditions['top_rho']  
  config.top_temperature = params.boundary_conditions['top_temperature']
  config.top_vel_bulk_x  = params.boundary_conditions['top_vel_bulk_x']
  config.top_vel_bulk_y  = params.boundary_conditions['top_vel_bulk_y']

  config.final_time = params.time['final_time']
  config.dt         = params.time['dt']
    
  config.fields_enabled  = params.EM_fields['enabled']
  config.charge_particle = params.EM_fields['charge_particle']

  config.collisions_enabled = params.collisions['enabled']
  config.collision_operator = params.collisions['collision_operator']
  config.tau                = params.collisions['tau']

  config.h_cross                       = 1
  config.number_of_bands               = 2
  config.chemical_potential_background = 0.005
  #config.chemical_potential_background = 0.00005
  config.fermi_velocity                = 137./300.
  config.tau_defect                    = 0.05
  config.tau_ee                        = np.inf
  config.delta_E_x_hat_ext             = 1e-5   + 0.*1j
  config.delta_E_y_hat_ext             = 0e-5 + 0*1j

  # Initialize the grids
  vel_x_max = config.vel_x_max
  vel_y_max = config.vel_y_max

  N_vel_x   = config.N_vel_x
  N_vel_y   = config.N_vel_y

  vel_x        = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  vel_y        = np.linspace(-vel_y_max, vel_y_max, N_vel_y)

  config.dv_x  = vel_x[1] - vel_x[0]
  config.dv_y  = vel_y[1] - vel_y[0]

  vel_x, vel_y = np.meshgrid(vel_x, vel_y)
  
  config.vel_x = vel_x
  config.vel_y = vel_y

  x        = np.linspace(0, 1, config.N_x)
  y        = np.linspace(0, 1, config.N_y)
  x, y     = np.meshgrid(x, y)

  config.x = x
  config.y = y

  def band_energy(p_x, p_y):
    
    p = np.sqrt(p_x**2. + p_y**2.)

    return np.array([ p*config.fermi_velocity, 
                     -p*config.fermi_velocity
                    ]
                   )

  def band_velocity(p_x, p_y):

    p     = np.sqrt(p_x**2. + p_y**2.)
    p_hat = np.array([p_x, p_y]) / (p + 1e-20)

    upper_band_velocity =  config.fermi_velocity * p_hat
    lower_band_velocity = -config.fermi_velocity * p_hat

    return np.array([upper_band_velocity, lower_band_velocity])

  p_x        = config.h_cross * vel_x
  p_y        = config.h_cross * vel_y
  config.p_x = p_x
  config.p_y = p_y
  config.dp_x = config.h_cross * config.dv_x
  config.dp_y = config.h_cross * config.dv_y

  config.band_energies   = band_energy(p_x, p_y)
  config.band_velocities = band_velocity(p_x, p_y)

  return config

def f_background(config):
  """
  Returns the value of f_background, depending on the parameters set in 
  the config object

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

  Output:
  -------
    f_background : Array which contains the values of f_background at different values
                   of vel_x.
  """
  
  rho     = config.rho_background
  m       = config.mass_particle
  k       = config.boltzmann_constant
  T       = config.temperature_background
  vx_bulk = config.vel_bulk_x_background
  vy_bulk = config.vel_bulk_y_background
  vx      = config.vel_x
  vy      = config.vel_y

#  # CAUTION: The normalization prefactor depends on the velocity space dimension
#
#  f_background =   rho * (m/(2*np.pi*k*T)) \
#                 * np.exp(-m*( (vx - vx_bulk)**2 + (vy - vy_bulk)**2 ) / (2*k*T) )

  energy = config.band_energies
  mu     = config.chemical_potential_background


  def heaviside_theta(x):
    if (x<=0):
        return 0.
    else:
        return 1. 
  
  heaviside_theta = np.vectorize(heaviside_theta)

  f_fermi_dirac  =  1/(np.exp((energy - mu)/(k*T)) + 1)

  return f_fermi_dirac

def dfdp_background(config):
  f_background_local = f_background(config)

  df_dpy = np.zeros(f_background_local.shape)
  df_dpx = np.zeros(f_background_local.shape)

  for band in range(config.number_of_bands):
    df_dpy_band, df_dpx_band = np.gradient(f_background_local[band], 
					   config.dp_y, config.dp_x
					  )

    df_dpy[band] = df_dpy_band
    df_dpx[band] = df_dpx_band

  return(np.array([df_dpy, df_dpx]) )


def time_array(config):
  """
  Returns the value of the time_array at which we solve for in the simulation. 
  The time_array is set depending on the options which have been mention in config.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

  Output:
  -------
    time_array : Array that contains the values of time at which the 
                 simulation evaluates the physical quantities. 
  """

  final_time = config.final_time
  dt         = config.dt

  time_array = np.arange(0, final_time + dt, dt)

  return time_array

def init_delta_f_hat(config):
  """
  Returns the initial value of delta_f_hat which is setup depending on
  the perturbation parameters set in config. 

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

  Output:
  -------
    delta_f_hat_initial : Array which contains the values of initial mode perturbation 
                          in the distribution function.

  """

  pert_real = config.pert_real 
  pert_imag = config.pert_imag 

#  delta_f_hat_initial = pert_real*f_background(config) +\
#                        pert_imag*f_background(config)*1j 

  E    = config.band_energies
  mu_0 = config.chemical_potential_background
  k    = config.boltzmann_constant
  T_0  = config.temperature_background

  tmp = (E - mu_0)/(k*T_0)

  delta_mu_hat = config.pert_real + 1j*config.pert_imag

#  delta_f_hat_initial = \
#    delta_mu_hat * np.exp(tmp) \
#  / (k*T * (np.exp(2.*tmp) + 2.*np.exp(tmp) + 1) )

  delta_f_hat_initial = \
    delta_mu_hat \
  / (k*T_0 * (np.exp(tmp) + 2. + np.exp(-tmp)) )

  return delta_f_hat_initial
