import numpy as np
import arrayfire as af

class options:
  def __init__(self):
    pass

def configuration_object(params):
  """
  Used to define an object that contains data about the parameters 
  that are used in the simulation

  Parameters:
  -----------
    params : Name of the file that contains the parameters for the 
             simulation run is passed to this function. 

  Output:
  -------
    config : Object whose attributes contain all the simulation parameters. 
             This is passed to the remaining solver functions.
  """

  # Declaring an instance of the class:
  config      = options()
  
  # Definining the dimensionality in velocity space:
  config.mode = params.mode

  # Defining non-dimensional constants:
  config.mass_particle      = params.constants['mass_particle']
  config.boltzmann_constant = params.constants['boltzmann_constant']

  # Defining non-dimensional background quantities for electrons:
  config.rho_background         = params.background_electrons['rho']
  config.temperature_background = params.background_electrons['temperature']
  config.vel_bulk_x_background  = params.background_electrons['vel_bulk_x']
  config.vel_bulk_y_background  = params.background_electrons['vel_bulk_y']
  config.vel_bulk_z_background  = params.background_electrons['vel_bulk_z']

  # Defining amplitude and wave number of the perturbation in the domain:
  config.pert_real = params.perturbation['pert_real']
  config.pert_imag = params.perturbation['pert_imag']
  config.k_x       = params.perturbation['k_x']
  config.k_y       = params.perturbation['k_y']

  # Defining the resolution in position space:
  config.N_x     = params.position_space['N_x']
  config.x_start = params.position_space['x_start']
  config.x_end   = params.position_space['x_end']
  config.dx      = (config.x_end - config.x_start)/config.N_x
  
  config.N_y     = params.position_space['N_y']
  config.y_start = params.position_space['y_start']
  config.y_end   = params.position_space['y_end']
  config.dy      = (config.y_end - config.y_start)/config.N_y

  config.N_ghost = params.position_space['N_ghost']

  # Defining the resolution in velocity space:
  config.N_vel_x   = params.velocity_space['N_vel_x']
  config.vel_x_max = params.velocity_space['vel_x_max']
  config.dv_x      = (2*config.vel_x_max)/config.N_vel_x
  
  config.N_vel_y   = params.velocity_space['N_vel_y']
  config.vel_y_max = params.velocity_space['vel_y_max']
  config.dv_y      = (2*config.vel_y_max)/config.N_vel_y

  config.N_vel_z   = params.velocity_space['N_vel_z']
  config.vel_z_max = params.velocity_space['vel_z_max']
  config.dv_z      = (2*config.vel_z_max)/config.N_vel_z

  # Defining the boundary condition that is utilized in x and y directions:
  config.bc_in_x = params.boundary_conditions['in_x']
  config.bc_in_y = params.boundary_conditions['in_y']

#  # Defining the quantities at the boundaries for Dirichlet boundary conditions:
#  if(config.bc_in_x == 'dirichlet'):
#    config.left_rho         = params.boundary_conditions['left_rho']  
#    config.left_temperature = params.boundary_conditions['left_temperature']
#    config.left_vel_bulk_x  = params.boundary_conditions['left_vel_bulk_x']
#    config.left_vel_bulk_y  = params.boundary_conditions['left_vel_bulk_y']
#    config.left_vel_bulk_z  = params.boundary_conditions['left_vel_bulk_z']
#    
#    config.right_rho         = params.boundary_conditions['right_rho']  
#    config.right_temperature = params.boundary_conditions['right_temperature']
#    config.right_vel_bulk_x  = params.boundary_conditions['right_vel_bulk_x']
#    config.right_vel_bulk_y  = params.boundary_conditions['right_vel_bulk_y']
#    config.right_vel_bulk_z  = params.boundary_conditions['right_vel_bulk_z']
#
#  if(config.bc_in_y == 'dirichlet'):
#    config.bot_rho         = params.boundary_conditions['bot_rho']  
#    config.bot_temperature = params.boundary_conditions['bot_temperature']
#    config.bot_vel_bulk_x  = params.boundary_conditions['bot_vel_bulk_x']
#    config.bot_vel_bulk_y  = params.boundary_conditions['bot_vel_bulk_y']
#    config.bot_vel_bulk_z  = params.boundary_conditions['bot_vel_bulk_z']
#    
#    config.top_rho         = params.boundary_conditions['top_rho']  
#    config.top_temperature = params.boundary_conditions['top_temperature']
#    config.top_vel_bulk_x  = params.boundary_conditions['top_vel_bulk_x']
#    config.top_vel_bulk_y  = params.boundary_conditions['top_vel_bulk_y']
#    config.top_vel_bulk_z  = params.boundary_conditions['top_vel_bulk_z']

  # Defining the resolution parameters for time:
  config.final_time = params.time['final_time']
  config.dt         = params.time['dt']
  
  # Defining the charge of electrons and ions:  
  config.charge_electron = params.EM_fields['charge_electron']
  config.charge_ion      = params.EM_fields['charge_ion']
  config.fields_solver   = params.EM_fields['solver']

  # Defining the collisional time-scale utilized:
  config.collision_operator = params.collisions['collision_operator']
  config.tau                = params.collisions['tau']

  config.h_cross                       = 1
  config.number_of_bands               = 2
  config.chemical_potential_background = 0.005*0.
  config.fermi_velocity                = 137./300.
  config.tau_defect                    = 0.01
  config.tau_ee                        = np.inf
  config.delta_E_x_hat_ext             = 1e-5   + 0.*1j
  config.delta_E_y_hat_ext             = 0e-5 + 0*1j

  # Initialize the grids

  def band_energy(p_x, p_y):
    
    p = af.sqrt(p_x**2. + p_y**2.)

    return([ p*config.fermi_velocity, 
            -p*config.fermi_velocity
           ])

  def band_velocity(p_x, p_y):

    p     = af.sqrt(p_x**2. + p_y**2.)
    p_hat = [p_x / (p + 1e-20), p_y / (p + 1e-20)]

    v_f   = config.fermi_velocity

    upper_band_velocity =  [ v_f * p_hat[0],  v_f * p_hat[1]]
    lower_band_velocity =  [-v_f * p_hat[0], -v_f * p_hat[1]]

    return([upper_band_velocity, lower_band_velocity])

  config.band_energy   = band_energy
  config.band_velocity = band_velocity

  return config

def time_array(config):
  """
  Returns the value of the time_array at which we solve for in the 
  simulation. The time_array is set depending on the options which 
  have been mention in config.

  Parameters:
  -----------
    config: Object config which is obtained by 
            configuration_object() is passed to this file
  Output:
  -------
    time_array : Array that contains the values of time at which 
                 the simulation evaluates the physical quantities. 
  """
  final_time = config.final_time
  dt         = config.dt

  time_array = np.arange(0, final_time + dt, dt)

  return(time_array)
