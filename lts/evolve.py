import numpy as np
from scipy.integrate import odeint
import params
import lts.initialize as initialize
from lts.collision_operators import BGK_collision_operator
from lts.collision_operators import RTA_defect_scattering
from lts.collision_operators import RTA_ee_scattering

config = initialize.set(params)

mass_particle      = config.mass_particle
boltzmann_constant = config.boltzmann_constant

rho_background         = config.rho_background
temperature_background = config.temperature_background

k_x = config.k_x   
k_y = config.k_y   

x     = config.x
y     = config.y

vel_x = config.vel_x
vel_y = config.vel_y

dv_x  = config.dv_x
dv_y  = config.dv_y

dp_x  = config.dp_x
dp_y  = config.dp_y

fields_enabled  = config.fields_enabled
charge_particle = config.charge_particle

collisions_enabled = config.collisions_enabled 

def compute_delta_rho_hat(delta_f_hat, config):

  delta_rho_hat = 4./(2.*np.pi)**2.*np.sum(delta_f_hat)*dv_x*dv_y

  return(delta_rho_hat)

def compute_delta_E_hat(delta_f_hat, config):

  delta_rho_hat = compute_delta_rho_hat(delta_f_hat, config)

  delta_phi_hat = - charge_particle * delta_rho_hat \
                  / (k_x**2 + k_y**2)
   
  delta_E_x_hat = - delta_phi_hat * (1j * k_x)
  delta_E_y_hat = - delta_phi_hat * (1j * k_y)

  return(np.array([delta_E_x_hat, delta_E_y_hat]) )

def compute_delta_J_hat(delta_f_hat, config):
  
  delta_J_x_hat = 0.; delta_J_y_hat = 0
  for band in range(config.number_of_bands):
    band_vel_x, band_vel_y = config.band_velocities[band]

    delta_J_x_hat +=  4./(2.*np.pi)**2. \
    		    * np.sum(band_vel_x*delta_f_hat[band, :, :]) * dp_x * dp_y
    delta_J_y_hat +=  4./(2.*np.pi)**2. \
    	 	    * np.sum(band_vel_y*delta_f_hat[band, :, :]) * dp_x * dp_y

  return(np.array([delta_J_x_hat, delta_J_y_hat]) )

def compute_delta_J(delta_f_hat, config):
 
  delta_J_x_hat, delta_J_y_hat = \
    compute_delta_J_hat(delta_f_hat, config)

  delta_J_x = delta_J_x_hat.real * np.cos(k_x*x + k_y*y) - \
              delta_J_x_hat.imag * np.sin(k_x*x + k_y*y)

  delta_J_y = delta_J_y_hat.real * np.cos(k_x*x + k_y*y) - \
              delta_J_y_hat.imag * np.sin(k_x*x + k_y*y)

  return(np.array([delta_J_x, delta_J_y]))

def compute_delta_rho(delta_f_hat, config):
  
  delta_rho_hat = compute_delta_rho_hat(delta_f_hat, config)
  
  delta_rho = delta_rho_hat.real * np.cos(k_x*x + k_y*y) - \
              delta_rho_hat.imag * np.sin(k_x*x + k_y*y)
 
  return(delta_rho)

def compute_delta_E(delta_f_hat, config):
  
  delta_E_x_hat, delta_E_y_hat = \
    compute_delta_E_hat(delta_f_hat, config)

  delta_E_x = delta_E_x_hat.real * np.cos(k_x*x + k_y*y) - \
              delta_E_x_hat.imag * np.sin(k_x*x + k_y*y)

  delta_E_y = delta_E_y_hat.real * np.cos(k_x*x + k_y*y) - \
              delta_E_y_hat.imag * np.sin(k_x*x + k_y*y)

  return(np.array([delta_E_x, delta_E_y]))

def ddelta_f_hat_dt(delta_f_hat, config):
  """
  Returns the value of the derivative of the mode perturbation of the distribution 
  function with respect to time. This is used to evolve the system with time.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

    delta_f_hat: Mode perturbation of the distribution function that is passed to the function.
                 The array fed to this function is the result of the last time-step's integration.
                 At t=0 the initial mode perturbation of the system as declared in the configuration
                 file is passed to this function.

  Output:
  -------
    ddelta_f_hat_dt : Array which contains the values of the derivative of the Fourier mode 
                      perturbation of the distribution function with respect to time.
  """

  delta_E_x_hat, delta_E_y_hat = compute_delta_E_hat(delta_f_hat, config)

  dfdv_y_background, dfdv_x_background = initialize.dfdp_background(config)

  fields_term = np.zeros([config.number_of_bands,
                          config.N_vel_y, config.N_vel_x
                         ], dtype=np.complex128
                        )

  q_by_m = charge_particle / mass_particle
  for band in range(config.number_of_bands):

    delta_E_x_hat_ext = config.delta_E_x_hat_ext
    delta_E_y_hat_ext = config.delta_E_y_hat_ext

    fields_term[band, :, :] = \
        q_by_m \
      * ((delta_E_x_hat + delta_E_x_hat_ext)* dfdv_x_background[band, :, :] + \
         (delta_E_y_hat + delta_E_y_hat_ext)* dfdv_y_background[band, :, :] \
        )

  #C_f   = int(collisions_enabled)*BGK_collision_operator(config, delta_f_hat)
  C_f   = RTA_defect_scattering(config, delta_f_hat) + \
          RTA_ee_scattering(config, delta_f_hat)

  #ddelta_f_hat_dt = -1j * (k_x * vel_x + k_y * vel_y) * delta_f_hat + fields_term + C_f

  d_dx            = 1j*np.array([k_x, k_y])

  v_dot_d_dx = np.zeros([config.number_of_bands,
                         config.N_vel_y, config.N_vel_x
                        ], dtype=np.complex128
                       )

  for band in range(config.number_of_bands):
    band_vel_x, band_vel_y = config.band_velocities[band]
    
    v_dot_d_dx[band, :, :] = band_vel_x*d_dx[0] + \
			     band_vel_y*d_dx[1]

  ddelta_f_hat_dt = - v_dot_d_dx * delta_f_hat \
                    - fields_term + C_f

  return ddelta_f_hat_dt

def RK4_step(config, delta_f_hat_initial, dt):

  k1 = ddelta_f_hat_dt(delta_f_hat_initial, config)
  k2 = ddelta_f_hat_dt(delta_f_hat_initial + 0.5*k1*dt, config)
  k3 = ddelta_f_hat_dt(delta_f_hat_initial + 0.5*k2*dt, config)
  k4 = ddelta_f_hat_dt(delta_f_hat_initial + k3*dt, config)

  return(delta_f_hat_initial + ((k1+2*k2+2*k3+k4)/6)*dt)

def time_integration(config, delta_f_hat_initial, time_array):
  """
  Performs the time integration for the simulation. This is the main function that
  evolves the system in time. The parameters this function evolves for are dictated
  by the parameters as has been set in the config object. Final distribution function
  and the array that shows the evolution of rho_hat is returned by this function.

  Parameters:
  -----------   
    config : Object config which is obtained by set() is passed to this file

    delta_f_hat_initial : Array containing the initial values of the delta_f_hat. The value
                          for this function is typically obtained from the appropriately named 
                          function from the initialize submodule.

    time_array : Array which consists of all the time points at which we are evolving the system.
                 Data such as the mode amplitude of the density perturbation is also computed at 
                 the time points.

  Output:
  -------
    density_data : The value of the amplitude of the mode expansion of the density perturbation computed at
                   the various points in time as declared in time_array

    new_delta_f_hat : This value that is returned by the function is the distribution function that is obtained at
                      the final time-step. This is particularly useful in cases where comparisons need to be made 
                      between results of the Cheng-Knorr and the linear theory codes.
  
  """

  delta_f_hat = np.zeros([time_array.size, 
                          config.number_of_bands,
                          config.N_vel_y, config.N_vel_x
                         ], dtype=np.complex128
                        )

  for time_index, t0 in enumerate(time_array):
    t0 = time_array[time_index]
    if (time_index == time_array.size - 1):
        break
    t1 = time_array[time_index + 1]
    t  = [t0, t1]
    dt = t1 - t0
    if(time_index != 0):
      delta_f_hat_initial = old_delta_f_hat.copy()

    delta_f_hat[time_index, :, :, :] = delta_f_hat_initial

    new_delta_f_hat = RK4_step(config, delta_f_hat_initial, dt)
    
#    delta_rho_hat            = np.sum(new_delta_f_hat)*dv_x*dv_y
#    density_data[time_index] = np.max(delta_rho_hat.real * np.cos(k_x*x + k_y*y) - \
#                                      delta_rho_hat.imag * np.sin(k_x*x + k_y*y)
#                                     )

    old_delta_f_hat          = new_delta_f_hat.copy()


  return(delta_f_hat)

#def thermodynamic_quantities(config, delta_f_hat):
#
#  density     = np.zeros([time_array.size])
#  vel_bulk    = np.zeros([time_array.size])
#  temperature = np.zeros([time_array.size])
#  pressure    = np.zeros([time_array.size])
#
#  delta_rho_hat = np.sum(delta_f_hat, axis=[1, 2, 3])*dv_x*dv_y
#
#
#  f       = f_background(config) +
#            (delta_f_hat.real * np.cos(k_x*x + k_y*y) - \
#             delta_f_hat.imag * np.sin(k_x*x + k_y*y)
#            )
#
# dp_x = dv_x
# dp_y = dv_y
# E    = band_energy(p_x, p_y)
# density = 0
# for band in range(f.shape[0]):
#   density +=   4./(4.*np.pi)**2. 
#              * np.sum(f[band, :, :] - heaviside_theta(-E[band])) * dp_x * dp_y

