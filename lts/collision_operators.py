import numpy as np

def RTA_defect_scattering(config, delta_f_hat):
  
  E    = config.band_energies
  mu_0 = config.chemical_potential_background
  k    = config.boltzmann_constant
  T_0  = config.temperature_background

  tmp = (E - mu_0)/(k*T_0)

  denominator =  (k*T_0**2.) \
               * (np.exp(tmp) + 2. + np.exp(-tmp) )

  a_00_integrand =     T_0        / denominator
  a_01_integrand =     (E - mu_0) / denominator
  a_10_integrand = E * T_0        / denominator
  a_11_integrand = E * (E - mu_0) / denominator

  b_0_integrand  =     delta_f_hat
  b_1_integrand  = E * delta_f_hat

  a_00 = np.sum(a_00_integrand) * config.dp_x * config.dp_y
  a_01 = np.sum(a_01_integrand) * config.dp_x * config.dp_y
  a_10 = np.sum(a_10_integrand) * config.dp_x * config.dp_y
  a_11 = np.sum(a_11_integrand) * config.dp_x * config.dp_y

  b_0  = np.sum(b_0_integrand)  * config.dp_x * config.dp_y
  b_1  = np.sum(b_1_integrand)  * config.dp_x * config.dp_y

  determinant       = (a_01*a_10 - a_00*a_11)
  delta_mu_hat_eqbm = -(a_11*b_0 - a_01*b_1) / determinant
  delta_T_hat_eqbm  = -(a_10*b_0 - a_00*b_1) / determinant

  numerator   = (E - mu_0)*delta_T_hat_eqbm + T_0*delta_mu_hat_eqbm

  delta_f_hat_local_eqbm_linearized = numerator/denominator

  C_f = -(delta_f_hat - delta_f_hat_local_eqbm_linearized)/config.tau_defect

  return C_f

def RTA_ee_scattering(config, delta_f_hat):

  E    = config.band_energies
  mu_0 = config.chemical_potential_background
  k    = config.boltzmann_constant
  T_0  = config.temperature_background

  tmp = (E - mu_0)/(k*T_0)

  denominator =  (k*T_0**2.) \
               * (np.exp(tmp) + 2. + np.exp(-tmp) )

  a = 1./denominator
  
  a_0 = T_0              * a
  a_1 = (E - mu_0)       * a
  a_2 = T_0 * config.p_x * a  
  a_3 = T_0 * config.p_y * a

  a_00 = np.sum(a_0) * config.dp_x * config.dp_y
  a_01 = np.sum(a_1) * config.dp_x * config.dp_y
  a_02 = np.sum(a_2) * config.dp_x * config.dp_y
  a_03 = np.sum(a_3) * config.dp_x * config.dp_y

  a_10 = np.sum(E * a_0) * config.dp_x * config.dp_y
  a_11 = np.sum(E * a_1) * config.dp_x * config.dp_y
  a_12 = np.sum(E * a_2) * config.dp_x * config.dp_y
  a_13 = np.sum(E * a_3) * config.dp_x * config.dp_y

  a_20 = np.sum(config.p_x * a_0) * config.dp_x * config.dp_y
  a_21 = np.sum(config.p_x * a_1) * config.dp_x * config.dp_y
  a_22 = np.sum(config.p_x * a_2) * config.dp_x * config.dp_y
  a_23 = np.sum(config.p_x * a_3) * config.dp_x * config.dp_y

  a_30 = np.sum(config.p_y * a_0) * config.dp_x * config.dp_y
  a_31 = np.sum(config.p_y * a_1) * config.dp_x * config.dp_y
  a_32 = np.sum(config.p_y * a_2) * config.dp_x * config.dp_y
  a_33 = np.sum(config.p_y * a_3) * config.dp_x * config.dp_y

  b_0  = np.sum(             delta_f_hat) * config.dp_x * config.dp_y
  b_1  = np.sum(E          * delta_f_hat) * config.dp_x * config.dp_y
  b_2  = np.sum(config.p_x * delta_f_hat) * config.dp_x * config.dp_y
  b_3  = np.sum(config.p_y * delta_f_hat) * config.dp_x * config.dp_y

  A = np.array([ [a_00, a_01, a_02, a_03], \
                 [a_10, a_11, a_12, a_13], \
                 [a_20, a_21, a_22, a_23], \
                 [a_30, a_31, a_32, a_33]  \
               ] \
              )
  b = np.array([b_0, b_1, b_2, b_3])

  delta_mu_hat, delta_T_hat, delta_vx_hat, delta_vy_hat = np.linalg.solve(A, b)

#  print "delta_mu_hat = ", delta_mu_hat, " delta_T_hat = ", delta_T_hat, \
#    "delta_vx_hat = ", delta_vx_hat, " delta_vy_hat = ", delta_vy_hat
  delta_f_hat_local_eqbm_linearized = \
                              (  T_0 * delta_vx_hat * config.p_x \
                               + T_0 * delta_vy_hat * config.p_y \
                               + (E - mu_0) * delta_T_hat \
                               + T_0 * delta_mu_hat \
                              ) / denominator

  C_f = -(delta_f_hat - delta_f_hat_local_eqbm_linearized
         ) / config.tau_ee

  return C_f 


def BGK_collision_operator(config, delta_f_hat):
  """
  Returns the array that contains the values of the linearized BGK collision operator.
  The expression that has been used may be understood more clearly by referring to the
  Sage worksheet on https://goo.gl/dXarsP

  Parameters:
  -----------

    An object args is passed to this function of which the following attributes
    are utilized:

    config : Object config which is obtained by set() is passed to this function

    delta_f_hat : The array of delta_f_hat which is obtained from each step
                  of the time integration. 

  Output:
  -------
    C_f : Array which contains the values of the linearized collision operator. 

  """
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x
  vel_x     = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  dv_x      = vel_x[1] - vel_x[0]

  if(config.mode =='2D2V'): 
    vel_y_max = config.vel_y_max
    N_vel_y   = config.N_vel_y
    vel_y     = np.linspace(-vel_y_max, vel_y_max, N_vel_y)
    dv_y      = vel_y[1] - vel_y[0]

    vel_x, vel_y = np.meshgrid(vel_x, vel_y)

  tau   = config.tau

  if(config.mode == '2D2V'):
    delta_rho_hat = np.sum(delta_f_hat) * dv_x * dv_y
    delta_T_hat   = np.sum(delta_f_hat * (0.5*(vel_x**2 + vel_y**2) -\
                                          temperature_background
                                          )
                          ) * dv_x * dv_y/rho_background
    delta_v_x_hat = np.sum(delta_f_hat * vel_x) * dv_x * dv_y/rho_background
    delta_v_y_hat = np.sum(delta_f_hat * vel_y) * dv_x * dv_y/rho_background
  
    expr_term_1 = delta_T_hat * mass_particle**2 * rho_background * vel_x**2
    expr_term_2 = delta_T_hat * mass_particle**2 * rho_background * vel_y
    expr_term_3 = 2 * temperature_background**2 * delta_rho_hat * boltzmann_constant * mass_particle
    expr_term_4 = 2 * (delta_v_x_hat * mass_particle**2 * rho_background*vel_x +\
                       delta_v_y_hat * mass_particle**2 * rho_background *vel_y -\
                       delta_T_hat * boltzmann_constant * mass_particle *rho_background
                      )*temperature_background
    
    C_f = ((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4)/(4*np.pi*boltzmann_constant**2*temperature_background**3)*\
          np.exp(-mass_particle/(2*boltzmann_constant*temperature_background) * (vel_x**2 + vel_y**2)) - delta_f_hat)/tau
  
  elif(config.mode == '1D1V'):
    delta_T_hat   = np.sum(delta_f_hat * (vel_x**2 - temperature_background)) * dv_x/rho_background
    delta_rho_hat = np.sum(delta_f_hat) * dv_x
    delta_v_x_hat = np.sum(delta_f_hat * vel_x) * dv_x/rho_background
    
    expr_term_1 = np.sqrt(2 * mass_particle**3) * delta_T_hat * rho_background * vel_x**2
    expr_term_2 = 2 * np.sqrt(2 * mass_particle) * boltzmann_constant * delta_rho_hat * temperature_background**2
    expr_term_3 = 2 * np.sqrt(2 * mass_particle**3) * rho_background * delta_v_x_hat * vel_x * temperature_background
    expr_term_4 = - np.sqrt(2 * mass_particle) * boltzmann_constant * delta_T_hat * rho_background * temperature_background
    
    C_f = (((expr_term_1 + expr_term_2 + expr_term_3 + expr_term_4)*\
           np.exp(-mass_particle * vel_x**2/(2 * boltzmann_constant * temperature_background))/\
           (4 * np.sqrt(np.pi * temperature_background**5 * boltzmann_constant**3)) - delta_f_hat
           )/tau
          )
  
  else:
    raise Exception('The mode mentioned in the config file is not supported')

  return C_f
