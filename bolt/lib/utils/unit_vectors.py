import arrayfire as af
import numpy as np


def normal_to_circle_unit_vec(theta):

    vel_x = 0.*theta
    vel_y = 0.*theta

    # (1)
    start_theta = 0.0
    end_theta   = np.pi/3
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*af.cos(theta)
    vel_y = (1-indices)*vel_y + indices*af.sin(theta)
    #(1-indices) is equivalent to complementing indices
    

    # (2)    
    start_theta = np.pi/3
    end_theta   = 2*np.pi/3
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*af.cos(theta)
    vel_y = (1-indices)*vel_y + indices*af.sin(theta)

    # (3)    
    start_theta = 2*np.pi/3
    end_theta   = np.pi
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*af.cos(theta)
    vel_y = (1-indices)*vel_y + indices*af.sin(theta)
    
    # (4)
    start_theta = -np.pi
    end_theta   = -2*np.pi/3
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*af.cos(theta)
    vel_y = (1-indices)*vel_y + indices*af.sin(theta)
    
    # (5)
    start_theta = -2*np.pi/3
    end_theta   = -np.pi/3
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*af.cos(theta)
    vel_y = (1-indices)*vel_y + indices*af.sin(theta)

    # (6)
    start_theta = -np.pi/3
    end_theta   = 0
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*af.cos(theta)
    vel_y = (1-indices)*vel_y + indices*af.sin(theta)

    return ([vel_x, vel_y]) 


def normal_to_hexagon_unit_vec(theta):

    vel_x = 0.*theta
    vel_y = 0.*theta

#
#           2*pi/3                 pi/3
#              #%%%%%%%%%%%%%%%%%%%*         
#             .  .               ,  *        
#           (     &     (2)     #     .      
#          .       .          .        (     
#        /           &       #               
#       ,     (3)      ,    .    (1)      (  
#  pi /                 & &                  
#     /------------------/------------------(    0
# -pi  .               #   #              .  
#        /    (4)            /     (6)    #   
#         *         %         #              
#           *      ,           #      #      
#            *   (      (5)      /           
#              ,,                 %(         
#              #####################
#         -2*pi/3                 -pi/3
#

    # (1)
    start_theta = 0.0
    end_theta   = np.pi/3
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*np.cos(mid_theta)
    vel_y = (1-indices)*vel_y + indices*np.sin(mid_theta)

    # (2)    
    start_theta = np.pi/3
    end_theta   = 2*np.pi/3
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*np.cos(mid_theta)
    vel_y = (1-indices)*vel_y + indices*np.sin(mid_theta)

    # (3)    
    start_theta = 2*np.pi/3
    end_theta   = np.pi
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*np.cos(mid_theta)
    vel_y = (1-indices)*vel_y + indices*np.sin(mid_theta)
    
    # (4)
    start_theta = -np.pi
    end_theta   = -2*np.pi/3
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*np.cos(mid_theta)
    vel_y = (1-indices)*vel_y + indices*np.sin(mid_theta)
    
    # (5)
    start_theta = -2*np.pi/3
    end_theta   = -np.pi/3
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*np.cos(mid_theta)
    vel_y = (1-indices)*vel_y + indices*np.sin(mid_theta)

    # (6)
    start_theta = -np.pi/3
    end_theta   = 0
    mid_theta   = 0.5*(end_theta + start_theta)
    indices = ((theta >= start_theta) & (theta < end_theta))
    vel_x = (1-indices)*vel_x + indices*np.cos(mid_theta)
    vel_y = (1-indices)*vel_y + indices*np.sin(mid_theta)

    return ([vel_x, vel_y]) 
