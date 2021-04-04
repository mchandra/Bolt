#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

def apply_shearing_box_bcs_f(self, boundary):
    """
    Applies the shearing box boundary conditions along boundary specified 
    for the distribution function
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """

    N_g = self.N_ghost
    q     = self.physical_system.params.q 
    omega = self.physical_system.params.omega
    
    L_q1  = self.q1_end - self.q1_start
    L_q2  = self.q2_end - self.q2_start

    if(boundary == 'left'):
        sheared_coordinates = self.q2_center[:, :, :N_g] - q * omega * L_q1 * self.time_elapsed
        
        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        self.f[:, :, :N_g] = af.reorder(af.approx2(af.reorder(self.f[:, :, :N_g], 2, 3, 0, 1),
                                                     af.reorder(self.q1_center[:, :, :N_g], 2, 3, 0, 1),
                                                     af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                     af.INTERP.BICUBIC_SPLINE,
                                                     xp = af.reorder(self.q1_center[:, :, :N_g], 2, 3, 0, 1),
                                                     yp = af.reorder(self.q2_center[:, :, :N_g], 2, 3, 0, 1)
                                                    ),
                                          2, 3, 0, 1
                                         )
        
    elif(boundary == 'right'):
        sheared_coordinates = self.q2_center[:, :, -N_g:] + q * omega * L_q1 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q2_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q2_end,
                                            sheared_coordinates - L_q2,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q2_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q2_start,
                                            sheared_coordinates + L_q2,
                                            sheared_coordinates
                                           )

        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        self.f[:, :, -N_g:] = af.reorder(af.approx2(af.reorder(self.f[:, :, -N_g:], 2, 3, 0, 1),
                                                      af.reorder(self.q1_center[:, :, -N_g:], 2, 3, 0, 1),
                                                      af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                      af.INTERP.BICUBIC_SPLINE,
                                                      xp = af.reorder(self.q1_center[:, :, -N_g:], 2, 3, 0, 1),
                                                      yp = af.reorder(self.q2_center[:, :, -N_g:], 2, 3, 0, 1)
                                                     ),
                                            2, 3, 0, 1
                                           )

    elif(boundary == 'bottom'):

        sheared_coordinates = self.q1_center[:, :, :, :N_g] - q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )

        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        self.f[:, :, :, :N_g] = af.reorder(af.approx2(af.reorder(self.f[:, :, :, :N_g], 2, 3, 0, 1),
                                                        af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                        af.reorder(self.q2_center[:, :, :, :N_g], 2, 3, 0, 1),
                                                        af.INTERP.BICUBIC_SPLINE,
                                                        xp = af.reorder(self.q1_center[:, :, :, :N_g], 2, 3, 0, 1),
                                                        yp = af.reorder(self.q2_center[:, :, :, :N_g], 2, 3, 0, 1)
                                                       ),
                                             2, 3, 0, 1
                                            )

    elif(boundary == 'top'):

        sheared_coordinates = self.q1_center[:, :, :, -N_g:] + q * omega * L_q2 * self.time_elapsed

        # Applying periodic boundary conditions to the points which are out of domain:
        while(af.sum(sheared_coordinates>self.q1_end) != 0):
            sheared_coordinates = af.select(sheared_coordinates>self.q1_end,
                                            sheared_coordinates - L_q1,
                                            sheared_coordinates
                                           )

        while(af.sum(sheared_coordinates<self.q1_start) != 0):
            sheared_coordinates = af.select(sheared_coordinates<self.q1_start,
                                            sheared_coordinates + L_q1,
                                            sheared_coordinates
                                           )
        
        # Reordering from (N_p, N_s, N_q1, N_q2) --> (N_q1, N_q2, N_p, N_s)
        # and reordering back from (N_q1, N_q2, N_p, N_s) --> (N_p, N_s, N_q1, N_q2)

        self.f[:, :, :, -N_g:] = af.reorder(af.approx2(af.reorder(self.f[:, :, :, -N_g:], 2, 3, 0, 1),
                                                         af.reorder(sheared_coordinates, 2, 3, 0, 1),
                                                         af.reorder(self.q2_center[:, :, :, -N_g:], 2, 3, 0, 1),
                                                         af.INTERP.BICUBIC_SPLINE,
                                                         xp = af.reorder(self.q1_center[:, :, :, -N_g:], 2, 3, 0, 1),
                                                         yp = af.reorder(self.q2_center[:, :, :, -N_g:], 2, 3, 0, 1)
                                                        ),
                                              2, 3, 0, 1
                                             )

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_dirichlet_bcs_f(self, boundary):
    """
    Applies Dirichlet boundary conditions along boundary specified 
    for the distribution function
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """

    N_g = self.N_ghost
    
    if(self.physical_system.params.solver_method_in_q == 'FVM'):
        velocity_q1, velocity_q2 = \
            af.broadcast(self._C_q, self.time_elapsed, 
                         self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center,
                         self.physical_system.params
                        )

    else:
        velocity_q1, velocity_q2 = \
            af.broadcast(self._A_q, self.time_elapsed, 
                         self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center,
                         self.physical_system.params
                        )

    if(velocity_q1.elements() == self.N_species * self.N_p1 * self.N_p2 * self.N_p3):
        # If velocity_q1 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, 1, Nq1, Nq2)
        velocity_q1 = af.tile(velocity_q1, 1, 1,
                              self.f.shape[2],
                              self.f.shape[3]
                             )

    if(velocity_q2.elements() == self.N_species * self.N_p1 * self.N_p2 * self.N_p3):
        # If velocity_q2 is of shape (Np1 * Np2 * Np3)
        # We tile to get it to form (Np1 * Np2 * Np3, 1, Nq1, Nq2)
        velocity_q2 = af.tile(velocity_q2, 1, 1,
                              self.f.shape[2],
                              self.f.shape[3]
                             )

    # Arguments that are passing to the called functions:
    args = (self.f, self.time_elapsed, self.q1_center, self.q2_center,
            self.p1_center, self.p2_center, self.p3_center, 
            self.physical_system.params
           )

    if(boundary == 'left'):
        f_left = self.boundary_conditions.f_left(*args)
        # Only changing inflowing characteristics:
        f_left = af.select(velocity_q1>0, f_left, self.f)
        self.f[:, :, :N_g] = f_left[:, :, :N_g]

    elif(boundary == 'right'):
        f_right = self.boundary_conditions.f_right(*args)
        # Only changing inflowing characteristics:
        f_right = af.select(velocity_q1<0, f_right, self.f)
        self.f[:, :, -N_g:] = f_right[:, :, -N_g:]

    elif(boundary == 'bottom'):
        f_bottom = self.boundary_conditions.f_bottom(*args)
        # Only changing inflowing characteristics:
        f_bottom = af.select(velocity_q2>0, f_bottom, self.f)
        self.f[:, :, :, :N_g] = f_bottom[:, :, :, :N_g]

    elif(boundary == 'top'):
        f_top = self.boundary_conditions.f_top(*args)
        # Only changing inflowing characteristics:
        f_top = af.select(velocity_q2<0, f_top, self.f)
        self.f[:, :, :, -N_g:] = f_top[:, :, :, -N_g:]

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_mirror_bcs_f_cartesian(self, boundary, mirror_start=None, mirror_end=None):
    """
    Applies mirror boundary conditions along boundary specified 
    for the distribution function when momentum space is on a cartesian grid
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """
    
    # TODO : Implement code to allow for reflections at arbitrary angles

    N_g = self.N_ghost
    dq1 = self.dq1
    dq2 = self.dq2

    if(boundary == 'left'):
        
        tmp = self.f.copy()
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        tmp[:, :, :N_g] = af.flip(tmp[:, :, N_g:2 * N_g], 2)
        
        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        if ((mirror_start != None) and (mirror_end != None)):
            #mirror_indices = (self.q2_center > mirror_start) & (self.q2_center < mirror_end)
            mirror_indices = (self.q2_center > mirror_start - dq2/50000) & (self.q2_center < mirror_end + dq2/50000)
            mirror_indices = af.tile(mirror_indices, self.N_p1*self.N_p2) 
            self.f[:, :, :N_g] = \
                mirror_indices[:, :, :N_g]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp),
                                                0
                                               )
                                       )[:, :, :N_g] + (1-mirror_indices)[:, :, :N_g]*self.f[:, :, :N_g]
        else:        
            self.f[:, :, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp),
                                                0
                                               )
                                       )[:, :, :N_g]

    elif(boundary == 'right'):
        tmp = self.f.copy()
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        tmp[:, :, -N_g:] = af.flip(tmp[:, :, -2 * N_g:-N_g], 2)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p1
        if ((mirror_start != None) and (mirror_end != None)):
            #mirror_indices = (self.q2_center > mirror_start) & (self.q2_center < mirror_end)
            mirror_indices = (self.q2_center > mirror_start - dq2/50000) & (self.q2_center < mirror_end + dq2/50000)
            mirror_indices = af.tile(mirror_indices, self.N_p1*self.N_p2) 
            self.f[:, :, -N_g:] = \
                mirror_indices[:, :, -N_g:]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp),
                                                0
                                               )
                                       )[:, :, -N_g:] + (1-mirror_indices)[:, :, -N_g:]*self.f[:, :, -N_g:]
        else : 
            self.f[:, :, -N_g:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp),
                                                0
                                               )
                                       )[:, :, -N_g:]

    elif(boundary == 'bottom'):
        tmp = self.f.copy()
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        tmp[:, :, :, :N_g] = af.flip(tmp[:, :, :, N_g:2 * N_g], 3)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        if ((mirror_start != None) and (mirror_end != None)):
            #mirror_indices = (self.q1_center > mirror_start) & (self.q1_center < mirror_end)
            mirror_indices = (self.q1_center > mirror_start - dq1/50000) & (self.q1_center < mirror_end + dq1/50000)
            mirror_indices = af.tile(mirror_indices, self.N_p1*self.N_p2) 
            self.f[:, :, :, :N_g] = \
                mirror_indices[:, :, :, :N_g]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, :N_g] + (1-mirror_indices)[:, :, :, :N_g]*self.f[:, :, :, :N_g]
        else :
            self.f[:, :, :, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, :N_g]

    elif(boundary == 'top'):
        tmp = self.f.copy()

        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        tmp[:, :, :, -N_g:] = af.flip(tmp[:, :, :, -2 * N_g:-N_g], 3)

        # The points in the ghost zone need to have direction 
        # of velocity reversed as compared to the physical zones 
        # they are mirroring. To do this we flip the axis that 
        # contains the variation in p2
        if ((mirror_start != None) and (mirror_end != None)):
            #mirror_indices = (self.q1_center > mirror_start) & (self.q1_center < mirror_end)
            mirror_indices = (self.q1_center > mirror_start - dq1/50000) & (self.q1_center < mirror_end + dq1/50000)
            mirror_indices = af.tile(mirror_indices, self.N_p1*self.N_p2) 
            self.f[:, :, :, -N_g:] = \
                mirror_indices[:, :, :, -N_g:]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, -N_g:] + (1-mirror_indices)[:, :, :, -N_g:]*self.f[:, :, :, -N_g:]
        else :
            self.f[:, :, :, -N_g:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, -N_g:]

    else:
        raise Exception('Invalid choice for boundary')

    return

def apply_mirror_bcs_f_polar2D_old(self, boundary, mirror_start = None, mirror_end = None):
    """
    Applies mirror boundary conditions along boundary specified 
    for the distribution function when momentum space is on a 2D polar grid
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """

    # TODO : This function cannot handle reflection at arbitrary angles.
    # It has been replaced by apply_mirror_bcs_f_polar_2D() and has been retained
    # for use only with the finite T polar formulation for now.

    N_g = self.N_ghost
    dq1 = self.dq1
    dq2 = self.dq2

    if(boundary == 'left'):
        tmp = self.f.copy()
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        #self.f[:, :, :N_g] = af.flip(self.f[:, :, N_g:2 * N_g], 2)
        tmp[:, :, :N_g] = af.flip(tmp[:, :, N_g:2 * N_g], 2)
        
        # For a particle moving with initial momentum at an angle \theta
        # with the x-axis, a collision with the left boundary changes
        # the angle of momentum after reflection to (pi - \theta)
        # To do this, we split the array into to equal halves,
        # flip each of the halves along the p_theta axis and then
        # join the two flipped halves together.
        
        N_theta = self.N_p2

        tmp1 = self._convert_to_p_expanded(tmp)[:, :N_theta/2, :, :]
        tmp1 = af.flip(tmp1, 1)
        tmp2 = self._convert_to_p_expanded(tmp)[:, N_theta/2:, :, :]
        tmp2 = af.flip(tmp2, 1)
        tmp3  = af.join(1, tmp1, tmp2)

        if ((mirror_start != None) and (mirror_end != None)):
            #mirror_indices = (self.q2_center > mirror_start) & (self.q2_center < mirror_end)
            mirror_indices = (self.q2_center > mirror_start - dq2/50000) & (self.q2_center < mirror_end + dq2/50000)
            mirror_indices = af.tile(mirror_indices, self.N_p1*self.N_p2) 
    
            self.f[:, :, :N_g] = \
                 mirror_indices[:, :, :N_g]*self._convert_to_q_expanded(tmp3)[:, :, :N_g] + \
                   (1 - mirror_indices)[:, :, :N_g]*self.f[:, :, :N_g]   
        
        else:        
    
            self.f[:, :, :N_g] = \
                    self._convert_to_q_expanded(tmp3)[:, :, :N_g]


    elif(boundary == 'right'):
        tmp = self.f.copy()
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        #self.f[:, :, -N_g:] = af.flip(self.f[:, :, -2 * N_g:-N_g], 2)
        tmp[:, :, -N_g:] = af.flip(tmp[:, :, -2 * N_g:-N_g], 2)

        # For a particle moving with initial momentum at an angle \theta
        # with the x-axis, a collision with the right boundary changes
        # the angle of momentum after reflection to (pi - \theta)
        # To do this, we split the array into to equal halves,
        # flip each of the halves along the p_theta axis and then
        # join the two flipped halves together.

        print("boundaries.py, right bc, rank, mirror_start, mirror_end :", self.physical_system.params.rank, mirror_start, mirror_end)

        N_theta = self.N_p2

        tmp1 = self._convert_to_p_expanded(tmp)[:, :N_theta/2, :, :]
        tmp1 = af.flip(tmp1, 1)
        tmp2 = self._convert_to_p_expanded(tmp)[:, N_theta/2:, :, :]
        tmp2 = af.flip(tmp2, 1)
        tmp3 = af.join(1, tmp1, tmp2)

        if ((mirror_start != None) and (mirror_end != None)):
            #mirror_indices = (self.q2_center > mirror_start) & (self.q2_center < mirror_end)
            mirror_indices = (self.q2_center > mirror_start - dq2/50000) & (self.q2_center < mirror_end + dq2/50000)
            mirror_indices = af.tile(mirror_indices, self.N_p1*self.N_p2) 
            
            self.f[:, :, -N_g:] = \
                mirror_indices[:, :, -N_g:]*self._convert_to_q_expanded(tmp3)[:, :, -N_g:] + \
                    (1-mirror_indices)[:, :, -N_g:]*self.f[:, :, -N_g:]

        else : 
            self.f[:, :, -N_g:] = \
                    self._convert_to_q_expanded(tmp3)[:, :, -N_g:]


    elif(boundary == 'bottom'):
        tmp = self.f.copy()
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        #self.f[:, :, :, :N_g] = af.flip(self.f[:, :, :, N_g:2 * N_g], 3)
        tmp[:, :, :, :N_g] = af.flip(tmp[:, :, :, N_g:2 * N_g], 3)

        # For a particle moving with initial momentum at an angle \theta
        # with the x-axis, a collision with the bottom boundary changes
        # the angle of momentum after reflection to (2*pi - \theta) = (-\theta)
        # To do this we flip the axis that contains the variation in p_theta
        if ((mirror_start != None) and (mirror_end != None)):
            #mirror_indices = (self.q1_center > mirror_start) & (self.q1_center < mirror_end)
            mirror_indices = (self.q1_center > mirror_start - dq1/50000) & (self.q1_center < mirror_end + dq1/50000)
            mirror_indices = af.tile(mirror_indices, self.N_p1*self.N_p2) 
            self.f[:, :, :, :N_g] = \
                mirror_indices[:, :, :, :N_g]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, :N_g] + (1-mirror_indices)[:, :, :, :N_g]*self.f[:, :, :, :N_g]
        else :
            self.f[:, :, :, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, :N_g]

    elif(boundary == 'top'):
        tmp = self.f.copy()
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        #self.f[:, :, :, -N_g:] = af.flip(self.f[:, :, :, -2 * N_g:-N_g], 3)
        tmp[:, :, :, -N_g:] = af.flip(tmp[:, :, :, -2 * N_g:-N_g], 3)

        # For a particle moving with initial momentum at an angle \theta
        # with the x-axis, a collision with the top boundary changes
        # the angle of momentum after reflection to (2*pi - \theta) = (-\theta)
        # To do this we flip the axis that contains the variation in p_theta
        if ((mirror_start != None) and (mirror_end != None)):
            #mirror_indices = (self.q1_center > mirror_start) & (self.q1_center < mirror_end)
            mirror_indices = (self.q1_center > mirror_start - dq1/50000) & (self.q1_center < mirror_end + dq1/50000)
            mirror_indices = af.tile(mirror_indices, self.N_p1*self.N_p2) 
            self.f[:, :, :, -N_g:] = \
                mirror_indices[:, :, :, -N_g:]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, -N_g:] + (1-mirror_indices)[:, :, :, -N_g:]*self.f[:, :, :, -N_g:]
        else :
            self.f[:, :, :, -N_g:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, -N_g:]

    else:
        raise Exception('Invalid choice for boundary')

    return


def apply_mirror_bcs_f_polar2D(self, boundary, mirror_start=None, mirror_end=None):
    """
    Applies mirror boundary conditions along boundary specified 
    for the distribution function when momentum space is on a 2D polar grid
    
    Parameters
    ----------
    boundary: str
              Boundary along which the boundary condition is to be applied.
    """
    # theta_p = incident angle of particle wrt positive x-axis
    # theta   = angle of the mirror boundary wrt positive x-axis

    # In polar coordinates, the reflection of a particle moving at an angle
    # theta_p wrt the positive x-axis off a boundary which is at an angle
    # theta wrt the positive x-axis results in the reflected particle moving
    # at an angle 2*theta - theta_p wrt the positive x-axis.
    
    # Operation to be performed : 2*theta - theta_p
    # We split this operation into 2 steps as shown below.

    # Operation 1 : theta_prime = theta_p - 2*theta
    # To do this, we shift the array along the axis that contains the variation in p_theta

    # Operation 2 : theta_out = -theta_prime
    # To do this we flip the axis that contains the variation in p_theta
    
    N_g        = self.N_ghost
    N_q1_local = self.f.dims()[2]
    N_q2_local = self.f.dims()[3]
    N_theta    = self.N_p2

    if(boundary == 'left'):

        tmp = self.f.copy()
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        tmp[:, :, :N_g] = af.flip(tmp[:, :, N_g:2 * N_g], 2)
        
        # Operation 1 : theta_prime = theta_p - 2*theta
        shift_indices = self.physical_system.params.shift_indices_left

        left_edge = 0

        for index in range(N_g): # For each ghost zone
            f_2D_flattened             = af.moddims(tmp[:, 0, left_edge+index, :], N_theta*N_q2_local)
            f_shifted_flattened        = f_2D_flattened[shift_indices]
            f_shifted                  = af.moddims(f_shifted_flattened, N_theta, 1, 1, N_q2_local)
            tmp[:, 0, left_edge+index, :] = f_shifted
        
        # Operation 2 : theta_out = -theta_prime
        if ((mirror_start != None) and (mirror_end != None)):
            mirror_indices = (self.q2_center > mirror_start) & (self.q2_center < mirror_end)
            mirror_indices = af.tile(mirror_indices, self.N_p2) 
            self.f[:, :, :N_g] = \
                mirror_indices[:, :, :N_g]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp),
                                                1
                                               )
                                       )[:, :, :N_g] + (1-mirror_indices)[:, :, :N_g]*self.f[:, :, :N_g]
        else:        
            self.f[:, :, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp),
                                                1
                                               )
                                       )[:, :, :N_g]
        
    elif(boundary == 'right'):
    
        tmp = self.f.copy()
        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        tmp[:, :, -N_g:] = af.flip(tmp[:, :, -2 * N_g:-N_g], 2)

        # Operation 1 : theta_prime = theta_p - 2*theta
        shift_indices = self.physical_system.params.shift_indices_right

        right_edge = -1

        for index in range(N_g): # For each ghost zone
            f_2D_flattened              = af.moddims(tmp[:, 0, right_edge-index, :], N_theta*N_q2_local)
            f_shifted_flattened         = f_2D_flattened[shift_indices]
            f_shifted                   = af.moddims(f_shifted_flattened, N_theta, 1, 1, N_q2_local)
            tmp[:, 0, right_edge-index, :] = f_shifted
        
        # Operation 2 : theta_out = -theta_prime
        if ((mirror_start != None) and (mirror_end != None)):
            mirror_indices = (self.q2_center > mirror_start) & (self.q2_center < mirror_end)
            mirror_indices = af.tile(mirror_indices, self.N_p2) 
            self.f[:, :, -N_g:] = \
                mirror_indices[:, :, -N_g:]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp),
                                                1
                                               )
                                       )[:, :, -N_g:] + (1-mirror_indices)[:, :, -N_g:]*self.f[:, :, -N_g:]
        else : 
            self.f[:, :, -N_g:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp),
                                                1
                                               )
                                       )[:, :, -N_g:]

    elif(boundary == 'bottom'):

        tmp = self.f.copy()
        # x-0-x-0-x-0-|-0-x-0-x-0-x-....
        #   0   1   2   3   4   5
        # For mirror boundary conditions:
        # 0 = 5; 1 = 4; 2 = 3;
        tmp[:, :, :, :N_g] = af.flip(tmp[:, :, :, N_g:2 * N_g], 3)
    
        # Operation 1 : theta_prime = theta_p - 2*theta
        shift_indices = self.physical_system.params.shift_indices_bottom

        bottom_edge = 0
    
        for index in range(N_g): # For each ghost zone
            f_2D_flattened               = af.moddims(tmp[:, 0, :, bottom_edge+index], N_theta*N_q1_local)
            f_shifted_flattened          = f_2D_flattened[shift_indices]
            f_shifted                    = af.moddims(f_shifted_flattened, N_theta, 1, N_q1_local, 1)
            tmp[:, 0, :, bottom_edge+index] = f_shifted
        
        # Operation 2 : theta_out = -theta_prime
        if ((mirror_start != None) and (mirror_end != None)):
            mirror_indices = (self.q1_center > mirror_start) & (self.q1_center < mirror_end)
            mirror_indices = af.tile(mirror_indices, self.N_p2) 
            self.f[:, :, :, :N_g] = \
                mirror_indices[:, :, :, :N_g]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, :N_g] + (1-mirror_indices)[:, :, :, :N_g]*self.f[:, :, :, :N_g]
        else :
            self.f[:, :, :, :N_g] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, :N_g]

    elif(boundary == 'top'):
        tmp = self.f.copy()

        # ...-x-0-x-0-x-0-|-0-x-0-x-0-x
        #      -6  -5  -4  -3  -2  -1
        # For mirror boundary conditions:
        # -1 = -6; -2 = -5; -3 = -4;
        tmp[:, :, :, -N_g:] = af.flip(tmp[:, :, :, -2 * N_g:-N_g], 3)
        
        # Operation 1 : theta_prime = theta_p - 2*theta
        shift_indices = self.physical_system.params.shift_indices_top

        top_edge = -1
        
        for index in range(N_g): # For each ghost zone
            f_2D_flattened             = af.moddims(tmp[:, 0, :, top_edge-index], N_theta*N_q1_local)
            f_shifted_flattened        = f_2D_flattened[shift_indices]
            f_shifted                  = af.moddims(f_shifted_flattened, N_theta, 1, N_q1_local, 1)
            tmp[:, 0, :, top_edge-index]  = f_shifted
        
        # Operation 2 : theta_out = -theta_prime
        if ((mirror_start != None) and (mirror_end != None)):
            mirror_indices = (self.q1_center > mirror_start) & (self.q1_center < mirror_end)
            mirror_indices = af.tile(mirror_indices, self.N_p2) 
            self.f[:, :, :, -N_g:] = \
                mirror_indices[:, :, :, -N_g:]*self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, -N_g:] + (1-mirror_indices)[:, :, :, -N_g:]*self.f[:, :, :, -N_g:]
        else :
            self.f[:, :, :, -N_g:] = \
                self._convert_to_q_expanded(af.flip(self._convert_to_p_expanded(tmp), 
                                                1
                                               )
                                       )[:, :, :, -N_g:]


    else:
        raise Exception('Invalid choice for boundary')

    return


def apply_bcs_f(self):
    """
    Applies boundary conditions to the distribution function as specified by 
    the user in params.
    """


    if(self.performance_test_flag == True):
        tic = af.time()

    # Obtaining start coordinates for the local zone
    # Additionally, we also obtain the size of the local zone
    ((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = self._da_f.getCorners()
    # Obtaining the end coordinates for the local zone
    (i_q1_end, i_q2_end) = (i_q1_start + N_q1_local - 1, i_q2_start + N_q2_local - 1)

    # If local zone includes the left physical boundary:
    if(i_q1_start == self.physical_system.params.left_dirichlet_boundary_index) and \
            (self.physical_system.params.rank not in self.physical_system.params.dont_apply_left_bc):

        if(self.boundary_conditions.in_q1_left == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'left')

        elif(self.boundary_conditions.in_q1_left == 'mirror'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'left')            
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if (self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'left') # TODO: Replace by apply_mirror_bcs_polar2D()
                elif (self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'left')
            else :
                raise NotImplementedError('Unsupported coordinate system in p_space')

        elif(self.boundary_conditions.in_q1_left == 'mirror+dirichlet'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'left')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if (self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'left') # TODO: Replace by apply_mirror_bcs_polar2D()
                elif (self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'left')
            else :
                raise NotImplementedError('Unsupported coordinate system in p_space')
            apply_dirichlet_bcs_f(self, 'left')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(   self.boundary_conditions.in_q1_left == 'periodic'
             or self.boundary_conditions.in_q1_left == 'none' # no ghost zones (1D)
            ):
            pass

        elif(self.boundary_conditions.in_q1_left == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'left')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')


#    # TODO : Testing!
#    # If local zone includes the second left physical boundary:
#    if(i_q1_start == self.physical_system.params.left_dirichlet_boundary_index_2) and \
#            (self.physical_system.params.rank not in self.physical_system.params.dont_apply_left_bc):
#
#        if(self.boundary_conditions.in_q1_left == 'dirichlet'):
#            apply_dirichlet_bcs_f(self, 'left')
#
#        elif(self.boundary_conditions.in_q1_left == 'mirror'):
#            if (self.physical_system.params.p_space_grid == 'cartesian'):
#                apply_mirror_bcs_f_cartesian(self, 'left')            
#            elif (self.physical_system.params.p_space_grid == 'polar2D'):
#                if (self.physical_system.params.p_dim == 2):
#                    apply_mirror_bcs_f_polar2D_old(self, 'left') # TODO: Replace by apply_mirror_bcs_polar2D()
#                elif (self.physical_system.params.p_dim == 1):
#                    apply_mirror_bcs_f_polar2D(self, 'left')
#            else :
#                raise NotImplementedError('Unsupported coordinate system in p_space')
#
#        elif(self.boundary_conditions.in_q1_left == 'mirror+dirichlet'):
#            if (self.physical_system.params.p_space_grid == 'cartesian'):
#                apply_mirror_bcs_f_cartesian(self, 'left')
#            elif (self.physical_system.params.p_space_grid == 'polar2D'):
#                if (self.physical_system.params.p_dim == 2):
#                    apply_mirror_bcs_f_polar2D_old(self, 'left') # TODO: Replace by apply_mirror_bcs_polar2D()
#                elif (self.physical_system.params.p_dim == 1):
#                    apply_mirror_bcs_f_polar2D(self, 'left')
#            else :
#                raise NotImplementedError('Unsupported coordinate system in p_space')
#            apply_dirichlet_bcs_f(self, 'left')
#        
#        # This is automatically handled by the PETSc function globalToLocal()
#        elif(   self.boundary_conditions.in_q1_left == 'periodic'
#             or self.boundary_conditions.in_q1_left == 'none' # no ghost zones (1D)
#            ):
#            pass
#
#        elif(self.boundary_conditions.in_q1_left == 'shearing-box'):
#            apply_shearing_box_bcs_f(self, 'left')
#
#        else:
#            raise NotImplementedError('Unavailable/Invalid boundary condition')



    # If local zone includes the right physical boundary:
    if(i_q1_end == self.physical_system.params.right_dirichlet_boundary_index) and \
            (self.physical_system.params.rank not in self.physical_system.params.dont_apply_right_bc):

        if(self.boundary_conditions.in_q1_right == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'right')

        elif(self.boundary_conditions.in_q1_right == 'mirror'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'right')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if (self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'right') # TODO: Replace by apply_mirror_bcs_polar2D()
                elif (self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'right')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
        
        elif(self.boundary_conditions.in_q1_right == 'mirror+dirichlet'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'right')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if (self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'right') # TODO: Replace by apply_mirror_bcs_polar2D()
                elif (self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'right')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
            apply_dirichlet_bcs_f(self, 'right')

        # This is automatically handled by the PETSc function globalToLocal()
        elif(   self.boundary_conditions.in_q1_right == 'periodic'
             or self.boundary_conditions.in_q1_right == 'none' # no ghost zones (1D)
            ):
            pass

        elif(self.boundary_conditions.in_q1_right == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'right')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the bottom physical boundary:
    if(i_q2_start == self.physical_system.params.bottom_dirichlet_boundary_index) and \
            (self.physical_system.params.rank not in self.physical_system.params.dont_apply_bottom_bc):

        if(self.boundary_conditions.in_q2_bottom == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'bottom')

        elif(self.boundary_conditions.in_q2_bottom == 'mirror'):
            if (self.physical_system.params.p_space_grid =='cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'bottom')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if (self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'bottom') # TODO: Replace by apply_mirror_bcs_polar2D()
                elif (self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'bottom')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')

        elif(self.boundary_conditions.in_q2_bottom == 'mirror+dirichlet'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'bottom')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if (self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'bottom') # TODO: Replace by apply_mirror_bcs_polar2D()
                elif (self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'bottom')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
            apply_dirichlet_bcs_f(self, 'bottom')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(   self.boundary_conditions.in_q2_bottom == 'periodic'
             or self.boundary_conditions.in_q2_bottom == 'none' # no ghost zones (1D)
            ):
            pass

        elif(self.boundary_conditions.in_q2_bottom == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'bottom')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # If local zone includes the top physical boundary:
    if(i_q2_end == self.physical_system.params.top_dirichlet_boundary_index) and \
            (self.physical_system.params.rank not in self.physical_system.params.dont_apply_top_bc):

        if(self.boundary_conditions.in_q2_top == 'dirichlet'):
            apply_dirichlet_bcs_f(self, 'top')

        elif(self.boundary_conditions.in_q2_top == 'mirror'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'top')
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if (self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'top') # TODO: Replace by apply_mirror_bcs_polar2D()
                elif (self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'top')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
        
        elif(self.boundary_conditions.in_q2_top == 'mirror+dirichlet'):
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'top')            
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if (self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'top') # TODO: Replace by apply_mirror_bcs_polar2D()
                elif (self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'top')
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
            apply_dirichlet_bcs_f(self, 'top')
        
        # This is automatically handled by the PETSc function globalToLocal()
        elif(   self.boundary_conditions.in_q2_top == 'periodic'
             or self.boundary_conditions.in_q2_top == 'none' # no ghost zones (1D)
            ):
            pass

        elif(self.boundary_conditions.in_q2_top == 'shearing-box'):
            apply_shearing_box_bcs_f(self, 'top')

        else:
            raise NotImplementedError('Unavailable/Invalid boundary condition')

    # Handle blocked boundaries in specified domains
    if self.physical_system.params.rank in self.physical_system.params.blocked_left_bc:
        if (self.physical_system.params.p_space_grid == 'cartesian'):
            apply_mirror_bcs_f_cartesian(self, 'left')            
        elif (self.physical_system.params.p_space_grid == 'polar2D'):
            apply_mirror_bcs_f_polar2D(self, 'left')
        else :
            raise NotImplementedError('Unsupported coordinate system in p_space')

    if self.physical_system.params.rank in self.physical_system.params.blocked_right_bc:
        if (self.physical_system.params.p_space_grid == 'cartesian'):
            apply_mirror_bcs_f_cartesian(self, 'right')            
        elif (self.physical_system.params.p_space_grid == 'polar2D'):
            apply_mirror_bcs_f_polar2D(self, 'right')
        else :
            raise NotImplementedError('Unsupported coordinate system in p_space')

    if self.physical_system.params.rank in self.physical_system.params.blocked_bottom_bc:
        if (self.physical_system.params.p_space_grid == 'cartesian'):
            apply_mirror_bcs_f_cartesian(self, 'bottom')            
        elif (self.physical_system.params.p_space_grid == 'polar2D'):
            apply_mirror_bcs_f_polar2D(self, 'bottom')
        else :
            raise NotImplementedError('Unsupported coordinate system in p_space')

    if self.physical_system.params.rank in self.physical_system.params.blocked_top_bc:
        if (self.physical_system.params.p_space_grid == 'cartesian'):
            apply_mirror_bcs_f_cartesian(self, 'top')            
        elif (self.physical_system.params.p_space_grid == 'polar2D'):
            apply_mirror_bcs_f_polar2D(self, 'top')
        else :
            raise NotImplementedError('Unsupported coordinate system in p_space')

    horizontal_boundaries    = self.physical_system.params.horizontal_boundaries
    horizontal_boundary_lims = self.physical_system.params.horizontal_boundary_lims
    for index in range(len(horizontal_boundaries)):
        # If local zone includes the internal mirror boundary at the bottom:
        if(i_q2_start == int(horizontal_boundaries[index])):
    
            if (self.physical_system.params.p_space_grid =='cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'bottom',
                    mirror_start = horizontal_boundary_lims[index][0],
                    mirror_end   = horizontal_boundary_lims[index][1])
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if(self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'bottom',
                         mirror_start = horizontal_boundary_lims[index][0],
                         mirror_end   = horizontal_boundary_lims[index][1])
                elif(self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'bottom',
                         mirror_start = horizontal_boundary_lims[index][0],
                         mirror_end   = horizontal_boundary_lims[index][1])
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')

    for index in range(len(horizontal_boundaries)):
        # If local zone includes the internal mirror boundary at the top:
        if(i_q2_end == int(horizontal_boundaries[index]) - 1):
    
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'top',
                    mirror_start = horizontal_boundary_lims[index][0],
                    mirror_end   = horizontal_boundary_lims[index][1])
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if(self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'top',
                         mirror_start = horizontal_boundary_lims[index][0],
                         mirror_end   = horizontal_boundary_lims[index][1])
                elif(self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'top',
                         mirror_start = horizontal_boundary_lims[index][0],
                         mirror_end   = horizontal_boundary_lims[index][1])
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')

    vertical_boundaries    = self.physical_system.params.vertical_boundaries
    vertical_boundary_lims = self.physical_system.params.vertical_boundary_lims
    for index in range(len(vertical_boundaries)):
        # If local zone includes the internal mirror boundary at the left:
        if(i_q1_start == int(vertical_boundaries[index])):
    
            if (self.physical_system.params.p_space_grid =='cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'left',
                     mirror_start = vertical_boundary_lims[index][0],
                     mirror_end   = vertical_boundary_lims[index][1])
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if(self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'left',
                         mirror_start = vertical_boundary_lims[index][0],
                         mirror_end   = vertical_boundary_lims[index][1])
                elif(self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'left',
                         mirror_start = vertical_boundary_lims[index][0],
                         mirror_end   = vertical_boundary_lims[index][1])
            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')

    for index in range(len(vertical_boundaries)):
        # If local zone includes the internal mirror boundary at the right:
        if(i_q1_end == int(vertical_boundaries[index]) - 1):
    
            if (self.physical_system.params.p_space_grid == 'cartesian'):
                apply_mirror_bcs_f_cartesian(self, 'right',
                     mirror_start = vertical_boundary_lims[index][0],
                     mirror_end   = vertical_boundary_lims[index][1])
            elif (self.physical_system.params.p_space_grid == 'polar2D'):
                if(self.physical_system.params.p_dim == 1):
                    apply_mirror_bcs_f_polar2D(self, 'right',
                         mirror_start = vertical_boundary_lims[index][0],
                         mirror_end   = vertical_boundary_lims[index][1])
                elif(self.physical_system.params.p_dim == 2):
                    apply_mirror_bcs_f_polar2D_old(self, 'right',
                         mirror_start = vertical_boundary_lims[index][0],
                         mirror_end   = vertical_boundary_lims[index][1])

            else:
                raise NotImplementedError('Unsupported coordinate system in p_space')
        

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_apply_bcs_f += toc - tic
   
    return

