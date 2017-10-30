#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np

def f_interp_2d(self, dt):
    
    if(self.performance_test_flag == True):
        tic = af.time()

    # Defining a lambda function to perform broadcasting operations
    # This is done using af.broadcast, which allows us to perform 
    # batched operations when operating on arrays of different sizes
    addition = lambda a, b:a + b

    # af.broadcast(function, *args) performs batched operations on
    # function(*args)
    q1_center_new = af.broadcast(addition, self.q1_center, - self._A_q1 * dt)
    q2_center_new = af.broadcast(addition, self.q2_center, - self._A_q2 * dt)

    # Reordering from (dof, N_q1, N_q2) --> (N_q1, N_q2, dof)
    self.f = af.approx2(af.reorder(self.f, 1, 2, 0),
                        af.reorder(q1_center_new, 1, 2, 0),
                        af.reorder(q2_center_new, 1, 2, 0),
                        af.INTERP.BILINEAR, 
                        xp = af.reorder(self.q1_center, 1, 2, 0),
                        yp = af.reorder(self.q2_center, 1, 2, 0)
                       )

    # Reordering from (N_q1, N_q2, dof) --> (dof, N_q1, N_q2)
    self.f = af.reorder(self.f, 2, 0, 1)

    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_interp2 += toc - tic

    return

def f_interp_p_3d(self, dt):
    """
    Since the interpolation function are being performed in velocity space,
    the arrays used in the computation need to be in p_expanded form.
    Hence we will need to convert the same:
    """
    # Following Strang Splitting:
    # af.broadcast, allows us to perform batched operations 
    # when operating on arrays of different sizes
    # af.broadcast(function, *args) performs batched operations on
    # function(*args)

    if(self.performance_test_flag == True):
        tic = af.time()
    
    E1 = self.cell_centered_EM_fields[0]
    E2 = self.cell_centered_EM_fields[1]
    E3 = self.cell_centered_EM_fields[2]

    B1_n = self.B_fields_at_nth_timestep[0]
    B2_n = self.B_fields_at_nth_timestep[1]
    B3_n = self.B_fields_at_nth_timestep[2]

    (A_p1, A_p2, A_p3) = af.broadcast(self._A_p, self.q1_center, self.q2_center,
                                      self.p1, self.p2, self.p3,
                                      E1, E2, E3, B1_n, B2_n, B3_n,
                                      self.physical_system.params
                                     )
    
    # Defining a lambda function to perform broadcasting operations
    # This is done using af.broadcast, which allows us to perform 
    # batched operations when operating on arrays of different sizes
    addition = lambda a,b:a + b
    
    # af.broadcast(function, *args) performs batched operations on
    # function(*args)
    p1_new = af.broadcast(addition, self.p1, - 0.5 * dt * A_p1)
    p2_new = af.broadcast(addition, self.p2, - dt * A_p2)
    p3_new = af.broadcast(addition, self.p3, - dt * A_p3)

    p1_new = self._convert_to_p_expanded(p1_new)
    p2_new = self._convert_to_p_expanded(p2_new)
    p3_new = self._convert_to_p_expanded(p3_new)

    # Transforming interpolant to go from [0, N_p - 1]:
    p1_lower_boundary = self.p1_start + 0.5 * self.dp1
    p2_lower_boundary = self.p2_start + 0.5 * self.dp2
    p3_lower_boundary = self.p3_start + 0.5 * self.dp3

    p1_interpolant = (p1_new - p1_lower_boundary) / self.dp1
    p2_interpolant = (p2_new - p2_lower_boundary) / self.dp2
    p3_interpolant = (p3_new - p3_lower_boundary) / self.dp3

    # We perform the 3d interpolation by performing
    # individual 1d + 2d interpolations. Reordering to bring the
    # variation in values along axis 0 and axis 1

    self.f = self._convert_to_p_expanded(self.f)

    self.f = af.approx1(self.f,
                        p1_interpolant,
                        af.INTERP.CUBIC_SPLINE
                       )

    # Changing f, p2, p3 from (Np1, Np2, Np3, Nq1*Nq2) --> (Np2, Np3, Np1, Nq1*Nq2)
    self.f = af.approx2(af.reorder(self.f, 1, 2, 0),
                        af.reorder(p2_interpolant, 1, 2, 0),
                        af.reorder(p3_interpolant, 1, 2, 0),
                        af.INTERP.BICUBIC_SPLINE
                       )

    # Changing f from (Np2, Np3, Np1, Nq1*Nq2) --> (Np1, Np2, Np3, Nq1*Nq2)
    self.f = af.approx1(af.reorder(self.f, 2, 0, 1),
                        p1_interpolant,
                        af.INTERP.CUBIC_SPLINE,
                       )

    self.f = self._convert_to_q_expanded(self.f)
    af.eval(self.f)

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_interp3 += toc - tic

    return
