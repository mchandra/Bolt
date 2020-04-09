#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

from bolt.src.utils.integral_over_p import integral_over_p

import params

def density(f, p1, p2, p3, integral_measure):
    return(integral_over_p(f, integral_measure))

def j_x(f, p1, p2, p3, integral_measure):
    return(integral_over_p(f * params.p_x, integral_measure))

def j_y(f, p1, p2, p3, integral_measure):
    return(integral_over_p(f * params.p_y, integral_measure))
