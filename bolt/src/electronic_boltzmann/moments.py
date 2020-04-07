#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

from bolt.src.utils.integral_over_p import integral_over_p

def density(f, p_x, p_y, p_z, integral_measure):
    return(integral_over_p(f, integral_measure))

def j_x(f, p_x, p_y, p_z, integral_measure):
    return(integral_over_p(f * p_x, integral_measure))

def j_y(f, p_x, p_y, p_z, integral_measure):
    return(integral_over_p(f * p_y, integral_measure))
