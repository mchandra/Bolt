#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

from bolt.src.utils.integral_over_v import integral_over_v

def density(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f, integral_measure))

def j_x(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * v1, integral_measure))

def j_y(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * v2, integral_measure))
