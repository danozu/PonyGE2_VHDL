#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:31:49 2021

@author: allan
"""

import numpy as np

# Define new functions
#def protected_div(a, b):
#    if b == 0:
#        return 1
#    else:
#        return np.divide(a,b)
    
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def psin(n):
    #try:
    return np.sin(n)
    #except Exception:
    #    return np.nan

def pcos(n):
    #try:
    return np.cos(n)
    #except Exception:
    #    return np.nan

def add(a, b):
    return np.add(a,b)

def sub(a, b):
    return np.subtract(a,b)

def mul(a, b):
    return np.multiply(a,b)

def sqrt(a):
    return np.sqrt(a)

def max_(a,b):
    return np.maximum(a, b)

def min_(a,b):
    return np.minimum(a, b)