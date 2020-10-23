# -*- coding: utf-8 -*-

'''

This file collected all the results of the alpha's coefficients

'''

import numpy as np

########alpha_0########

"""
    This function compute alpha 0 
    
    INPUT:
        d: length of local dictionnary 
        nu: kernel width
              
    OUTPUT:
        expectation: expectation
        
"""

def alpha_0(d,nu):
    sum_expectation=0
    for si in np.linspace(1,d,d):
        sum_expectation=sum_expectation+np.exp((-((1-(np.sqrt((d-si)/(d)))))**2)/(2*(nu**2)))
        
    expectation=(sum_expectation/d)
    
    return expectation 

"""
    This function returns the limit of alpha 0 
    
    INPUT:
        d: length of local dictionnary 
              
    OUTPUT:
        lim: limit
        
"""
def lim_alpha_0(d):
    lim=1
    return lim


########alpha_1########
    
"""
    This function compute alpha 1
    
    INPUT:
        d: length of local dictionnary 
        nu: kernel width
              
    OUTPUT:
        expectation: expectation
        
"""
    
def alpha_1(d,nu):
    result=0
    for u in np.linspace(1,int(d),int(d)):
        result=result + (1-(u/d))*np.exp(-(((1-np.sqrt((d-u)/(d))))**2)/(2*(nu**2)))
    result=(result)/(d)
    
    return result 

"""
    This function returns the limit of alpha 1
    
    INPUT:
        d: length of local dictionnary 
              
    OUTPUT:
        lim: limit
        
"""
def lim_alpha_1(d):
    lim=(d-1)/(2*d)
    return lim

########alpha_2########
    
"""
    This function compute alpha 2
    
    INPUT:
        d: length of local dictionnary 
        nu: kernel width
              
    OUTPUT:
        expectation: expectation
        
"""

def alpha_2(d,nu):
    result=0

    for u in np.linspace(1,int(d),int(d)):
        result=result + (1-(u/d))*((d-1-u)/(d-1))*np.exp(-((1-np.sqrt((d-u)/(d))))**2/(2*(nu**2)))
    result=(result)/(d)
    
    return result

"""
    This function returns the limit of alpha 02
    
    INPUT:
        d: length of local dictionnary 
              
    OUTPUT:
        lim: limit
        
"""
def lim_alpha_2(d):
    lim=(d-2)/(3*d)
    return lim 

########alpha_3########
    
"""
    This function compute alpha 3
    
    INPUT:
        d: length of local dictionnary 
        nu: kernel width
              
    OUTPUT:
        expectation: expectation
        
"""
    
def alpha_3(d,nu):
    result=0

    for u in np.linspace(1,int(d),int(d)):
        result=result + (1-(u/d))*((d-1-u)/(d-1))*((d-u-2)/(d-2))*np.exp(-((1-np.sqrt((d-u)/(d))))**2/(2*(nu**2)))
    result=(result)/(d)
    return result

"""
    This function returns the limit of alpha 03
    
    INPUT:
        d: length of local dictionnary 
              
    OUTPUT:
        lim: limit
        
"""
def lim_alpha_3(d):
    lim=(d-3)/(4*d)
    return lim 

########alpha_4########

"""
    This function compute alpha 4
    
    INPUT:
        d: length of local dictionnary 
        nu: kernel width
              
    OUTPUT:
        expectation: expectation
        
"""

def alpha_4(d,nu):
    result=0

    for u in np.linspace(1,int(d),int(d)):
        result=result + (1-(u/d))*((d-1-u)/(d-1))*((d-u-2)/(d-2))*((d-u-3)/(d-3))* np.exp(-((1-np.sqrt((d-u)/(d))))**2/(2*(nu**2)))
    result=(result)/(d)
    return result

"""
    This function returns the limit of alpha 4
    
    INPUT:
        d: length of local dictionnary 
              
    OUTPUT:
        lim: limit
        
"""
def lim_alpha_4(d):
    lim=(d-4)/(5*d)
    return lim 



