# -*- coding: utf-8 -*-

'''

This file collected all the results of the sigma's coefficients

'''

########sigma_0########

"""
    This function returns sigma 0 
    
    INPUT:
        d: length of local dictionnary 
        a0: alpha 0 
        a1: alpha 1 
        a2= alpha 2
              
    OUTPUT:
        sigma 0 
        
"""
def sigma_0(d,a0,a1,a2):
    c1=-(a1+a2*(d-1))
    c2=d*a1**2-a0*a1-a0*a2*(d-1)
    return c1/c2

"""
    This function returns the limit of sigma 0 
    
    INPUT:
        d: length of local dictionnary 

    OUTPUT:
        limite of sigma_0 
        
"""
def lim_sigma_0(d):
    lim_0=(2*(2*d -1 ))/(d + 1)
    return lim_0

########sigma_1########

"""
    This function returns sigma 1 
    
    INPUT:
        d: length of local dictionnary 
        a0: alpha 0 
        a1: alpha 1 
        a2= alpha 2
              
    OUTPUT:
        sigma 1
        
"""
def sigma_1(d,a0,a1,a2):
    c1=a1
    c2=d*a1**2-a0*a1-a0*a2*(d-1)
    return c1/c2


"""
    This function returns the limit of sigma 1
    
    INPUT:
        d: length of local dictionnary 

    OUTPUT:
        limite of sigma_1
        
"""
def lim_sigma_1(d):
    lim_1=-6/(d+1)
    return lim_1

########sigma_2########
    
"""
    This function returns sigma 2
    
    INPUT:
        d: length of local dictionnary 
        a0: alpha 0 
        a1: alpha 1 
        a2= alpha 2
              
    OUTPUT:
        sigma 2
        
"""

def sigma_2(d,a0,a1,a2):
    c1=-a0*a1-(d-2)*(a2*a0)+(d-1)*a1**2
    c2=(d*a1**2-a0*a1-a0*a2*(d-1))*(a1-a2)
    return c1/c2

"""
    This function returns the limit of sigma 2
    
    INPUT:
        d: length of local dictionnary 

    OUTPUT:
        limite of sigma_2
        
"""
def lim_sigma_2(d):
    sigma_2=(6*(d**2-2*d+3))/((d+1)*(d-1))
    return sigma_2

########sigma_3########
"""
    This function returns sigma 3
    
    INPUT:
        d: length of local dictionnary 
        a0: alpha 0 
        a1: alpha 1 
        a2= alpha 2
              
    OUTPUT:
        sigma 3
        
"""
def sigma_3(d,a0,a1,a2):
    c1=a0*a2-a1**2
    c2=(d*a1**2-a0*a1-a0*a2*(d-1))*(a1-a2)
    return c1/c2

"""
    This function returns the limit of sigma 3
    
    INPUT:
        d: length of local dictionnary 

    OUTPUT:
        limite of sigma_3
        
"""
def lim_sigma_3(d):
    sigma_3=-(6*(d-3))/((d+1)*(d-1))
    return sigma_3