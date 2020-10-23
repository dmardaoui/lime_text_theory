# -*- coding: utf-8 -*-

"""

Auxilliary functions are collected in this file.

"""


"""
    This function extract a column from a dict list 
    
    INPUT:
        liste: dictionnary list
        classe: column of interest 
              
    OUTPUT:
        l: column of interest
        
"""
def extract(liste,classe):
        l=[]
        for t in liste:
            coeff=t[classe] 
            l.append(coeff)
        return l