# -*- coding: utf-8 -*-

'''

This file  collect all the results of the beta coefficients for specific model: linear,decision tree

'''

#import 
from theory.alphas import alpha_0,alpha_1,alpha_2,alpha_3,alpha_4
from theory.sigmas import sigma_0,sigma_1,sigma_2,sigma_3
import numpy as np 

"""
    This function compute the value of a decision tree with three words
    
    INPUT:
        d: length of local dictionnary 
        l: word 1
        m: word 2
        t: word 3
        nu: kernelwidth
              
    OUTPUT:
        beta: beta coefficients 
        
"""
def beta_f_decision_tree_three_words(d,l,m,t,nu):
 
    a0=alpha_0(d=d,nu=nu)
    a1=alpha_1(d=d,nu=nu)
    a2=alpha_2(d=d,nu=nu)
    a3=alpha_3(d=d,nu=nu)
    a4=alpha_4(d=d,nu=nu)
    
    sig0=sigma_0(d=d,a0=a0,a1=a1,a2=a2)
    sig1=sigma_1(d=d,a0=a0,a1=a1,a2=a2)
    sig2=sigma_2(d=d,a0=a0,a1=a1,a2=a2)
    sig3=sigma_3(d=d,a0=a0,a1=a1,a2=a2)
    
    B0=sig0*(a1+a2-a3)+sig1*(a1+a3*(d-5)) +sig1*(a2*(d+1) -a4*(d-3))
    Bl=sig1*(a1+a2-a3) + sig2*a1 +sig3*(a2*(d+1) + a3*(d-5) -a4*(d-3) )
    Bmt=sig1*(a1+a2-a3)  + sig2*(2*a2-a3) + sig3*(a1+a2*(d-1) +a3*(d-4) -a4*(d-3))
    Bdiff=sig1*(a1+a2-a3) +sig2*(a2+a3-a4)+ sig3*(d-4)*(a2+a3-a4) +sig3*(a1 + 2*(2*a2-a3))
    
        
    beta=Bdiff*np.ones((d+1,1)) 
    beta[0,0]=B0
    beta[l+1,0]=Bl
    beta[m+1,0]=Bmt
    beta[t+1,0]=Bmt
    
    return beta

"""
    This function compute the value of a decision tree with five words
    
    INPUT:
        d: length of local dictionnary 
        l: word 1
        m: word 2
        t: word 3
        r: word 4 
        v: word 5 
        nu: kernelwidth
              
    OUTPUT:
        beta: beta coefficients 
        
"""
def beta_f_decision_tree_complex(d,l,m,t,r,v,nu):
    print("nu:",nu) 
    a0=alpha_0(d=d,nu=nu)
    a1=alpha_1(d=d,nu=nu)
    a2=alpha_2(d=d,nu=nu)
    a3=alpha_3(d=d,nu=nu)
    a4=alpha_4(d=d,nu=nu)

    
    #sig0=sigma_0(d=d,a0=a0,a1=a1,a2=a2)
    sig1=sigma_1(d=d,a0=a0,a1=a1,a2=a2)
    sig2=sigma_2(d=d,a0=a0,a1=a1,a2=a2)
    sig3=sigma_3(d=d,a0=a0,a1=a1,a2=a2)
    
    #B0=sig0*(2*(a1+a2) - a3) +sig1*(2*a1 + 2*(d+1)*a2 + a3*(2*d-7) - (d-3)*a4)
    Bl=sig1*(2*(a1+a2) -a3) +sig2*( a1+ a2 + a3) +sig3*(a1+11*a2+2*a3-2*a4 + (d-5)*(2*(a2+a3) - a4))
    Bdiff=sig1*(2*(a1+a2) - a3) +sig2*(2*(a2+a3)-a4) +sig3*(2*a1+12*a2+3*a3-2*a4 + (d-6)*(2*(a2+a3) - a4)) 
    Bmt=sig1*(2*(a1+a2) - a3)  +sig2*(3*a2) +sig3*(2*a1+9*a2+3*a3-2*a4 + (d-5)*(2*(a2+a3) - a4)) 
    Br= sig1*(2*(a1+a2) - a3)  +sig2*(a1+2*a2+a3-a4)+sig3*(a1+10*a2+2*a3-a4 + (d-5)*(2*(a2+a3) - a4)) 
    Bv= sig1*(2*(a1+a2) - a3)  +sig2*(3*a2+a3-a4) +sig3*(2*a1+9*a2+2*a3-a4 + (d-5)*(2*(a2+a3) - a4)) 

        
    beta=Bdiff*np.ones((d+1,1)) 
    beta[l+1,0]=Bl
    beta[m+1,0]=Bmt
    beta[t+1,0]=Bmt
    beta[r+1,0]=Br
    beta[v+1,0]=Bv

    return beta

"""
    This function compute the value of a linear function generated randomly
    
    INPUT:
        d: length of local dictionnary 
        tfidf: TF-IDF transformation 
        f: coefficient of the linear model 
        dico: local dictionnary 
        gd: global dictionnary 
              
    OUTPUT:
        beta_final: beta coefficients 
        
"""
def beta_f_linear(d,tfidf,f,dico,gd):
    beta=np.zeros((d,1)) 
    produit=1.36*tfidf.toarray()[0]*f
    
    for n,word in enumerate(dico):
        if word in gd:
            index=gd.index(word)
            beta[n,0]=produit[index]
        else:
            beta[n,0]=0
            
    beta_final=np.zeros((d+1,1))   
    beta_final[1:,0]=beta[:,0]

    return beta_final


