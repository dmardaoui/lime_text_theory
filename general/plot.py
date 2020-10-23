# -*- coding: utf-8 -*-

"""

In this file are collected the functions used to plot the graphics

"""



#import 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# linewidth
lw=5

# small font 
small_fs = 30

#big font 
big_fs = 40

#default kernel
default_kernel=0.25

"""
    This function plot all the beta in function of nu 
    
    INPUT:
        n_beta: number of beta
        words: local dictionnary
        bandwidth_max: maximum bandwidth 
        matrix_beta_kw: beta at different nu 
        matrix_variances: variances at different nu
        
    GRAPHIC:
        n_beta graphics representing each beta in function of nu 
        
"""
def plot_cancellation(n_beta,words,bandwidth_max,matrix_beta_kw,matrix_variances):             
    for k in range(0,n_beta):
            plt.figure(figsize=(15,10))
            ax=plt.axes()
            B_x="B"+(str(int(k+1)))
            Bx=r"$\hat{\beta}_{%s}$" % (str(int(k+1)))
            ax.plot(np.linspace(1,bandwidth_max,bandwidth_max)*0.01,matrix_beta_kw[k,:],color='MidnightBlue',label=Bx,linewidth=3)
            ax.axhline(y=0,color='black',linewidth=3,linestyle='dashed') 
            ax.axvline(x=0.25,color='red',linestyle='-',label=r"$\nu=0.25$",linewidth=3)
            ax.fill_between(np.linspace(1,bandwidth_max,bandwidth_max)*0.01, matrix_beta_kw[k,:]-matrix_variances[k,:], matrix_beta_kw[k,:]+matrix_variances[k,:],edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=3) 
            plt.title("Cancellation of interpretable coefficients",fontsize=50)
            plt.ylabel('%s' % (Bx),fontsize=60,rotation=0) 
            plt.xlabel(r'$\nu$',fontsize=60)  
            plt.xticks(fontsize=40)
            plt.yticks(fontsize=40)
            plt.legend(fontsize=50,loc=4)
            #save graphic
            s_name = "results/"+B_x
            plt.savefig(s_name + '.pdf',format='pdf',bbox_inches = 'tight',pad_inches = 0)
            

    

"""
    Plots whisker boxes for interpretable coefficients.
    
    INPUT:
        my_data: raw explanations (size (n_exp,dim+1))
        axis: plt axis (type matplotlib.axes._subplots.AxesSubplot)
        title: title of the figure (str)
        xlabel: label for the x axis (str)
        theo: theoretical values marked by crosses on the plot (size (dim+1,))
        rotate: classical view if True (bool)
        feature_names: default is 1,2,...
        ylims: providing ylims if needed
        color: color of the crosses
        c1: color of the box 
        alpha: transparency
        c2: color of the median 
        label: label 
        c3: color of the fliers
        
"""
def plot_whisker_boxes(my_data,
                       axis,
                       title=None,
                       xlabel=None,
                       theo=None,
                       rotate=False,
                       feature_names=None,
                       ylims=None,
                       color="red",
                       c1='black',
                       alpha=1,
                       c2="blue",
                       label="",
                       c3="black"):
    
    # get the dimension of the data
    dim = my_data.shape[1] -1
    
    # horizontal whiskerboxes
    if rotate:
        axis.boxplot(my_data[:,1:],showmeans=False,
                   boxprops= dict(linewidth=lw, color=c1,alpha=alpha), 
                   whiskerprops=dict(linestyle='-',linewidth=lw, color=c1,alpha=alpha),
                   medianprops=dict(linestyle='-',linewidth=lw,color=c2),
                   flierprops=dict(marker='o',markerfacecolor=c3,linewidth=lw),
                   capprops=dict(linewidth=lw,color=c1,alpha=alpha),
                   vert=False)
        axis.axvline(x=0,c='k',linestyle='--')
        
        
        y_pos = np.arange(dim) + 1
        axis.set_yticks(y_pos)

        if feature_names is None:
            feature_names = np.arange(1,dim+1)
            
        if feature_names is not None:
            print(feature_names)
        
        axis.set_yticklabels(feature_names)
        axis.invert_yaxis()
        axis.tick_params(labelsize=small_fs)
        
    # vertical whisker boxes
    else:
        
        # not plotting the intercept
        bp=axis.boxplot(my_data[:,1:],showmeans=False,
                   boxprops= dict(linewidth=lw, color=c1,alpha=alpha), 
                   whiskerprops=dict(linestyle='-',linewidth=lw, color=c1,alpha=alpha),
                   medianprops=dict(linestyle='-',linewidth=lw,color=c2,alpha=alpha),
                   flierprops=dict(marker='o',markerfacecolor=c3,linewidth=lw),
                   capprops=dict(linewidth=lw,color=c1,alpha=alpha))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axis.set_ylim(ylims)
    
        # plotting horizontal line to denote 0
        axis.axhline(y=0,c='k',linestyle='--')
        
    
        # plotting the theoretical predictions if any
        if theo is not None:

            for i_feature in range(dim):
                axis.plot(i_feature+1,
                        theo[i_feature+1],
                        'x',
                        #10: linear
                        #30: decision_tree
                        #markersize=5 decision tree
                        #markersize=3
                        markersize=18,
                        markeredgewidth=3,
                        zorder=10,
                        color=color) 
        # setting the labels
        if xlabel is None:
            axis.set_xlabel("words",fontsize=small_fs)
        
        # setting xticks and yticks
        axis.set_xticklabels(feature_names, rotation=90, fontsize=small_fs)
        axis.tick_params(labelsize=small_fs)
    
    # setting the title
    if title is None:
        title = "Coefficients of the surrogate model"
    axis.set_title(title,fontsize=big_fs)
    return bp

