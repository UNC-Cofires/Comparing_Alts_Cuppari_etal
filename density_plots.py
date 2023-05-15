# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 08:10:28 2022

@author: rcuppari
"""
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt 
import random 

################################################################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def match_points_density(data, points):
    for c in range(0,len(data.columns)-1):
        fig, ax = plt.subplots()
        sns.kdeplot(np.array(data.iloc[:, c]), ax=ax, legend=False)
        plt.close(fig)
        x = ax.lines[-1].get_xdata()
        y = ax.lines[-1].get_ydata()

        for i in range(0,len(data)):
            ##have to get each point's range from 0 to the curve at that neg net revenue 
            ##what is closest point to Net rev? 
            # print(data.iloc[i,c])
            point = find_nearest(x, data.iloc[i,c])
            ##retrieve the np array location for that point
            loc = np.where(x==point)
            y_max = y[loc]
            ##set up the range to sample 
            points.iloc[i,c*2+1] = random.uniform(0,y_max)
            points.iloc[:,c*2] = data.iloc[:,c]
##
            
## create function to make density plots with scatter plot inside
## need to have TDA data to differentiate between dry/wet
##showing the distribution 
## if same plot just put them in subplots. Else, separate figures
def density_scatter_plot(data, tda, points, same_plot = True, var = 95): 
    
    match_points_density(data, points)
    points['TDA flow'] = tda['TDA flow']
    dry = points[points.iloc[:,-1] == 'Dry']
    normal = points[points.iloc[:,-1] == 'Normal']
    
    rows = int((len(points.columns)-1)/2)
    
    if same_plot == True: 
        fig,axs = plt.subplots(rows,1,sharex=True)
    
        for c in range(0,rows):     
##need to hide the spine at some point
        #axs[c].kdeplot(np.array(data.iloc[:,c*2]))
        ##x = net rev, y = loc 
            axs[c].scatter(normal.iloc[:,c*2],normal.iloc[:,c*2+1],label='Normal',color="tab:blue",alpha=.6)
            axs[c].scatter(dry.iloc[:,c*2],dry.iloc[:,c*2+1],label='Dry',color="orange",alpha=.6) 
            axs[c].spines['top'].set_visible(False)
            axs[c].spines['right'].set_visible(False)
            axs[c].spines['left'].set_visible(False)
            axs[c].set_ylabel(str(data.columns[c]),fontsize=18)
            axs[c].set_yticks([],[])
            axs[c].set_xticks([-600000000,-400000000,-200000000,0,200000000,400000000,600000000])
            axs[c].set_xticklabels(['-600','-400','-200','0','200','400','600'],fontsize=20)
            axs[c].set_xlim(-600000000,600000000)
            axs[c].vlines(points.iloc[:,c*2].mean(), ymin=points.iloc[:,c*2+1].min(),
                          ymax=points.iloc[:,c*2+1].max(), color="black", linewidth=4,
                          linestyle='--', label = 'Mean Net Revenues')
            axs[c].vlines(np.quantile(points.iloc[:,c*2], (100-var)/100), ymin=points.iloc[:,c*2+1].min(),
                          ymax=points.iloc[:,c*2+1].max(),color="red",linewidth=4,
                          linestyle='--', label = f'{var}% VaR')
            
            if c < rows-1: 
                axs[c].spines['bottom'].set_visible(False)
                axs[c].set_xticks([],[])
                axs[c].legend(frameon = False, fontsize=14) 
#                axs[c].set_xlabel("Net Revenues ($M)", fontsize = 20)

    else: 
        for c in range(0,rows):     
##need to hide the spine at some point
        #axs[c].kdeplot(np.array(data.iloc[:,c*2]))
        ##x = net rev, y = loc 
            ax = "ax" + str(c)
            fig, ax = plt.subplots()
            ax.scatter(normal.iloc[:,c*2], normal.iloc[:,c*2+1], label='Normal', color="tab:blue",alpha=.6)
            ax.scatter(dry.iloc[:,c*2], dry.iloc[:,c*2+1], label='Dry', color="orange", alpha=.6) 
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_ylabel(str(data.columns[c]),fontsize=18)
            ax.set_yticks([],[])
            ax.set_xticks([-600000000,-400000000,-200000000,0,200000000,400000000,600000000])
            ax.set_xticklabels(['-600','-400','-200','0','200','400','600'],fontsize=20)
            ax.set_xlim(-600000000,600000000)
            ax.vlines(points.iloc[:,c*2].mean(),ymin=points.iloc[:,c*2+1].min(),ymax=points.iloc[:,c*2+1].max(),color="black",linewidth=4,linestyle='--')
            ax.vlines(np.quantile(points.iloc[:,c*2],0.05),ymin=points.iloc[:,c*2+1].min(),ymax=points.iloc[:,c*2+1].max(),color="red",linewidth=4,linestyle='--')
            ax.legend(frameon = False, fontsize=16)
            ax.set_xlabel("Net Revenues ($)", fontsize = 18)
           
                
def multiple_pdf(data, colors, var): 
    rows = int((len(data.columns)))
    
    fig, axs = plt.subplots(rows, sharex = True, sharey = True)

    for c in range(0,rows):     
##need to hide the spine at some point
    #axs[c].kdeplot(np.array(data.iloc[:,c*2]))
    ##x = net rev, y = loc 
        color1 = colors[c*2]
        color2 = colors[c*2+1]
        print(color1)
        print(color2)
        sns.kdeplot(data.iloc[:,c], fill = True, alpha = .7, 
                color = color1, ax = axs[c])
        axs[c].vlines(np.percentile(data.iloc[:,c], 100-var), 0, 6*pow(10,-9), label = '95% VaR', 
                      color = color2, linestyle = '--', linewidth = 3)
        axs[c].vlines(data.iloc[:,c].min(), 0, 6*pow(10,-9), label = 'Floor', color = color2,
                      linestyle = 'dotted', linewidth = 3)
        axs[c].vlines((data.iloc[:,c].mean()), 0, 6*pow(10,-9), label = 'Average', color = color2,
                      linestyle = 'dotted', linewidth = 3)
        axs[c].set_yticks([0],[""])
        axs[c].set_ylabel("Density", fontsize = 16)
        plt.xticks([-750000000, -500000000, -250000000, 0, 
                        250000000, 500000000, 750000000], 
                       ['-$750', '-$500', '-$250', '$0', '$250', "$500", "$750"], 
                       fontsize = 16)
        axs[c].set_title(str(data.columns[c]),fontsize=16)
        axs[c].set_xlabel("")
        axs[c].legend(loc = 'upper right', fontsize = 12)
        axs[c].patch.set_edgecolor('black')  
        axs[c].patch.set_linewidth('1')  
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
