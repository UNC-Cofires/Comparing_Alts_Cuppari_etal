# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:45:26 2020

@author: sdenaro
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta  
import numpy as np
from numpy import matlib as matlib
import seaborn as sns
import statsmodels.api as sm
sns.set(style='whitegrid')
import matplotlib.cm as cm 
from matplotlib.colors import ListedColormap
import os 

os.chdir("Results")
redux='res_nomax'

######################################################################
##### ENSEMBLE ANALYSIS ##############
#
##Create single ensemble horizonatal panels plot 
#plt.rcParams.update({'font.size': 12})
#for e in range (1,60):
#    Net_rev_e=pow(10,-9)*pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[7])['Net_Rev']
#    Positive=Net_rev_e>0
#    fig, axes = plt.subplots(nrows=5, ncols=1)
#    ax1=axes[0]
#    Net_rev_e.plot(kind="bar",
#                                          linewidth=0.2,
#                                          ax=ax1, 
#                                          color=Positive.map({True:'blue', False:'red'}))  # make bar plots
#    ax1.set_title('Net Revenue Ensemble '+str(e), pad=0.6)
#    ax1.xaxis.set_ticks(range(1, 21, 1))
#    ax1.set_xticklabels([],[])
#    #ax1.set_xticklabels([i for i in np.arange(1,21,1)])
#    ax1.set_ylabel('B$')
#    ax1.set_xlim(-0.5,19.5)
#    ax1.grid(linestyle='-', linewidth=0.2, axis='x')
#    ax1.get_yaxis().set_label_coords(-0.08,0.5)
#    
#    Reserves_e=pow(10,-9)*pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[1])
#    Reserves_e=Reserves_e.append(pd.Series(Reserves_e.iloc[19]))
#    Reserves_e.reset_index(inplace=True, drop=True)
#    Treas_fac1=320*pow(10,-3)   # Treasury facility (1)
#    ax2 = axes[1]
#    ax2.axhline(0.608691000-Treas_fac1, color='r') 
#    ax2.axhline(0, color='r') 
#    ax2.plot(Reserves_e ) 
#    ax2.set_title('Reserves', pad=0.6)
#    ax2.xaxis.set_ticks(range(1, 21, 1))
#    ax2.set_xticklabels([],[])
#    #ax2.set_xticklabels([i for i in np.arange(1,21,1)])
#    ax2.set_ylabel('B$')
#    ax2.set_xlim(0.5,20.5)
#    ax2.grid(linestyle='-', linewidth=0.2, axis='x')
#    ax2.get_yaxis().set_label_coords(-0.08,0.5)
#
#    
#    Remaining_BA_e=pow(10,-9)*pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[3])
#    Remaining_BA_e=Remaining_BA_e.append(pd.Series(Remaining_BA_e.iloc[19]))
#    Remaining_BA_e.reset_index(inplace=True, drop=True)
#    ax3 = axes[2]
#    ax3.axhline(0, color='r') 
#    ax3.plot(Remaining_BA_e) 
#    ax3.set_title('Remaining Borrowing Authority', pad=0.6)
#    ax3.xaxis.set_ticks(range(1, 21, 1))
#    ax3.set_xticklabels([],[])    
#    ax3.set_ylabel('B$')
#    ax3.set_xlim(0.5,20.5)
#    ax3.set_ylim(bottom=-0.08)
#    ax3.grid(linestyle='-', linewidth=0.2, axis='x')
#    ax3.get_yaxis().set_label_coords(-0.08,0.5)
#
#    
#    
#    
#    TF1=pow(10,-9)*pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[4])
#    TF2=pow(10,-9)*pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[5])
#    TF_e=pd.concat([TF1, TF2], axis=1)
#    TF_e.append(TF_e.iloc[19,:])
#    TF_e.reset_index(inplace=True, drop=True)
#    ax4 = axes[3]
#    TF_e.plot(ax=ax4, kind='bar', stacked=True, color=['g','y'],  linewidth=0.2)
#    ax4.set_title('Treasury Facility', pad=0.6)
#    ax4.set_xticklabels([],[])
#    #ax4.set_xticklabels([i for i in np.arange(1,21,1)])
#    ax4.set_ylabel('B$')
#    ax4.xaxis.set_ticks(range(1, 21, 1))
#    ax4.set_ylabel('B$')
#    ax4.set_xlim(0.5,20.5)
#    ax4.grid(linestyle='-', linewidth=0.2, axis='x')
#    ax4.get_yaxis().set_label_coords(-0.08,0.5)
#    
#    CRAC_e=pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[6])
#    CRAC_e=CRAC_e.append(pd.Series(CRAC_e.iloc[19]))
#    CRAC_e.reset_index(inplace=True, drop=True)
#    ax5 = axes[4]
#    #plot percent increase
#    #ax5.plot(CRAC_e*100/PF_rates_avg, 'darkviolet')
#    #plot $/MWh increase
#    ax5.plot(CRAC_e, 'darkviolet')
#    ax5.set_title('Surcharge', pad=0.6)
#    ax5.xaxis.set_ticks(range(1, 21, 1))
#    ax5.set_xticklabels([i for i in np.arange(1,21,1)])
#    #ax5.set_ylabel('%')
#    ax5.set_ylabel('$/MWh')
#    ax5.set_xlim(0.5,20.5)
#    ax5.grid(linestyle='-', linewidth=0.2, axis='x')
#    ax5.get_yaxis().set_label_coords(-0.08,0.5)
#    
#    plt.subplots_adjust(left=0.11, bottom=0.065, right=0.985, top=0.945, wspace=0.2, hspace=0.345)
#    #plt.savefig('figures/Ensembles'+ redux + '/Ensemble'+ str(e))
#    


########### QuantilePlots
# CRAC ensemble  
CRAC_e=pd.DataFrame()
for e in range (1,60):
    CRAC_e=pd.concat([CRAC_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[6])], axis=1)
#Qc=(100/PF_rates_avg)*CRAC_e.T
Qc=CRAC_e.T
Qc.reset_index(inplace=True, drop=True)

#CRAC distribution
count=np.sum(CRAC_e.any())
percent1=100*count/59  #BAU=11.86% 
print ('Percent of CRAC ensembles: %.2f' % percent1 )

#Reserves ensembles
Reserves_e=pd.DataFrame()
for e in range (1,60):
    Reserves_e=(pd.concat([Reserves_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[1])['Reserves'] - 
                           pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[4])['TF1']-
                           pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[5])['TF2'] ], axis=1)) 
Qr=pow(10,-9)*Reserves_e.T
Qr.reset_index(inplace=True, drop=True)

#BA ensembles
Remaining_BA_e=pd.DataFrame()
for e in range (1,60):
    Remaining_BA_e=pd.concat([Remaining_BA_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[3])], axis=1) 
Qba=pow(10,-9)*Remaining_BA_e.T
Qba.reset_index(inplace=True, drop=True)


#Revenues ensembles
Revs_e=pd.DataFrame()
for e in range (1,60):
    Revs_e=pd.concat([Revs_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[7])['Net_Rev']], axis=1)
Qrev=pow(10,-9)*Revs_e.T
Qrev.reset_index(inplace=True, drop=True)


TTP_e=pd.DataFrame()
for e in range (1,60):
    TTP_e=pd.concat([TTP_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[2])], axis=1)
count=sum(sum(sum([TTP_e != 1]).values)) #0% for both BAU and minus 10% and minus20%
percent2=100-100*count/1188  
print ('TPP: %.2f' % percent2 )
    
## QuantilePlot ensembles function
def quantileplot(Q, ax, color, ci, name, start_day, end_day, realization, tick_interval, log):
    # plot a selected streamflow realization (realization arg) over the
    # quantiles of all streamflow realizations
    if log:
        Q = np.log10(Q)
    ps = np.arange(0,1.01,0.05)*100
    for j in range(1,len(ps)):
        u = np.percentile(Q.iloc[:, start_day:end_day], ps[j], axis=0)
        l = np.percentile(Q.iloc[:, start_day:end_day], ps[j-1], axis=0)
        if ax == ax1:
            ax.fill_between(np.arange(0,len(Q.iloc[0,start_day:end_day])), l, u, \
                            color=cm.twilight_shifted(ps[j-1]/100.0), alpha=0.75, edgecolor='none', label=[str(int(ps[j-1]))+'% to '+ str(int(ps[j])) +'%'])
                            #color=cm.PuOr(ps[j-1]/100.0), alpha=0.75, edgecolor='none')
        else:
            ax.fill_between(np.arange(0,len(Q.iloc[0,start_day:end_day])), l, u, \
                            color=cm.GnBu(ps[j-1]/100.0), alpha=0.75, edgecolor='none',  label=[str(int(ps[j-1]))+'% to '+ str(int(ps[j])) +'%'])
                #color=cm.RdYlBu_r(ps[j-1]/100.0), alpha=0.75, edgecolor='none')
                
    ax.set_xlim([0, end_day-start_day])
#    ax.set_xticks(np.arange(0, end_day-start_day+tick_interval, tick_interval))
#    ax.set_xticklabels(np.arange(start_day+1, end_day+tick_interval, tick_interval))

    ax.plot(np.arange(0,len(Q.iloc[0,start_day:end_day])), Q.median(), color='k', linewidth=2, label='median')
    #ax.plot(np.arange(0,len(Q.iloc[0,start_day:end_day])), Q.iloc[(realization-1), \
    #    start_day:end_day], color='k', linewidth=2)
    #ax.set_ylim([0, 5])
    #ax.set_yticks(np.arange(6))
    #ax.set_yticklabels([0, '', '', '', '', 5])
    #ax.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct', 'Jan', 'Apr', 'Jul', 'Oct'])

    ax.set_ylabel(name, fontsize=12)
    #ax.set_xlabel('Simulation Day')
    #for xl,yl in zip(ax.get_xgridlines(), ax.get_ygridlines()):
    #   xl.set_linewidth(0.5)
    #    yl.set_linewidth(0.5)
    plt.legend()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(left=0.075, right=0.82, top=0.96, bottom=0.055)
    

fig, ax1 = plt.subplots(1, 1)
quantileplot(Qr, color='k', ax=ax1, ci=90, name='B$', \
    start_day=0, end_day=20, realization=59, tick_interval=1, log=False)
ax1.axhline(0, color='r', linestyle='--')
ax1.set_title('Net Reserves', size=15)
plt.xlim(0,19)
plt.ylim(-0.67,0.35)
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Ensembles/Reserves' + redux, format='jpg')
#plt.savefig('figures/Ensembles/Reserves'  + redux  )



fig, ax1 = plt.subplots(1, 1)
quantileplot(Qba, color='k', ax=ax1, ci=90, name='B$', \
    start_day=0, end_day=20, realization=59, tick_interval=1, log=False)
ax1.axhline(0, color='r', linestyle='--')
ax1.set_title('Remaining Borrowing Authority', size=15)
plt.xlim(0,19)
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Ensembles/Remaining_BA' + redux, format='jpg')



fig, ax1 = plt.subplots(1, 1)
quantileplot(Qrev, color='k', ax=ax1, ci=90, name='B$', \
    start_day=0, end_day=20, realization=59, tick_interval=1, log=False)
ax1.axhline(0, color='r', linestyle='--')
ax1.set_title('Net Revenue', size=15) 
plt.xlim(0,19)
plt.ylim(-0.8, 1.9)
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Ensembles/Net_Rev' + redux, format='jpg')


fig, ax1 = plt.subplots(1, 1)
quantileplot(Qc, color='k', ax=ax1, ci=90, name='$/MWh', \
    start_day=0, end_day=20, realization=59, tick_interval=1, log=False)
ax1.axhline(0, color='r', linestyle='--')
ax1.set_title('Rate increase', size=15) 
plt.xlim(0,19)
plt.ylim(0,9.5)
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Ensembles/CRAC'  + redux , format='jpg')


######################################################################
# Yearly firm loads (aMW)
df_load=pd.read_excel('../net_rev_data.xlsx',sheet_name=0,skiprows=[0,1], usecols=[9])
PF_load_y=df_load.loc[[13]].values 
IP_load_y=df_load.loc[[3]].values 
Tot_load=PF_load_y+IP_load_y

Results=pd.concat([Revs_e.melt().drop('variable', 1), Reserves_e.melt().drop('variable', 1), CRAC_e.melt().drop('variable', 1)], axis=1)
Results.columns=['Net_revs','Reserves_','CRAC_']
Results['TF']= -Results.Reserves_.where(Results.Reserves_<0, 0).shift(-1)
Results['Reserves']= Results.Reserves_.where(Results.Reserves_>0, 0)
Results['CRAC']=Results['CRAC_'].shift(-1)
Results['CRAC']=((Results['CRAC']- Results['CRAC_'])*int(Tot_load)*24*365).shift(-1)
Results['year']= np.tile(np.arange(1,21),59)
Results['Uncovered_'] = (np.minimum((Results['Net_revs'] + Results['Reserves'] + Results['TF']), 0) 
+ (Results['CRAC']- Results['CRAC_'])).where(Results['Net_revs']<0, 0) 
#When positive it mens that CRAC was activated even if TF covered all losses
Results['Uncovered']=-(Results['Uncovered_'].where(Results['Uncovered_']<0, 0))
Results['Used_Res'] = np.minimum(np.abs(Results['Net_revs']), np.abs(Results['Reserves'])).where(Results['Net_revs']<0, 0) 



#sort the net revs
Results_sort=Results.sort_values(by='Net_revs',ascending=True)
Results_sort.reset_index(inplace=True, drop=True)

#Calculate probabilities and divide per year
VaR_90_5 = Results_sort.where((Results_sort.Net_revs <= Results_sort.Net_revs.quantile(0.1)) &
(Results_sort.Net_revs > Results_sort.Net_revs.quantile(0.05)) & (Results_sort.year<=5)).dropna().mean(axis=0)

VaR_95_5 = Results_sort.where((Results_sort.Net_revs <= Results_sort.Net_revs.quantile(0.05)) & 
                              (Results_sort.Net_revs > Results_sort.Net_revs.quantile(0.01)) & (Results_sort.year<=5)).dropna().mean(axis=0)

VaR_99_5 = Results_sort.where((Results_sort.Net_revs <= Results_sort.Net_revs.quantile(0.01)) & (Results_sort.year<=5)).dropna().mean(axis=0)

df5=pd.concat([VaR_90_5, VaR_95_5, VaR_99_5], axis=1).reset_index()
df5.columns=['Value', '0.1','0.05','0.01']
df5['sort']=[9, 8, 7,1,6,2,5,4,3,0]
df5=df5.drop([0,1,2,4,6,7]).sort_values(by='sort')
df5=df5.drop(['sort'], axis=1)


df5.set_index('Value').T.plot(kind='bar', stacked=True,  width=0.45,
             colormap=ListedColormap(sns.color_palette('YlOrRd', 10)))
ax=plt.gca()
ax.set_ylim([0,0.62*pow(10,9)])
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Tools_5' + redux, format='eps')



#Calculate probabilities and divide per year
VaR_90_10 = Results_sort.where((Results_sort.Net_revs <= Results_sort.Net_revs.quantile(0.1)) &
(Results_sort.Net_revs > Results_sort.Net_revs.quantile(0.05)) & (Results_sort.year>5) & (Results_sort.year<=10)).dropna().mean(axis=0)

VaR_95_10 = Results_sort.where((Results_sort.Net_revs <= Results_sort.Net_revs.quantile(0.05)) & 
                              (Results_sort.Net_revs > Results_sort.Net_revs.quantile(0.01)) & (Results_sort.year>5) & (Results_sort.year<=10)).dropna().mean(axis=0)

VaR_99_10 = Results_sort.where((Results_sort.Net_revs <= Results_sort.Net_revs.quantile(0.01)) & (Results_sort.year>5) & (Results_sort.year<=10)).dropna().mean(axis=0)

df10=pd.concat([VaR_90_10, VaR_95_10, VaR_99_10], axis=1).reset_index()
df10.columns=['Value', '0.1','0.05','0.01']
df10['sort']=[9, 8, 7,1,6,2,5,4,3,0]
df10=df10.drop([0,1,2,4,6,7]).sort_values(by='sort')
df10=df10.drop(['sort'], axis=1)


df10.set_index('Value').T.plot(kind='bar', stacked=True,  width=0.45,
             colormap=ListedColormap(sns.color_palette('YlOrRd', 10)))
ax=plt.gca()
ax.set_ylim([0,0.62*pow(10,9)])
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Tools_10'+ redux, format='eps')




#Calculate probabilities and divide per year
VaR_90_20 = Results_sort.where((Results_sort.Net_revs <= Results_sort.Net_revs.quantile(0.1)) &
(Results_sort.Net_revs > Results_sort.Net_revs.quantile(0.05)) & (Results_sort.year>10) ).dropna().mean(axis=0)

VaR_95_20 = Results_sort.where((Results_sort.Net_revs <= Results_sort.Net_revs.quantile(0.05)) & 
                              (Results_sort.Net_revs > Results_sort.Net_revs.quantile(0.01)) & (Results_sort.year>10)).dropna().mean(axis=0)

VaR_99_20 = Results_sort.where((Results_sort.Net_revs <= Results_sort.Net_revs.quantile(0.01)) & (Results_sort.year>10)).dropna().mean(axis=0)

df20=pd.concat([VaR_90_20, VaR_95_20, VaR_99_20], axis=1).reset_index()
df20.columns=['Value', '0.1','0.05','0.01']
df20['sort']=[9, 8, 7,1,6,2,5,4,3,0]
df20=df20.drop([0,1,2,4,6,7]).sort_values(by='sort')
df20=df20.drop(['sort'], axis=1)


df20.set_index('Value').T.plot(kind='bar', stacked=True, width=0.45,
             colormap=ListedColormap(sns.color_palette('YlOrRd', 10)))
ax=plt.gca()
ax.set_ylim([0,0.62*pow(10,9)])
plt.subplots_adjust(left=0.105, bottom=0.055, right=0.735, top=0.95)
plt.savefig('figures/Tools_20' + redux, format='eps')


def repaid(redux, color):
    Repaid_e=pd.DataFrame()
    
    for e in range (1,60):
        Repaid_e=pd.concat([Repaid_e, pd.read_excel('BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=['repaid'])], axis = 1)

#    Repaid_e.fillna(0, inplace = True)
    Repaid_e = Repaid_e * pow(10, -2)
    repaid_avg = Repaid_e.mean(axis = 1)

    fig = plt.subplots(1, 1)
    plt.plot(Repaid_e, color = color, linewidth = 2, alpha = .7)
    plt.plot(repaid_avg, color = 'black', linewidth = 4, label = 'Average Repayment')
    plt.xlabel("Year", fontsize = 20)
    plt.ylabel("$M", fontsize = 20)
    plt.yticks([0, 25000000, 50000000, 75000000, 100000000, 125000000, 150000000, 175000000], 
               ['0', '25', '50', '75', '100', '125', '150', '175'], fontsize = 18)
    plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 
               ['1', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20'], fontsize = 18)
    plt.show()
    
    return Repaid_e, repaid_avg 

infinite, infin2 = repaid('infinite', 'lightblue')
#og, og2 = repaid('og', 'orange')
avg, avg2 = repaid('avg', 'orange')













