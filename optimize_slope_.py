# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 22:09:33 2021

@author: rcuppari
"""
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.cm as cm
#######################################
##how to optimize index slope 
#######################################

##############################################################################
##only need the net revenue timeseries 
##and then want to essentially optimize the payouts given the costs 
##can also run the wang transform, but only do this for a couple because 
##shouldn't change much given the same strike
##############################################################################

##want to maximize 5th percentile of debt 
##AND the net revenues, AKA minimize the premium via the wang transform
##which means expected value since the premium % won't change wildly 
##will need to consider all 1200 years to get the expected value 
##can optimize for 5th percentile & reserves while keeping E[payouts] <=

##need to get the net revenues, form the index/extract the slope of the line
##need to calc payouts according to index + new slope 
##need to compare different slopes

##########################################################################
##ADAPTED FROM HAMILTON ET AL 2020 (SWE)
##########################################################################

##dataframe with payouts pre-calculated, assuming payouts are simply 
##following the regression line 
contracts=['index','ind_swap','collar'] #contracts
payouts=pd.read_csv("payouts_index.csv").iloc[:,1]
loading=0.8173384 ##for TDA 10th percentile strike

##all of them use the same "put" -- the index insurance with same strike and regression line
##only the other side is important

#if contract=='index': 
#    loading=0.817
#    payouts=pd.read_csv("payouts_index.csv").iloc[:,1] 
#elif contract=='ind_swap':
#    loading=0.652
#    payouts=pd.read_csv("payouts_index.csv").iloc[:,1] 
#else: 
#    loading=0
#    payouts=pd.read_csv("payouts_index.csv").iloc[:,1] 
##need to make the net revenue function a defined function so that it will just run 
import BPA_net_rev_func
#zero=np.zeros((1188,1))
#BPA_net_rev_func.BPA_net_rev("sq",zero,0,1)
##########################################################################
### calc min & avg adj rev given only contracts, 
### for different strikes/slopes (fig 6)####
### returns value ####
# ##########################################################################

def plot_cfd_slope_effect(dir_figs,payouts,loading,contract):
  
  v_list = np.arange(0,151,10)/100
  hedgeStats = np.empty([len(v_list), 3])
  count = 0
  
  
  for v in v_list:  # swap stats as function of slope v
    df_payouts= v * payouts
    BPA_net_rev_func.BPA_net_rev(contract,df_payouts,loading,v,0)
    
    BPA_Net_rev_y=pd.read_csv('E://Research//PhD//BPA//Results//ann_net_rev_'+str(contract)+str(v)+'.csv')
    hedgeStats[count, :] = [v, BPA_Net_rev_y.iloc[:,1].mean(), np.quantile(BPA_Net_rev_y.iloc[:,1], 0.05)]
    count += 1
    print(str(contract)+', slope is '+str(v))
  plt.figure()
  cmap = cm.get_cmap('viridis_r')
  cols = cmap(np.arange(2, hedgeStats.shape[0]+2) / (hedgeStats.shape[0]+2))
  cmapScalar = cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=0, vmax=1.5))
  cmapScalar._A = []
  plt.scatter(hedgeStats[:, 1], hedgeStats[:, 2], c=cols)
  plt.xlabel('Expected Hedged Net Revenue ($M/year)')
  plt.ylabel('Q05 Hedged Net Revenue ($M/year)')
  cbar = plt.colorbar(cmapScalar)
  cbar.ax.set_ylabel('Contract slope ($\$$M/inch)', rotation=270, labelpad=20)

  plot_name = dir_figs + str(contract)+'fig_cfdMarginal.jpg'
  plt.savefig(plot_name, bbox_inches='tight', dpi=1200)

  return 

directory="Figures/Contract_Slope"

for c in contracts:
    plot_cfd_slope_effect(directory, payouts, loading,c)

#for i in range(0,sim_years):
#    for j in range(0,hist_years):
#        RMSE = (np.sum(np.abs(monthly_sim_T[3:8,i]-monthly_hist_T[3:8,j])))
#        CHECK[i,j]=RMSE
#        if RMSE <= Best_RMSE:
#            year_list[i] = j
#            Best_RMSE=RMSE
            
#        else:
#            pass
#    Best_RMSE = 9999999999


#import os
#os.chdir("E:/Research/PhD/BPA/Results/")

#count=0
#hedgeStats = np.empty([len(v_list), 3])
#wap=pd.DataFrame()
#for v in v_list:  # swap stats as function of slope v
#    df=pd.read_csv("ann_net_rev_ind_swap"+str(v)+'.csv').iloc[:,1]
#    swap=pd.concat([swap,df],axis=1)
#    hedgeStats[count, :] = [v, df.mean(), np.quantile(df, 0.05)]
#    plt.figure()
#    cmap = cm.get_cmap('viridis_r')
#    cols = cmap(np.arange(2, hedgeStats.shape[0]+2) / (hedgeStats.shape[0]+2))
#    cmapScalar = cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=0, vmax=1.5))
#    cmapScalar._A = []
#    plt.scatter(hedgeStats[:, 1], hedgeStats[:, 2], c=cols)
#    plt.xlabel('Expected Hedged Net Revenue ($M/year)')
#    plt.ylabel('Q05 Hedged Net Revenue ($M/year)')
#    cbar = plt.colorbar(cmapScalar)
#    cbar.ax.set_ylabel('Contract slope ($\$$M/inch)', rotation=270, labelpad=20)
#    count+=1

#swap.columns=['0','0.2','0.4','0.6','0.8','1','1.2','1.4']



#v_list=v_list[1:]
#ind=pd.DataFrame()
#for v in v_list:  # swap stats as function of slope v
#    df=pd.read_csv("ann_net_rev_index"+str(v)+'.csv').iloc[:,1]
#    ind=pd.concat([ind,df],axis=1)

#ind.columns=['0.2','0.4','0.6','0.8','1','1.2','1.4']
    

##############################################################################
## PARALLEL PLOTS
## make a parallel plot with the categories as v's and a subplot for each FI 
## include the status quo (no fin instrument) as a bolded line in each 
##############################################################################
import os
os.chdir("E:/Research/PhD/BPA/Results/")

v_list = np.arange(0,151,10)/100
count=0

##status quo
#Reserves=pd.DataFrame(columns=['Reserves'])
#TTP=pd.DataFrame(columns=['TTP'])
#Net_rev=pd.DataFrame(columns=['Net_rev'])
#for e in range (1,60):
#    Net_rev=pd.read_csv('ann_net_rev_0.csv')
#    Reserves=Reserves.append(pd.read_excel('BPA_net_rev_stoc_y0.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
#    TTP=TTP.append(pd.read_excel('BPA_net_rev_stoc_y0.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))

#mean=Net_rev.mean()
#VAR=np.quantile(Net_rev,0.05)
#Res=np.quantile(Reserves,0.10)
#ttp=TTP[TTP['TTP']==1].count()[0]
#ttp_pct=1188-TTP[TTP['TTP']==1].count()[0]
#hedgeStats=pd.DataFrame([['mean',mean],['VAR',VAR],['Reserves',Res],
#                        ['TTP',ttp],['TTP_pct',ttp_pct]])

hedgeStats_swap = pd.DataFrame(index=v_list,columns=['v','mean','VAR']) 
hedgeStats2_swap=pd.DataFrame(index=v_list,columns=['Reserves','TTP','CRAC'])

Reserves_swap=pd.DataFrame(columns=['Reserves'])
TTP_swap=pd.DataFrame(columns=['TTP'])
CRAC_swap=pd.DataFrame(columns=['CRAC'])

for v in v_list:  # swap stats as function of slope v
    df=pd.read_csv("ann_net_rev_ind_swap"+str(v)+'.csv').iloc[:,1]
    hedgeStats_swap.iloc[count, :] = [v, df.mean(), np.quantile(df, 0.05)]
    
    for e in range (1,60):
        Reserves_swap=Reserves_swap.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap'+str(v)+'.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
        TTP_swap=TTP_swap.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap'+str(v)+'.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
        CRAC_swap=CRAC_swap.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap'+str(v)+'.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
    Reserves_swap.reset_index(inplace=True, drop=True)
    TTP_swap.reset_index(inplace=True, drop=True)
    CRAC_swap.reset_index(inplace=True, drop=True)

    hedgeStats2_swap.loc[v,'Reserves']=np.quantile(Reserves_swap,0.10)
    hedgeStats2_swap.loc[v,'TTP']=TTP_swap[TTP_swap['TTP']==1].count()[0]
    hedgeStats2_swap.loc[v,'CRAC']=np.quantile(CRAC_swap,.95)
        
    Reserves_swap=pd.DataFrame(columns=['Reserves'])
    TTP_swap=pd.DataFrame(columns=['TTP'])
    CRAC_swap=pd.DataFrame(columns=['CRAC'])
    
    count+=1


count=0
hedgeStats_ind = pd.DataFrame(index=v_list,columns=['v','mean','VAR']) 
hedgeStats2_ind=pd.DataFrame(index=v_list,columns=['Reserves','TTP','CRAC'])

Reserves_ind=pd.DataFrame(columns=['Reserves'])
TTP_ind=pd.DataFrame(columns=['TTP'])
CRAC_ind=pd.DataFrame(columns=['CRAC'])

for v in v_list:  # swap stats as function of slope v
    df=pd.read_csv("ann_net_rev_index"+str(v)+'.csv').iloc[:,1]
    hedgeStats_ind.iloc[count, :] = [v, df.mean(), np.quantile(df, 0.05)]
    
    for e in range (1,60):
        Reserves_ind=Reserves_ind.append(pd.read_excel('BPA_net_rev_stoc_y_index'+str(v)+'.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
        TTP_ind=TTP_ind.append(pd.read_excel('BPA_net_rev_stoc_y_index'+str(v)+'.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
        CRAC_ind=CRAC_ind.append(pd.read_excel('BPA_net_rev_stoc_y_index'+str(v)+'.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
    TTP_ind.reset_index(inplace=True, drop=True)
    Reserves_ind.reset_index(inplace=True, drop=True)
    CRAC_ind.reset_index(inplace=True, drop=True)

    hedgeStats2_ind.loc[v,'Reserves']=np.quantile(Reserves_ind,0.10)
    hedgeStats2_ind.loc[v,'TTP']=TTP_ind[TTP_ind['TTP']==1].count()[0]
    hedgeStats2_ind.loc[v,'CRAC']=np.quantile(CRAC_ind,.95)
        
    Reserves_ind=pd.DataFrame(columns=['Reserves'])
    TTP_ind=pd.DataFrame(columns=['TTP'])
    CRAC_ind=pd.DataFrame(columns=['CRAC'])
         
    count+=1
    
    
count=0
hedgeStats_coll = pd.DataFrame(index=v_list,columns=['v','mean','VAR']) 
hedgeStats2_coll=pd.DataFrame(index=v_list,columns=['Reserves','TTP','CRAC'])

Reserves_coll=pd.DataFrame(columns=['Reserves'])
TTP_coll=pd.DataFrame(columns=['TTP'])
CRAC_coll=pd.DataFrame(columns=['CRAC'])


for v in v_list:  # swap stats as function of slope v
    df=pd.read_csv("ann_net_rev_collar"+str(v)+'.csv').iloc[:,1]
    hedgeStats_coll.iloc[count, :] = [v, df.mean(), np.quantile(df, 0.05)]
    
    for e in range (1,60):
        Reserves_coll=Reserves_coll.append(pd.read_excel('BPA_net_rev_stoc_y_collar'+str(v)+'.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
        TTP_coll=TTP_coll.append(pd.read_excel('BPA_net_rev_stoc_y_collar'+str(v)+'.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
        CRAC_coll=CRAC_coll.append(pd.read_excel('BPA_net_rev_stoc_y_collar'+str(v)+'.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
    TTP_coll.reset_index(inplace=True, drop=True)
    Reserves_coll.reset_index(inplace=True, drop=True)
    CRAC_coll.reset_index(inplace=True, drop=True)

    hedgeStats2_coll.loc[v,'Reserves']=np.quantile(Reserves_coll,0.10)
    hedgeStats2_coll.loc[v,'TTP']=TTP_coll[TTP_coll['TTP']==1].count()[0]
    hedgeStats2_coll.loc[v,'CRAC']=np.quantile(CRAC_coll,.95)
        
    Reserves_coll=pd.DataFrame(columns=['Reserves'])
    TTP_coll=pd.DataFrame(columns=['TTP'])
    CRAC_coll=pd.DataFrame(columns=['CRAC'])
        
    count+=1

hedgeStats2_swap['TTP_pct']=(hedgeStats2_swap['TTP']/1188)*100
hedgeStats2_ind['TTP_pct']=(hedgeStats2_ind['TTP']/1188)*100
hedgeStats2_coll['TTP_pct']=(hedgeStats2_coll['TTP']/1188)*100

swap_stats=pd.concat([hedgeStats_swap,hedgeStats2_swap],axis=1)
index_stats=pd.concat([hedgeStats_ind,hedgeStats2_ind],axis=1)
collar_stats=pd.concat([hedgeStats_coll,hedgeStats2_coll],axis=1)

##HERE
#swap_stats.loc['contract',:]=['Swap']
#hedgeStats.loc['contract',:]=['Status Quo']
#index_stats['contract',:]=['Index']
#collar_stats.loc['contract',:]=['Collar']

#all_stats=pd.merge(swap_stats,hedgeStats.T)
#pd.plotting.parallel_coordinates(all_stats,'v')

pd.plotting.parallel_coordinates(swap_stats,'v',colormap='viridis')

import plotly.express as px
import plotly.io as pio
pio.renderers.default='png'

import plotly.graph_objects as go

##shouldn't need to calculate "status quo" because that should be v=0

df=swap_stats
#df.loc[17,'v']=-1
#df.loc[17,1:]=mean[1],VAR,Res,ttp,ttp_pct
fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['v'],
                   colorscale = 'Sunset',
                   showscale=True),
        dimensions = list([
            dict(range = [94000000,136000000],
                #constraintrange = [98,136],
                label = 'Avg Net Rev', values = df['mean'],
                tickvals=[94000000,100000000,110000000,120000000,130000000,136000000],
                ticktext = [' ',' ',' ',' ',' ',' ']),
            dict(range = [-148000000,-80000000],
                #constraintrange = [-80,-134],
                label = '95th% VAR', values = df['VAR'],
                tickvals=[-80000000,-100000000,-120000000,-140000000,-148000000],
                ticktext = [' ',' ',' ',' ',' ']),                
            dict(range = [4500000,55000000],
                #constraintrange = [23,40],
                label = '10th% Reserves', values = df['Reserves'],
                tickvals=[4500000,15000000,25000000,35000000,45000000,55000000],
                ticktext = [' ',' ',' ',' ',' ',' ']),      
            dict(range = [92,98],
                #constraintrange = [2,8],
                label = 'Pct. Make Treasury Payment', values = df['TTP_pct'],
                tickvals=[92,94,96,98],
                ticktext = [' ',' ',' ',' ']),      
            dict(range = [4,1],
                #constraintrange = [0,10],
                label = 'Avg. CRAC Surcharge', values = df['CRAC'],
                tickvals=[1,2,3,4],
                ticktext = [' ',' ',' ',' ']),      
        ])
    )
)

fig.show()
pio.write_image(fig,'figures/parallel_swap.png',width=900,height=400)


df=index_stats

fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['v'],
                   colorscale = 'Sunset',
                   showscale=True),
        dimensions = list([
            dict(range = [94000000,136000000],
                #constraintrange = [98,136],
                label = 'Avg Net Rev', values = df['mean'],
                tickvals=[94000000,100000000,110000000,120000000,130000000,136000000],
                ticktext = [' ',' ',' ',' ',' ',' ']),
            dict(range = [-148000000,-80000000],
                #constraintrange = [-80,-134],
                label = '95th% VAR', values = df['VAR'],
                tickvals=[-80000000,-100000000,-120000000,-140000000,-148000000],
                ticktext = [' ',' ',' ',' ',' ']),                
            dict(range = [4500000,55000000],
                #constraintrange = [23,40],
                label = '10th% Reserves', values = df['Reserves'],
                tickvals=[4500000,15000000,25000000,35000000,45000000,55000000],
                ticktext = [' ',' ',' ',' ',' ',' ']),      
            dict(range = [92,98],
                #constraintrange = [2,8],
                label = 'Pct. Make Treasury Payment', values = df['TTP_pct'],
                tickvals=[92,94,96,98],
                ticktext = [' ',' ',' ',' ']),      
            dict(range = [4,1],
                #constraintrange = [0,10],
                label = 'Avg. CRAC Surcharge', values = df['CRAC'],
                tickvals=[1,2,3,4],
                ticktext = [' ',' ',' ',' ']),      
        ])
    )
)

fig.show()
pio.write_image(fig,'figures/parallel_index.png',width=900,height=400)

df=collar_stats
fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df['v'],
                   colorscale = 'Sunset',
                   showscale=True),
        dimensions = list([
            dict(range = [94000000,136000000],
                #constraintrange = [98,136],
                label = 'Avg Net Rev', values = df['mean'],
                tickvals=[94000000,100000000,110000000,120000000,130000000,136000000],
                ticktext = [' ',' ',' ',' ',' ',' ']),
            dict(range = [-148000000,-80000000],
                #constraintrange = [-80,-134],
                label = '95th% VAR', values = df['VAR'],
                tickvals=[-80000000,-100000000,-120000000,-140000000,-148000000],
                ticktext = [' ',' ',' ',' ',' ']),                
            dict(range = [4500000,55000000],
                #constraintrange = [23,40],
                label = '10th% Reserves', values = df['Reserves'],
                tickvals=[4500000,15000000,25000000,35000000,45000000,55000000],
                ticktext = [' ',' ',' ',' ',' ',' ']),      
            dict(range = [92,98],
                #constraintrange = [2,8],
                label = 'Pct. Make Treasury Payment', values = df['TTP_pct'],
                tickvals=[92,94,96,98],
                ticktext = [' ',' ',' ',' ']),      
            dict(range = [4,1],
                #constraintrange = [0,10],
                label = 'Avg. CRAC Surcharge', values = df['CRAC'],
                tickvals=[1,2,3,4],
                ticktext = [' ',' ',' ',' ']),      
        ])
    )
)

fig.show()
pio.write_image(fig,'figures/parallel_collar.png',width=900,height=400)


all_stats=pd.concat([index_stats.iloc[10,:], swap_stats.iloc[10,:],collar_stats.iloc[10,:]],axis=1)
all_stats.columns=['index','swap','colllar']


##calculate net payouts and join with TDA flow
os.chdir("C:/Users/rcuppari/OneDrive - University of North Carolina at Chapel Hill/Research/PhD Work/Columbia/BPA")
ind=pd.read_csv("payouts_index.csv").iloc[:,1]
ind_prem=ind.mean()*0.8173384 

coll=(pd.read_csv("BPA_payment_collar.csv").iloc[:,1])*v #payment tied to index + loading premium and scaled with v 
coll_prem=(coll.mean()*(1+0.6379174))

swap=(pd.read_csv("BPA_payment_swap.csv").iloc[:,1])*v #payment tied to index + loading premium and scaled with v 
swap_prem=(swap.mean()*(1+0.667731))

net_ind=ind-ind_prem
net_swap=ind+swap_prem-ind_prem-swap
net_coll=ind+coll_prem-ind_prem-coll

TDA=pd.read_csv("../../Yufei_data/Synthetic_streamflows/synthetic_streamflows_TDA.csv")
TDA.columns=['day','flow']

years=pd.DataFrame(np.arange(0,1218))
df=pd.DataFrame({'year':years.values.repeat(365)})
TDA['year']=df['year']

##run 6x200 so need to drop the first year and last two every 200 years
drop=[0,201,202,203,404,405,406,607,608,609,810,811,812,1013,1014,1015,1216,1217]
TDA=TDA[~TDA.year.isin(drop)]

years=pd.DataFrame(np.arange(0,1200))
df=pd.DataFrame({'year':years.values.repeat(365)})

TDA2=TDA.loc[:,['day','flow']]
TDA2=TDA2.reset_index()
TDA2.loc[:,'year']=df['year']

drop2=[82,150,374,377,540,616,928,940,974,980,1129,1191]
TDA2=TDA2[~TDA2.year.isin(drop2)]

TDA2.reset_index(inplace=True)

years=pd.DataFrame(np.arange(0,1188))
df=pd.DataFrame({'year':years.values.repeat(365)})

TDA2.loc[:,'year']=df['year']

ann_TDA=TDA2.groupby('year').agg('mean')
ann_TDA=ann_TDA.loc[:,'flow']

np.quantile(ann_TDA,.1)

plt.scatter(ann_TDA,net_swap,label="Capped CFD Contract",alpha=1,s=70,color='orange')
plt.scatter(ann_TDA,net_ind,label="Index Insurance",s=70,alpha=.5,color='lightblue')
plt.yticks([-25000000,0,100000000,200000000,300000000,400000000,500000000,600000000,700000000],
           ['-25','0','100','200','300','400','500','600','700'],fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Net Payment ($M)",fontsize=16)
plt.xlabel("TDA Annual Avg. Streamflow (cfs)",fontsize=16)
plt.axvline(145516,linewidth=3,linestyle='--',color='black',label="TDA 10th%")
plt.legend(fontsize=16)
plt.show() 

plt.scatter(ann_TDA,net_coll,label="Capped Collar Contract",alpha=1,s=70,color='green')
plt.scatter(ann_TDA,net_ind,label="Index Insurance",s=70,alpha=.5,color='lightblue')
plt.yticks([-25000000,0,100000000,200000000,300000000,400000000,500000000,600000000,700000000],
           ['-25','0','100','200','300','400','500','600','700'],fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Net Payment ($M)",fontsize=16)
plt.xlabel("TDA Annual Avg. Streamflow (cfs)",fontsize=16)
plt.axvline(145516,linewidth=3,linestyle='--',color='black',label="TDA 10th%")
plt.legend(fontsize=16)



















