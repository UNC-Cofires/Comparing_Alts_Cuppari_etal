# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 22:21:05 2021

@author: rcuppari
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import statsmodels.api as sm
import seaborn as sns

#import BPA_net_rev_func


df_payouts=pd.read_csv("payouts_index_5th.csv").iloc[:,1]
loading=0.4497447 ##for TDA 10th percentile strike
v=1
#BPA_net_rev_func.BPA_net_rev('collar',df_payouts,loading,v,0,other='5th')
#BPA_net_rev_func.BPA_net_rev('ind_swap',df_payouts,loading,v,0,other='5th')

payouts=pd.read_csv("payouts_index_15th.csv").iloc[:,1]
loading=0.3835421 ##for TDA 10th percentile strike
#BPA_net_rev_func.BPA_net_rev('collar',df_payouts,loading,v,0,other='15th')
#BPA_net_rev_func.BPA_net_rev('ind_swap',df_payouts,loading,v,0,other='15th')

import os 
os.chdir("E:/Research/PhD/BPA/Results")
Reserves5=pd.DataFrame(columns=['Reserves'])
TTP5=pd.DataFrame(columns=['TTP'])
CRAC5=pd.DataFrame(columns=['CRAC'])

for e in range (1,60):
    Reserves5=Reserves5.append(pd.read_excel('BPA_net_rev_stoc_y_index_51.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
    TTP5=TTP5.append(pd.read_excel('BPA_net_rev_stoc_y_index_51.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
    CRAC5=CRAC5.append(pd.read_excel('BPA_net_rev_stoc_y_index_51.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))

Reserves5.reset_index(inplace=True, drop=True)
TTP5.reset_index(inplace=True, drop=True)
CRAC5.reset_index(inplace=True, drop=True)

Reserves15=pd.DataFrame(columns=['Reserves'])
TTP15=pd.DataFrame(columns=['TTP'])
CRAC15=pd.DataFrame(columns=['CRAC'])

for e in range (1,60):
    Reserves15=Reserves15.append(pd.read_excel('BPA_net_rev_stoc_y_index_151.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
    TTP15=TTP15.append(pd.read_excel('BPA_net_rev_stoc_y_index_151.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
    CRAC15=CRAC15.append(pd.read_excel('BPA_net_rev_stoc_y_index_151.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
Reserves15.reset_index(inplace=True, drop=True)
TTP15.reset_index(inplace=True, drop=True)
CRAC15.reset_index(inplace=True, drop=True)


Reserves=pd.DataFrame(columns=['Reserves'])
TTP=pd.DataFrame(columns=['TTP'])
CRAC=pd.DataFrame(columns=['CRAC'])

for e in range (1,60):
    Reserves=Reserves.append(pd.read_excel('BPA_net_rev_stoc_y_index1.0.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
    TTP=TTP.append(pd.read_excel('BPA_net_rev_stoc_y_index1.0.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
    CRAC=CRAC.append(pd.read_excel('BPA_net_rev_stoc_y_index1.0.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
Reserves.reset_index(inplace=True, drop=True)
TTP.reset_index(inplace=True, drop=True)
CRAC.reset_index(inplace=True, drop=True)


NR=pd.read_csv("ann_net_rev_index1.0.csv").iloc[:,1]
NR15=pd.read_csv("ann_net_rev_index_151.csv").iloc[:,1]
NR5=pd.read_csv("ann_net_rev_index_51.csv").iloc[:,1]

hedgeStats=pd.DataFrame(columns=['AvgNR','95%VaR','Reserves','TTP','CRAC'])
hedgeStats.loc[0,'Reserves']=np.quantile(Reserves5,0.10)
hedgeStats.loc[0,'TTP']=TTP5[TTP5['TTP']==1].count()[0]
hedgeStats.loc[0,'CRAC']=np.quantile(CRAC5,.95)
hedgeStats.loc[0,'AvgNR']=NR5.mean()
hedgeStats.loc[0,'95%VaR']=np.quantile(NR5,.05)

hedgeStats.loc[2,'Reserves']=np.quantile(Reserves15,0.10)
hedgeStats.loc[2,'TTP']=TTP15[TTP15['TTP']==1].count()[0]
hedgeStats.loc[2,'CRAC']=np.quantile(CRAC15,.95)
hedgeStats.loc[2,'AvgNR']=NR15.mean()
hedgeStats.loc[2,'95%VaR']=np.quantile(NR15,.05)


hedgeStats.loc[1,'Reserves']=np.quantile(Reserves,0.10)
hedgeStats.loc[1,'TTP']=TTP[TTP15['TTP']==1].count()[0]
hedgeStats.loc[1,'CRAC']=np.quantile(CRAC,.95)
hedgeStats.loc[1,'AvgNR']=NR.mean()
hedgeStats.loc[1,'95%VaR']=np.quantile(NR,.05)

################################################
##same for swap
################################################
ReservesS=pd.DataFrame(columns=['Reserves'])
TTPS=pd.DataFrame(columns=['TTP'])
CRACS=pd.DataFrame(columns=['CRAC'])

for e in range (1,60):
    ReservesS=ReservesS.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap1.0.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
    TTPS=TTPS.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap1.0.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
    CRACS=CRACS.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap1.0.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
ReservesS.reset_index(inplace=True, drop=True)
TTPS.reset_index(inplace=True, drop=True)
CRACS.reset_index(inplace=True, drop=True)

ReservesS15=pd.DataFrame(columns=['Reserves'])
TTPS15=pd.DataFrame(columns=['TTP'])
CRACS15=pd.DataFrame(columns=['CRAC'])

for e in range (1,60):
    ReservesS15=ReservesS15.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap115th.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
    TTPS15=TTPS15.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap115th.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
    CRACS15=CRACS15.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap115th.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))

ReservesS15.reset_index(inplace=True, drop=True)
TTPS15.reset_index(inplace=True, drop=True)
CRACS15.reset_index(inplace=True, drop=True)


ReservesS5=pd.DataFrame(columns=['Reserves'])
TTPS5=pd.DataFrame(columns=['TTP'])
CRACS5=pd.DataFrame(columns=['CRAC'])

for e in range (1,60):
    ReservesS5=ReservesS5.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap15th.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
    TTPS5=TTPS5.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap15th.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
    CRACS5=CRACS5.append(pd.read_excel('BPA_net_rev_stoc_y_ind_swap15th.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
ReservesS5.reset_index(inplace=True, drop=True)
TTPS5.reset_index(inplace=True, drop=True)
CRACS5.reset_index(inplace=True, drop=True)


NRS=pd.read_csv("ann_net_rev_ind_swap1.0.csv").iloc[:,1]
NRS15=pd.read_csv("ann_net_rev_ind_swap115th.csv").iloc[:,1]
NRS5=pd.read_csv("ann_net_rev_ind_swap15th.csv").iloc[:,1]

hedgeStats.loc[3,'Reserves']=np.quantile(ReservesS5,0.10)
hedgeStats.loc[3,'TTP']=TTPS5[TTPS5['TTP']==1].count()[0]
hedgeStats.loc[3,'CRAC']=np.quantile(CRACS5,.95)
hedgeStats.loc[3,'AvgNR']=NRS5.mean()
hedgeStats.loc[3,'95%VaR']=np.quantile(NRS5,.05)

hedgeStats.loc[5,'Reserves']=np.quantile(ReservesS15,0.10)
hedgeStats.loc[5,'TTP']=TTPS15[TTPS15['TTP']==1].count()[0]
hedgeStats.loc[5,'CRAC']=np.quantile(CRACS15,.95)
hedgeStats.loc[5,'AvgNR']=NRS15.mean()
hedgeStats.loc[5,'95%VaR']=np.quantile(NRS15,.05)


hedgeStats.loc[4,'Reserves']=np.quantile(ReservesS,0.10)
hedgeStats.loc[4,'TTP']=TTPS[TTPS15['TTP']==1].count()[0]
hedgeStats.loc[4,'CRAC']=np.quantile(CRACS,.95)
hedgeStats.loc[4,'AvgNR']=NRS.mean()
hedgeStats.loc[4,'95%VaR']=np.quantile(NRS,.05)

################################################
##same for collar 
################################################
ReservesC=pd.DataFrame(columns=['Reserves'])
TTPC=pd.DataFrame(columns=['TTP'])
CRACC=pd.DataFrame(columns=['CRAC'])

for e in range (1,60):
    ReservesC=ReservesC.append(pd.read_excel('BPA_net_rev_stoc_y_collar1.0.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
    TTPC=TTPC.append(pd.read_excel('BPA_net_rev_stoc_y_collar1.0.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
    CRACC=CRACC.append(pd.read_excel('BPA_net_rev_stoc_y_collar1.0.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
ReservesC.reset_index(inplace=True, drop=True)
TTPC.reset_index(inplace=True, drop=True)
CRACC.reset_index(inplace=True, drop=True)

ReservesC15=pd.DataFrame(columns=['Reserves'])
TTPC15=pd.DataFrame(columns=['TTP'])
CRACC15=pd.DataFrame(columns=['CRAC'])

for e in range (1,60):
    ReservesC15=ReservesC15.append(pd.read_excel('BPA_net_rev_stoc_y_collar115th.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
    TTPC15=TTPC15.append(pd.read_excel('BPA_net_rev_stoc_y_collar115th.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
    CRACC15=CRACC15.append(pd.read_excel('BPA_net_rev_stoc_y_collar115th.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
ReservesC15.reset_index(inplace=True, drop=True)
TTPC15.reset_index(inplace=True, drop=True)
CRACC15.reset_index(inplace=True, drop=True)


ReservesC5=pd.DataFrame(columns=['Reserves'])
TTPC5=pd.DataFrame(columns=['TTP'])
CRACC5=pd.DataFrame(columns=['CRAC'])

for e in range (1,60):
    ReservesC5=ReservesC5.append(pd.read_excel('BPA_net_rev_stoc_y_collar15th.xlsx', sheet_name='ensemble' + str(e), usecols=[1]))
    TTPC5=TTPC5.append(pd.read_excel('BPA_net_rev_stoc_y_collar15th.xlsx', sheet_name='ensemble' + str(e), usecols=[2]))
    CRACC5=CRACC5.append(pd.read_excel('BPA_net_rev_stoc_y_collar15th.xlsx', sheet_name='ensemble' + str(e), usecols=[5]))
ReservesC5.reset_index(inplace=True, drop=True)
TTPC5.reset_index(inplace=True, drop=True)
CRACC5.reset_index(inplace=True, drop=True)


NRC=pd.read_csv("ann_net_rev_collar1.0.csv").iloc[:,1]
NRC15=pd.read_csv("ann_net_rev_collar115th.csv").iloc[:,1]
NRC5=pd.read_csv("ann_net_rev_collar15th.csv").iloc[:,1]

hedgeStats.loc[6,'Reserves']=np.quantile(ReservesC5,0.10)
hedgeStats.loc[6,'TTP']=TTPC5[TTPC5['TTP']==1].count()[0]
hedgeStats.loc[6,'CRAC']=np.quantile(CRACC5,.95)
hedgeStats.loc[6,'AvgNR']=NRC5.mean()
hedgeStats.loc[6,'95%VaR']=np.quantile(NRC5,.05)

hedgeStats.loc[8,'Reserves']=np.quantile(ReservesC15,0.10)
hedgeStats.loc[8,'TTP']=TTPC15[TTPC15['TTP']==1].count()[0]
hedgeStats.loc[8,'CRAC']=np.quantile(CRACC15,.95)
hedgeStats.loc[8,'AvgNR']=NRC15.mean()
hedgeStats.loc[8,'95%VaR']=np.quantile(NRC15,.05)


hedgeStats.loc[7,'Reserves']=np.quantile(ReservesC,0.10)
hedgeStats.loc[7,'TTP']=TTPC[TTPC15['TTP']==1].count()[0]
hedgeStats.loc[7,'CRAC']=np.quantile(CRACC,.95)
hedgeStats.loc[7,'AvgNR']=NRC.mean()
hedgeStats.loc[7,'95%VaR']=np.quantile(NRC,.05)


hedgeStats['Strike']=['index, 5th','index, 10th','index, 15th',
          'swap, 5th','swap, 10th','swap, 15th',
          'collar, 5th','collar, 10th','collar, 15th',]       

hedgeStats['TTP_pct']=(hedgeStats['TTP']/1188)*100

hedgeStats.to_csv("alt_strike_PC_stats.csv")
import plotly.express as px
import plotly.io as pio
pio.renderers.default='png'

import plotly.graph_objects as go
df=hedgeStats
fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = df.index,
                   showscale=True),
        dimensions = list([
            dict(range = [110000000,120000000],
                #constraintrange = [98,136],
                label = 'Avg Net Rev', values = df['AvgNR'],
                tickvals=[110000000,115000000,120000000],
                ticktext = [' .',' .',' .']),
            dict(range = [-115000000,-75000000],
                #constraintrange = [-80,-134],
                label = '95th% VAR', values = df['95%VaR'],
                tickvals=[-75000000,-95000000,-115000000],
                ticktext = [' .',' .',' .']),                
            dict(range = [25000000,55000000],
                #constraintrange = [23,40],
                label = '10th% Reserves', values = df['Reserves'],
                tickvals=[25000000,35000000,45000000,55000000],
                ticktext = [' ',' ',' ',' ',' ',' ']),      
            dict(range = [96,98],
                #constraintrange = [2,8],
                label = 'Pct. Make Treasury Payment', values = df['TTP_pct'],
                tickvals=[96,97,98],
                ticktext = [' ',' ',' ',' ']),      
            dict(range = [2.5,1],
                #constraintrange = [0,10],
                label = 'Avg. CRAC Surcharge', values = df['CRAC'],
                tickvals=[2.5,2,1.5,1],
                ticktext = [' ',' ',' ',' ']),      
        ])
    )
)
fig.show()
pio.write_image(fig,'figures/alt_strikes.png',width=900,height=400)




fig = go.Figure(data=
    go.Parcoords(
        line = dict(#color = df['Strike'],
                   colorscale='viridis',
                   showscale=True),
        dimensions = list([
            dict(range = [100000000,120000000],
                #constraintrange = [98,136],
                label = 'Avg Net Rev', values = df['AvgNR'],
                tickvals=[100000000,110000000,120000000]),
#                ticktext = [' ',' ',' ',' ',' ',' ']),
            dict(range = [-110000000,-90000000],
                #constraintrange = [-80,-134],
                label = '95th% VAR', values = df['95%VaR'],
                tickvals=[-90000000,-100000000,-110000000]),
#                ticktext = [' ',' ',' ',' ',' ']),                
            dict(range = [25000000,35000000],
                #constraintrange = [23,40],
                label = '10th% Reserves', values = df['Reserves'],
                tickvals=[25000000,30000000,35000000]),
#                ticktext = [' ',' ',' ',' ',' ',' ']),      
            dict(range = [95,97],
                #constraintrange = [2,8],
                label = 'Pct. Make Treasury Payment', values = df['TTP_pct'],
                tickvals=[95,96,97]),
#                ticktext = [' ',' ',' ',' ']),      
            dict(range = [2,2.5],
                #constraintrange = [0,10],
                label = 'Avg. CRAC Surcharge', values = df['CRAC'],
                tickvals=[2,2.25,2.5]),
#                ticktext = [' ',' ',' ',' ']),      
        ])
    )
)
fig.show()
