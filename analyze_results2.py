# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:31:52 2021

@author: rcuppari
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns 
import os 
import random
from density_plots import density_scatter_plot

#######################################################################
##Make density plot for the two of them (maybe in violin plot format)
##but simultaneously plot points underneath the curve
#######################################################################
Net_rev = pd.read_csv("Results/ann_net_rev_2021update.csv")
nr_swap_slope = pd.read_csv("Results/ann_net_rev_swap_fixed_400.csv").iloc[:,1]
nr_swap_fixed = pd.read_csv("Results/ann_net_rev_swap_slope_400.csv").iloc[:,1]
nr_coll_fixed = pd.read_csv("Results/ann_net_rev_coll_fixed_400.csv").iloc[:,1]
nr_coll_slope = pd.read_csv("Results/ann_net_rev_coll_slope_400.csv").iloc[:,1]
nr_ind_fixed = pd.read_csv("Results/ann_net_rev_index_fixed_400.csv").iloc[:,1]
nr_ind_slope = pd.read_csv("Results/ann_net_rev_index_slope_400.csv").iloc[:,1]
###############################################################
## visualize results 
###############################################################
## read in relevant inputs 
fcrps = pd.read_csv("CAPOW_data/Synthetic_streamflows_FCRPS.csv").iloc[:,1:]
names = pd.read_excel("CAPOW_data//BPA_name.xlsx")
fcrps.columns = names.iloc[0,:]

years=pd.DataFrame(np.arange(0,1200))
df=pd.DataFrame({'year':years.values.repeat(365)})

fcrps.loc[:,'year']=df['year']

drop = [82,150,374,377,540,616,928,940,974,980,1129,1191]
fcrps2 = fcrps[~fcrps.year.isin(drop)]
fcrps2.reset_index(inplace = True)

ann_comb = fcrps2.groupby(np.arange(len(fcrps2.index)) // 365).mean()

tda = pd.DataFrame(ann_comb['Dalls ARF'])
tda['TDA flow'] = "Normal"

## set tda strike
strike_pct = .10
strike_pct2 = .5

strike = np.quantile(ann_comb['Dalls ARF'], strike_pct)
strike2 = np.quantile(ann_comb['Dalls ARF'], strike_pct2)

for i in range(0,len(tda)):
    if tda.iloc[i,0] < strike:
        tda.iloc[i,1] = "Dry"

#tda.reset_index(inplace=True)
all_data=pd.concat([Net_rev.iloc[:,1], nr_swap_slope, nr_swap_fixed, \
                    nr_coll_fixed, nr_coll_slope, \
                    nr_ind_fixed, nr_ind_slope, tda['TDA flow']],axis=1)
all_data.columns=['Status Quo', 'Modified CFD', 'CFD', \
                  'Capped Collar', 'Modified Collar', \
                  'Index','Modified Index','TDA flow']
all_data.dropna(inplace=True)

melted=all_data.melt(id_vars=['TDA flow'])

melted_slim=all_data[['Status Quo', 'Modified CFD', 'CFD', \
                  'Capped Collar', 'Modified Collar', \
                  'Index','Modified Index','TDA flow']] \
                 .melt(id_vars=['TDA flow'])

slim = all_data[['Status Quo', 'Modified CFD', 'CFD', \
                  'Capped Collar', 'Modified Collar', \
                  'Index','Modified Index','TDA flow']]
    
full_cols=['Status Quo','SQ X', 'Mod CFD','Mod CFD X', \
           'CFD', 'CFD X', 'Collar', 'Collar X', \
           'Mod Collar','Mod Collar X','Index','Index X', \
           'Mod Index', 'Mod Index X', 'TDA flow']
points=pd.DataFrame(index=slim.index,columns=full_cols)



density_scatter_plot(slim, tda, points, same_plot = True) 



os.chdir("Results")
Net_rev = pd.read_csv("ann_net_rev_2021update.csv")#.iloc[:,1]
Net_rev.colnames = ['Net_Rev']

Net_rev_index = pd.read_csv("ann_net_rev_index_fixed_400.csv").iloc[:,1]
Net_rev_swap = pd.read_csv("ann_net_rev_swap_fixed_400.csv").iloc[:,1]
Net_rev_collar = pd.read_csv("ann_net_rev_coll_fixed_400.csv").iloc[:,1]
Net_Rev = Net_rev

######################### function to append NR ###########################

## prefix = string with file name before xlsx, num_ensemb = # 20 year ensembles
def append_nr(prefix, num_ensem = 60):
    for e in range (1, num_ensem):
        Net_rev=Net_rev.append(pd.read_excel(prefix + '.xlsx', sheet_name='ensemble' + str(e), usecols=[7]))
    Net_rev.reset_index(inplace=True, drop=True)
    return Net_rev

def CRAC(prefix, num_ensem):
    CRAC_e=pd.DataFrame()
    for e in range (1,60):
        CRAC_e=pd.concat([CRAC_e, pd.read_excel(prefix + '.xlsx', sheet_name='ensemble' + str(e), usecols=[6])], axis=1)
    CRAC_med=CRAC_e.max(axis=0)
    return CRAC_e, CRAC_med

############################### plots ######################################
all_data=pd.concat([Net_rev,Net_rev_index,Net_rev_swap,Net_rev_collar,
                    tda['TDA flow']],axis=1)
all_data.columns=['Status Quo','Index','Swap','Collar','TDA flow']
all_data.dropna(inplace=True)

melted=all_data.melt(id_vars=['TDA flow'])

melted_slim=all_data[['Status Quo','Index','Swap','Collar','TDA flow']].melt(id_vars=['TDA flow'])

#######################################################################
##Make density plot for the two of them (maybe in violin plot format)
##but simultaneously plot points underneath the curve
#######################################################################
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(np.array(Net_rev), ax=ax, legend=False)
x = ax.lines[-1].get_xdata()
y = ax.lines[-1].get_ydata()

###############################################################
##need to convert density to net revenues
###############################################################

slim=all_data[['Status Quo','Index','Swap','Collar']]
full_cols=['Status Quo','SQ X','Index','Index X','Swap','Swap X','Collar','Collar X']
points=pd.DataFrame(index=slim.index,columns=full_cols)

density_scatter_plot(slim, tda, points, same_plot = True) 

os.chdir("C:/Users/rcuppari/OneDrive - University of North Carolina at Chapel Hill/Research/PhD Work/Columbia/BPA/Hist_data")
MidC=pd.read_csv('../BPA_Net_Revenue_Simulations/CAPOW_data/MidC_daily_prices_new.csv').iloc[:, 1:]
MidC=MidC.iloc[:,0]
MidC=pd.DataFrame(np.reshape(MidC.values, (365,1200), order='F'))
MidC.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
MidC=pd.DataFrame(np.reshape(MidC.values, (365*1188), order='F'))
MidC['year']=df['year']
MidC=MidC.groupby('year').agg({'mean'})

plt.figure()
plt.scatter(Net_rev, MidC.iloc[:,0])
plt.xlabel("Annual Net Revenues ($M)", fontsize = 14)
plt.ylabel("Avg. Mid C Price ($/MWh)", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show()

plt.figure()
plt.scatter(Net_rev, tda.iloc[:,1])
plt.xlabel("Annual Net Revenues")
plt.ylabel("Annual Streamflow")
plt.show()

plt.figure()
plt.scatter(tda.iloc[:,1], MidC.iloc[:,0])
plt.xlabel("Annual Streamflow")
plt.ylabel("Avg. Mid C Price")
plt.show()
###############################################################################

sns.kdeplot(np.array(Net_rev['Net_Rev']))
plt.scatter(dry.iloc[:,2],dry.iloc[:,1],label='Dry')
plt.scatter(normal.iloc[:,2],normal.iloc[:,1],label='Normal')
plt.title("Distribution of Yearly Net Revenue",fontsize=18)
plt.xlabel("Net Revenue ($M)",fontsize=18)
plt.xticks([-800000000,-600000000,-400000000,-200000000,0,200000000,400000000,600000000],
           ['-800','-600','-400','-200','0','200','400','600'],fontsize=16)
plt.yticks([],[])
plt.ylabel("Density",fontsize=18)
plt.legend(fontsize=18)


##########################################
##Annual net revenues 
##########################################
fig, ax = plt.subplots(figsize=(12, 6))
sns.kdeplot(np.array(Net_rev['Net_Rev']), ax=ax, legend=False)
x = ax.lines[-1].get_xdata()
y = ax.lines[-1].get_ydata()
ax.vlines(x0,0, y.max(), linestyle='--',linewidth=3,color='black') ##instead of 3 can use np.interp(x0, x, y) to fit under curve
ax.fill_between(x, y, where=x <= x0, color='red',alpha=.8)
ax.fill_between(x, y, where=x > x0,alpha=.8)
plt.show()
plt.title("Distribution of Yearly Net Revenue",fontsize=18)
plt.xlabel("Net Revenue ($M)",fontsize=18)
plt.xticks([-800000000,-600000000,-400000000,-200000000,0,200000000,400000000,600000000],
           ['-800','-600','-400','-200','0','200','400','600'],fontsize=16)
plt.yticks([],[])
plt.ylabel("Density",fontsize=18)





###############################################################################
##now need to figure out why the swap isn't improving any of the really bad 
##years (specifically the biggest loss year)
###############################################################################
plt.scatter(tda['Dalls ARF'].iloc[:1180],Net_rev['Net_Rev'])
tda_payouts=pd.read_csv("C:/Users/rcuppari/OneDrive - University of North Carolina at Chapel Hill/Research/PhD Work/Columbia/BPA/payouts_tda_swap.csv")
w=pd.concat([tda['Dalls ARF'].iloc[:1180],Net_rev['Net_Rev'],tda_payouts.iloc[:1180,1]],axis=1)


daily_rates=pd.read_csv("MidC_09_18_daily.csv")

import datetime as dt
daily_rates.iloc[:,0]=pd.to_datetime(daily_rates.iloc[:,0])
daily_rates['month']=daily_rates.iloc[:,0].dt.month
daily_rates['year']=daily_rates.iloc[:,0].dt.year

monthly_rates=daily_rates.groupby(['year','month']).agg('mean')
monthly_rates.reset_index(inplace=True)
monthly_rates['day']=1
monthly_rates['date']=pd.to_datetime(monthly_rates[["year",'month','day']])
monthly_rates.columns=['year','month','price','day','date']

pf_monthly=pd.read_excel("net_rev_data.xlsx",sheet_name="PF_rates")
pf_monthly=pf_monthly.iloc[:12,:9]

#pf_monthly=pf_monthly.T
pf_monthly.iloc[:,0]=[10,11,12,1,2,3,4,5,6,7,8,9]
pf_monthly=pf_monthly.sort_values(by=['MONTH '])

melt_pf=pf_monthly.melt(id_vars=['MONTH '])
melt_pf.columns=['month','year','PF']
melt_pf['day']=1
melt_pf['date']=pd.to_datetime(melt_pf[["year",'month','day']])

melt_pf['PF']=pd.to_numeric(melt_pf['PF'])
ann_pf=melt_pf.groupby(['year']).agg({'PF':'mean'})

ann_midc=daily_rates.groupby(['year']).agg('mean')

annual=pd.merge(ann_midc,ann_pf,on='year')
annual.reset_index(inplace=True)
annual.columns=['year','MidC','month','PF']
annual['year']=pd.to_numeric(annual['year'])

#plt.scatter(annual.iloc[:,0],annual.iloc[:,1],"Mid-C Market")
#plt.scatter(annual.iloc[:,0],annual.iloc[:,3],"Preference Customer")

plt.plot(annual['year'],annual['MidC'],label="Mid-C Prices",color='blue',alpha=.7,linewidth=3)
plt.plot(annual['year'],annual['PF'],label="PF Prices",color='orange',alpha=.7,linewidth=3)
plt.axhline(annual['MidC'].mean(),color='blue', linestyle='--',linewidth=3,alpha=.7,label='Average')
plt.axhline(annual['PF'].mean(),color='orange', linestyle='--',linewidth=3,alpha=.7,label='Average')
plt.xlabel("Year",fontsize=16)
plt.xticks(fontsize=14)
plt.ylabel("Average Annual Price ($/MWh)",fontsize=16)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
#plt.savefig('../Figures/annual_prices.png', bbox_inches='tight', dpi=1200)

mon2=monthly_rates[monthly_rates['year']>=2012]
plt.plot(mon2.date,mon2.price,label="Mid-C Prices",color='blue',alpha=.7,linewidth=3)
plt.axhline(mon2.price.mean(),linestyle='--',label="Average Historical Mid-C Price",color="blue",alpha=.5)
plt.plot(melt_pf.date,melt_pf.PF,label="PF Prices",color='orange',alpha=.7,linewidth=3)
plt.axhline(melt_pf.PF.mean(),linestyle='--',label="Average Historical PF Price",color="orange",alpha=.5)
plt.xlabel("Year",fontsize=16)
plt.ylabel("Average Monthly Price ($/MWh)",fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
#plt.savefig('../Figures/monthly_prices.png', bbox_inches='tight', dpi=1200)

##borrowing authority 
d={'year':[2017,2018,2019,2020,2021,2022,2023,2025,2028],'remain':[2.5,2.05,1.98,1.5,1,.5,0,-1,-2.5]}
BA=pd.DataFrame(d)

nr=pd.DataFrame({'year':[2017,2018,2019,2020],'nr':[.301, .361,.321,.334]})
avg=pd.DataFrame({'year':[2020,2021,2022,2023,2025,2028],'nr':[.334,.346, .346,.346,.346,.346]})

plt.plot(BA.year,BA.remain,color="black",alpha=.7,linewidth=3,label="Remaining BA")
plt.plot(nr.year,nr.nr,color="blue",alpha=.7,linewidth=2,label="Observed Net Revenues")
plt.plot(avg.year,avg.nr,color="blue",alpha=.7,linewidth=2,linestyle='--',label="Average Net Revenues")
plt.axhline(0,color='red',linestyle='--')
plt.xlabel("Year",fontsize=16)
plt.ylabel("Remaining Borrowing Authority ($B)",fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
#plt.savefig('../Figures/remain_BA.png', bbox_inches='tight', dpi=1200)


plt.scatter(tda.iloc[:1180,1],Net_rev_index0,label="SQ")
plt.scatter(tda.iloc[:1180,1],Net_rev_index,label="index",alpha=.5)
#plt.scatter(tda.iloc[:1180,1],Net_rev_swap,label="swap")
plt.legend()


payout=pd.read_csv("payouts_index.csv")
plt.scatter(tda.iloc[:1180,1],payout.iloc[:1180,1])


valid=pd.read_excel("net_rev_data.xlsx",sheet_name="Validation")
valid=valid.iloc[[0,8,10],:7]
valid=valid.T
valid.columns=['Year','Historic','Simulated']
valid.index=valid.loc[:,'Year']
valid['year2']=valid['Year']+0.4

import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
fig, ax = plt.subplots()
ax.scatter(valid['Historic'],valid['Simulated'])
plt.ylabel("Observed Net Revenues",fontsize=16)
plt.xlabel("Simulated Net Revenues",fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
line = mlines.Line2D([0, 1], [0, 1], color='red',linestyle='--')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.show()

plt.bar(valid['Year'],valid['Historic'],label="Historical",width=.4,alpha=.7)
plt.bar(valid['year2'],valid['Simulated'],label="Simulated",width=.4,alpha=.7)
plt.xlabel("Year")
plt.ylabel("Net Revenues ($)")
plt.legend()

import plotly.express as px
import plotly.graph_objects as go

fig =go.Figure(go.Sunburst(
    labels=[ "Net Revenues","Positive","Negative","Losses Covered", "Reserves + Uncovered", "Uncovered Losses", "TP Deferred"],
    parents=["","Net Revenues","Net Revenues","Negative","Negative","Negative","Uncovered Losses"],
    values=[1180,990,190,106,47,35,8],
    branchvalues="total"
))
fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
fig.show()
#plt.savefig('figures/chart',format='eps')


fig =px.sunburst(
    data,
    names='character',
    parents='parent',
    values='value',
)
fig.show()


