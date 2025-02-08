# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:37:36 2022

@author: rcuppari
"""

from datetime import timedelta
from cost_calculations import used
from datetime import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from cost_calculations import pv_calc
from cost_calculations import expected_repay
from cost_calculations import plot_cum_pv
from cost_calculations import compound_interest
from cost_calculations import TF_used
from cost_calculations import repaid as repaid_func
from cost_calculations import read_reserves
from cost_calculations import crac as crac_func
from cost_calculations import pv_opp_cost
from density_plots import density_scatter_plot

##############################################################################
##### SCRIPT USED TO MAKE ALL PLOTS IN MAIN TEXT AND SUPPLEMENTARY INFO ######
##### NOTE: figure 1 made in ArcGIS Online, and figure 2 made as PPT file ####
##############################################################################

## discount rate being used -- modify based on plots to make
discount = 2.5
time_horizon = 20

## names of files (sorry folks)
strike_suff = 'index05_long'
no_loc_suff = 'no_loc_10k'
loc0_suff = 'loc_repay0_10k'
loc_all_suff = 'loc_all_repay_10k'

## bootstrapping order 
sequences = pd.DataFrame(pd.read_csv("random_sequences.csv").iloc[:,1])

## color coordination
color_no_loc = 'darkorange'
color_loc0 = 'gold'
color_loc_all = 'olivedrab'
color_ins = '#71AFE2'


###############################################################################
########################### read in all data ##################################

# annual net revenue results for strategies with and without full reserves
# strategy 1
nr_no_loc = pd.read_csv("Results/ann_net_rev_" + no_loc_suff + ".csv").iloc[:, 2]
# strategies 2
nr_loc0 = pd.read_csv("Results/ann_net_rev_" + loc0_suff + ".csv").iloc[:, 2]
nr_loc_all = pd.read_csv("Results/ann_net_rev_" + loc_all_suff + ".csv").iloc[:, 2]
# strategy 3
nr_ins = pd.read_csv("Results/ann_net_rev_" + strike_suff + ".csv").iloc[:, 2]
            
# net payouts -- the min value (the only negative one) is the premium
premium = -pd.read_csv("Results/net_payouts2" + '05cheap2.csv').iloc[:,1].min()
            
# # CRAC
crac_no_loc = pd.DataFrame(pd.read_csv("Results/CRAC_" + no_loc_suff + ".csv").iloc[1:,1])
no_loc_crac = pd.DataFrame(np.reshape(crac_no_loc.values, (int(len(crac_no_loc)/20),20), order = 'F'))

crac_loc0 = pd.DataFrame(pd.read_csv("Results/CRAC_" + loc0_suff + ".csv").iloc[1:,1])
loc0_crac = pd.DataFrame(np.reshape(crac_loc0.values, (int(len(crac_no_loc)/20),20), order = 'F'))

crac_loc_all = pd.DataFrame(pd.read_csv("Results/CRAC_" + loc_all_suff + ".csv").iloc[1:,1]) 
loc_all_crac = pd.DataFrame(np.reshape(crac_loc_all.values, (int(len(crac_no_loc)/20),20), order = 'F'))

crac_ind = pd.DataFrame(pd.read_csv("Results/CRAC_" + strike_suff + ".csv").iloc[1:,1])
ins_crac = pd.DataFrame(np.reshape(crac_ind.values, (int(len(crac_no_loc)/20),20), order = 'F'))

## repayment
loc0_repaid = pd.DataFrame(pd.read_csv("Results/repaid_" + loc0_suff + ".csv").iloc[1:,1]) 
loc0_repaid = pd.DataFrame(np.reshape(loc0_repaid.values, (20,int(len(crac_no_loc)/20)), order = 'F'))

loc_all_repaid = pd.DataFrame(pd.read_csv("Results/repaid_" + loc_all_suff + ".csv").iloc[:,1])
loc_all_repaid = pd.DataFrame(np.reshape(loc_all_repaid.values, (20,int(len(crac_no_loc)/20)), order = 'F'))

###############################################################################
######################### FIG 3 Index Performance #############################
###############################################################################

# previously had this embedded in the index making code, but have now saved
# the predicted outputs so that they can be read in here
rev_above_index = pd.read_csv("Results/rev_above_index0.05.csv").iloc[:, 1]
rev_below_index = pd.read_csv("Results/rev_below_index0.05.csv").iloc[:, 1]

pred_above = pd.read_csv("Results/index_pred_above0.05.csv").iloc[:, 1]
pred_below = pd.read_csv("Results/index_pred_below0.05.csv").iloc[:, 1]

tda_index_above = pd.read_csv("Results/tda_index_above0.05.csv").iloc[:, 1]
tda_index_below = pd.read_csv("Results/tda_index_below0.05.csv").iloc[:, 1]

#####################################
# overlay the historical points also
hist_rev = pd.read_excel("net_rev_data.xlsx", sheet_name="hist_rev").loc[:, [
    "Year", "Adjusted Net Operating Revenues"]]

hist_rev.columns = ['year', 'outcome']
hist_rev['outcome'] = hist_rev['outcome']*pow(10, 3)

# now let's get streamflow at relevant dams
tda = pd.read_csv("../Hist_data/TDA6ARF_daily.csv")
midc = pd.read_excel("../Hist_data/MidC_yearly_01_21.xlsx")
midc.columns = ['date', 'midc']

tda['year'] = pd.to_datetime(tda['date']).dt.year
midc['year'] = pd.to_datetime(midc.iloc[:, 0]).dt.year

ann = tda.groupby('year').mean()
ann.columns = ['tda']

combined = ann.merge(midc, left_index=True,
                     right_on='year').merge(hist_rev, on='year')
combined = combined[combined['year'] >= 2009]

## index previously trained
pred_ind = 5.1*pow(10, 8) + 708*combined['tda'] - 2.6*pow(10, 7)*combined['midc']
pred_tda = -4.69*pow(10, 8) + 2977*combined['tda']

combined['ind_pred'] = pred_ind.copy()
combined['diff'] = combined['outcome'] - pred_ind

xmin = -800*pow(10, 6)
xmax = 650*pow(10, 6)
hspan = abs(xmin)/(abs(xmin) + xmax)

lab_font = 16
tick_font = 14

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize = (18,15))
fig.subplots_adjust(top = .98, bottom = .125)#, hspace = 0.1)
ax1.scatter(rev_above_index, tda_index_above, label="Normal/Wet Years",
            color='cornflowerblue', s=50, alpha=0.8)
ax1.scatter(rev_below_index, tda_index_below, label="Dry Years (payout)",
            color='brown', s=50, alpha=0.8)
ax1.set_ylabel("Dalles Based Index Value ($)", fontsize=lab_font)
#ax1.set_xlabel("Net Revenues ($M)", fontsize=lab_font)
ax1.axhspan(xmin, 0, 0, hspan, color='red', alpha=.1,
            label='Negative Net Revenues')
ax1.hlines(0, xmin, 0, color='black', linestyles='--',
            linewidth=3)  # , label = '$0 Index Value')
ax1.vlines(0, xmin, 0, color='black', linestyles='--',
            linewidth=3)  # , label = '$0 Net Revenues')
ax1.set_xticks([-800*pow(10, 6), -600000000, -400000000, -200000000, 0, 200000000, 400000000,
                600000000, 800000000],[])
                #['-800', '-600', '-400', '-200', '0', '200', '400', '600', '800'],
                #fontsize=tick_font)
ax1.set_yticks([-800*pow(10, 6), -600000000, -400000000, -200000000, 0, 200000000, 400000000,
                600000000, 800000000],
                ['-800', '-600', '-400', '-200', '0', '200', '400', '600', '800'],
                fontsize=tick_font)
ax1.scatter(combined['outcome'], pred_tda, marker='*', s=85,
            color='black', label='Historical Data')
ax1.plot([xmin, 0], [xmin, 0], linestyle='-.', color='black',
          label='1:1 line')
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(xmin, xmax)
ax1.legend(fontsize=tick_font, frameon=False)#, loc='upper left')

ax2.scatter(rev_above_index, pred_above, label='Normal/Wet Years',
            color='cornflowerblue', s=50, alpha=0.8)
ax2.scatter(rev_below_index, pred_below, alpha=.8,
            label='Dry Years (payout)', color='brown', s=50)
ax2.hlines(0, xmin, 0, color='black', linestyles='--',
            linewidth=3)  # , label = '$0 Index Value')
ax2.vlines(0, xmin, 0, color='black', linestyles='--',
            linewidth=3)  # , label = '$0 Net Revenues')
ax2.axhspan(xmin, 0, 0, hspan, color='red', alpha=.1, #.5, color='red', alpha=.1,
            label='Negative Net Revenues')
ax2.set_xlabel("Net Revenues ($M)", fontsize=lab_font)
ax2.set_ylabel("Composite Index Value ($M)", fontsize=lab_font)
ax2.set_xticks([-800*pow(10, 6), -600000000, -400000000, -200000000, 0, 200000000, 400000000,
                600000000, 800000000],
                ['-800', '-600', '-400', '-200', '0', '200', '400', '600', '800'],
                fontsize=tick_font)
ax2.set_yticks([-800*pow(10, 6), -600*pow(10, 6), -400*pow(10, 6), -200*pow(10, 6), 0,
                200*pow(10, 6), 400*pow(10, 6), 600*pow(10, 6), 800*pow(10, 6)],
                ['-800', '-600', '-400', '-200', '0', '200', '400', '600', '800'],
                fontsize=tick_font)
scatter = ax2.scatter(combined['outcome'], pred_ind, marker='*', s=85,
            color='black', label='Historical Data')
ax2.plot([xmin, 0], [xmin, 0], linestyle='-.', color='black',
          label='1:1 line')
#ax2.legend(frameon=False, fontsize=tick_font, loc='upper left')
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(xmin, xmax)


for i, txt in enumerate(combined.outcome):
    
    if (combined['year'].iloc[i] == 2010) | (combined['year'].iloc[i] == 2013) | (combined['year'].iloc[i] == 2011):
        ax2.annotate(combined['year'].iloc[i], (combined.outcome.iloc[i], combined.ind_pred.iloc[i]), 
                    xytext=(-15,-14), textcoords='offset points', fontsize = 12, weight='bold')
        ax1.annotate(combined['year'].iloc[i], (combined['outcome'].iloc[i], pred_tda.iloc[i]),
                      xytext=(-15,10), textcoords = 'offset points', fontsize = 12, weight = 'bold') 


    elif combined['year'].iloc[i] == 2009:
        ax2.annotate(combined['year'].iloc[i], (combined.outcome.iloc[i], combined.ind_pred.iloc[i]), 
                    xytext=(-13,8), textcoords='offset points', fontsize = 12, weight='bold')
        ax1.annotate(combined['year'].iloc[i], (combined['outcome'].iloc[i], pred_tda.iloc[i]),
                      xytext=(6,6), textcoords = 'offset points', fontsize = 12, weight = 'bold') 
    else:
        ax2.annotate(combined['year'].iloc[i], (combined.outcome.iloc[i], combined.ind_pred.iloc[i]), 
                    xytext=(-14,8), textcoords='offset points', fontsize = 12, weight='bold')
        ax1.annotate(combined['year'].iloc[i], (combined['outcome'].iloc[i], pred_tda.iloc[i]),
                      xytext=(6,6), textcoords = 'offset points', fontsize = 12, weight = 'bold') 
        
###############################################################################
####################### FIG 4 Historical Rates/BA #############################
###############################################################################

# hist reserves & ba
res_ba = pd.read_excel('net_rev_data.xlsx', sheet_name='res_ba')
res_ba.set_index('Unnamed: 0', inplace=True)
res_ba = res_ba.T

res = np.array(res_ba.loc[2010:2021, 'start_res'])
res = pd.to_numeric(res)

# historical repayment
repaid = pd.read_excel('BPA_Historical_Data/Debt/BPA_debt.xlsx', sheet_name='Repayment',
                       usecols=[0, 8, 9, 10, 11, 12, 13, 14, 15, 16]).iloc[0:4, :]
repaid.set_index('Unnamed: 0', inplace=True)
repaid2 = repaid.T

# hist BA and CRAC
ba = res_ba.loc[:2022, 'BA']
ba_add = res_ba.loc[:2022, 'addition']
ba_add.fillna(0, inplace=True)

for i in np.arange(2000, 2023):
    if ba_add[i] != 0:
        ba_add[i] = ba.max()

crac = res_ba.loc[:2022, 'CRAC']
crac.replace('Y', ba.max(), inplace=True)
crac.replace('n', 0, inplace=True)
crac[2000] = 0

# remove years that don't have both us treasury wacc + non-federal
bpa_wacc = pd.read_excel('../Hist_data/BPA_debt.xlsx', sheet_name='WAI')
bpa_wacc = bpa_wacc[(bpa_wacc['Name'] == 'Non-federal Total') | (bpa_wacc['Name'] == 'US Treasury ') |
                    (bpa_wacc['Name'] == 'Federal Spread') | (bpa_wacc['Name'] == 'Treasury Spread') |
                    (bpa_wacc['Name'] == 'Federal appropriations')]
bpa_wacc = bpa_wacc.drop(['Term', 'Average', 'Unnamed: 26', 'Source: Annual reports'],\
                         axis = 1)
bpa_wacc.set_index('Name', inplace=True)
# *100*100 ## convert from decimal to percent and then percent to basis points (100 basis pt = 1%)
wacc2 = bpa_wacc.T
#wacc2.index = pd.to_numeric(wacc2.index)

fig, axs = plt.subplots(2,1, figsize = (15,20), sharex = True)
axs[0].plot(ba, linewidth = 4, color = '#7BAFDE', label = 'Available Line of Credit')
axs[0].bar(crac.index, crac, color = "#E8601C", alpha = .3, label = 'Surcharge Triggered')
axs[0].bar(ba_add.index, ba_add, color = "#90C987", alpha = .4, label = 'Line of Credit Expanded')
axs[0].legend(loc = 'upper center', bbox_to_anchor=(0.6,1), fontsize = 14, frameon = False)
axs[0].set_xticks([2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022],
                  ['2000', '2002', '2004', '2006', '2008', '2010',
                    '2012', '2014', '2016', '2018', '2020', '2022'], fontsize = 14)
axs[0].set_yticks([2,4,6,8,10,12],
                  ['2', '4', '6', '8', '10', '12'], fontsize = 14)
axs[0].set_ylabel("Line of Credit Available ($B)", fontsize = 16)
axs[0].set_xlabel("Year", fontsize = 16)


axs[1].xaxis.set_tick_params(which='both', labelbottom=True)
axs[1].plot(wacc2['Non-federal Total'], color = '#EE8866', linestyle = '-.', \
          label = 'Market Rates', linewidth = 4)
axs[1].plot(wacc2['US Treasury '], color = '#77AADD', alpha = .8, \
          linestyle = '--', linewidth = 4, label = 'Line of Credit Rates')
axs[1].set_ylabel("Weighted Average Interest Rate (%)", fontsize = 16)
axs[1].set_xlabel("Year", fontsize = 16)
axs[1].set_yticks([0, .01, .02, .03, .04, .05,  .06, .07],
            ['0', '1', '2', '3', '4', '5', '6', '7'], fontsize = 14)
axs[1].legend(fontsize = 14, frameon = False)
axs[1].set_xlabel("Year", fontsize = 16)
axs[1].set_xticks([2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018,\
                    2020, 2022],
                  ['2000', '2002', '2004', '2006', '2008', '2010',
                    '2012', '2014', '2016', '2018', '2020', '2022'], fontsize = 14)
plt.setp(axs[1].get_xticklabels(), visible=True)
plt.setp(axs[0].get_xticklabels(), visible=True)

###############################################################################
####################### FIG 5 Dot density plots ###############################
###############################################################################

# read in relevant inputs
fcrps = pd.read_csv("CAPOW_data/Synthetic_streamflows_FCRPS.csv").iloc[:, 1:]
names = pd.read_excel("CAPOW_data//BPA_name.xlsx")
fcrps.columns = names.iloc[0, :]

years = pd.DataFrame(np.arange(0, 1200))
df = pd.DataFrame({'year': years.values.repeat(365)})

fcrps.loc[:, 'year'] = df['year']

drop = [82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191]
fcrps2 = fcrps[~fcrps.year.isin(drop)]
fcrps2.reset_index(inplace=True)

ann_comb = fcrps2.groupby(np.arange(len(fcrps2.index)) // 365).mean()

tda = pd.DataFrame(ann_comb['Dalls ARF'])
tda = tda.T
tda = tda.loc[:, sequences.iloc[:,0]]
tda = tda.T
tda.reset_index(inplace = True)
tda = pd.DataFrame(tda.iloc[:,1])

tda['TDA flow'] = "Dry"

strike = np.percentile(tda['Dalls ARF'], 5)
for i in range(0, len(tda)):
    if tda.iloc[i, 0] > strike:
        tda.iloc[i, 1] = "Normal"


# can make one just for the LOC repayment scenarios
no_tools = pd.read_csv("Results//ann_net_rev_no_tools_10k.csv").iloc[:,1]

all_data = pd.concat([no_tools, nr_no_loc, nr_loc0, \
                     nr_loc_all, nr_ins, \
                     tda['TDA flow']], axis=1)

all_data.columns = ['no_tools', 'no_loc', 'LOC0', \
                    'LOC_all', 'ins', \
                    'TDA flow']
#all_data.dropna(inplace=True)

slim = all_data[['no_tools', 'no_loc', 'LOC0', \
                 'LOC_all', 'ins', \
                 'TDA flow']]

full_cols = [['no_tools', 'none1', 'no_loc', 'none', 'LOC0', 'LOC0x', \
             'LOC_all', 'all_x', 'ins', 'insx', \
              'TDA flow']]

points = pd.DataFrame(index=slim.index, columns=full_cols)

density_scatter_plot(slim, tda, points, same_plot=True, var=95, 
                     cvar = False, label_font=14)

###############################################################################
######################## FIG 6 Opportunity Costs ##############################
###############################################################################

#################### calculate the pv/value differences #######################
# need to read in rates to calculate pv
t_rates = pd.read_csv('../Hist_data/bonds_20year.csv')
t_rates['date'] = pd.to_datetime(t_rates['DATE'])
t_rates['year'] = t_rates['date'].dt.year
t_rates = t_rates[['year', 'DGS20']]
t_rates = t_rates[t_rates['DGS20'] != '.']
t_rates['DGS20'] = pd.to_numeric(t_rates['DGS20'])

ann_bond = t_rates.groupby('year').mean()

rates_non_fed = [wacc2['Non-federal Total'].mean()] * 20

# two panel plot with opportunity costs for
# reserves (across three strategies)
# premium (across high/low res)

# calculate pv expected based on the average Treasury return for BPA
# treas = wacc2['US Treasury '].mean()#/100/100
rates_loc = [wacc2['US Treasury '].mean()] * 20

######## going to use treasury rates as the alternative, safe investment ######
######### use bond rates to get opp costs: alt. earnings ########

# pv of premium payments
t_rates2000s = t_rates[t_rates['year'] >= 2000]
avg_tr = t_rates2000s['DGS20'].mean()
rates_tr = pd.DataFrame(np.repeat(avg_tr/100, 20))

premium_ = pd.DataFrame(np.repeat(premium, 20))

# "rates" here doesn't matter because pv calc just uses the discount rate
# it was for the amortization function
disc_prems = pv_calc(premium_, discount=discount)
#pv_prem = pv_calc(premium_, discount = discount)

# make premiums into a list because I trapped myself in my code
# in this function, the opportunity cost is calculated and then discounted
# to get the PV opportunity cost
pv_prem = pd.DataFrame(pv_opp_cost(money=np.repeat(premium, 20),
                       time_horizon=20, discount=discount,
                       safe_rate=0, alt_rate=avg_tr))

# there is the opportunity cost of the liquid gov't funds that are *unused*
# within any given year, and then there is the subsidy from the money lent
# (i.e., the part used) and that part is the foregone repayments, so different
# just using the liquid TF

# this is just the 750M TF
# note: these should be the same because usage just based on reserves + raw NR (they are)
# but can read them in regardless to check

used_tf_all = pd.read_csv("Results//used_tf2_" + loc_all_suff + '.csv').iloc[1:,1]
used_tf_all = pd.DataFrame(np.reshape(used_tf_all.values,(20,int(len(crac_no_loc)/20)), order = 'F'))
used_tf0 = pd.read_csv("Results//used_tf2_" + loc0_suff + '.csv').iloc[1:,1]
used_tf0 = pd.DataFrame(np.reshape(used_tf0.values,(20,int(len(crac_no_loc)/20)), order = 'F'))

held_tf0 = (750*pow(10, 6) - used_tf0).mean(axis=1)
held_tf_all = (750*pow(10, 6) - used_tf_all).mean(axis=1)

# these are all the same because no matter what repay,
# the TF is available
# under the unlimited scenario
opp_cost_tf0 = pd.DataFrame(pv_opp_cost(held_tf0, time_horizon=1, discount=discount,
                                        safe_rate=0, alt_rate=avg_tr))
opp_cost_tf_all = pd.DataFrame(pv_opp_cost(held_tf_all, time_horizon=1, discount=discount,
                                          safe_rate=0, alt_rate=avg_tr))


###############################################################################
# AMORTIZATION SCHEDULE: i.e., expected repayment to repay all BA
# for now, ignoring the further 30 years
def read_amort(redux, rate=rates_tr, discount=discount, time_horizon=time_horizon,
               folder = 'Results'):
    amort = pd.read_csv(f"{folder}/amort_sched_{redux}.csv").iloc[:, 1:]
    pv_amort = pv_calc(amort, discount=discount, rate=rate, modelled=False,
                       time_horizon=time_horizon)
    return amort, pv_amort


##############
# Strategy 2a (nf = non-federal rate, to use as point of comparison for interest rate sub)
amort0, amort0_pv1 = read_amort(loc0_suff, folder = 'Results/Long')
amort0_pv = amort0_pv1.iloc[:20, :].mean(axis=1)

amort0_nf, amort0_pv_nf1 = read_amort('loc0_repay_nf_10k', folder = 'Results/Long')
amort0_pv_nf = amort0_pv_nf1.iloc[:20, :].mean(axis=1)

##############
# Strategy 2b
amort_all, amort_all_pv1 = read_amort(loc_all_suff, folder = 'Results/Long')
amort_all_pv = amort_all_pv1.iloc[:20, :].mean(axis=1)

amort_all_nf, amort_all_pv_nf1 = read_amort('loc_all_repay_nf_10k', folder = 'Results/Long')
amort_all_pv_nf = amort_all_pv_nf1.iloc[:20, :].mean(axis=1)

(amort_all_pv - amort_all_pv_nf).iloc[-1]

# NOTE: also want to know how much is left at the end of the period...
# so can do PV through the end, and then subtract last year from year 20,
# which is what we have accounted for
# these should all be the same (no repayment after year 20) and they are! :)
remain_loc0 = (amort0_pv1.iloc[-1,:] - amort0_pv1.iloc[19,:]).mean()
remain_loc_all = (amort_all_pv1.iloc[-1,:] - amort_all_pv1.iloc[19,:]).mean()

print(f"Remaining, Strategy 2a: {round(remain_loc0):,}")
print(f"Remaining, Strategy 2b: {round(remain_loc_all):,}")
print()

############## total costs of CRAC -- $$ adjustments * PF sales ###############
loc_crac_all_ens = loc_all_crac.mean()
no_loc_crac_ens = no_loc_crac.mean()
ins_crac_ens = ins_crac.mean()

df_synth_load = pd.read_csv('CAPOW_data/Sim_hourly_load.csv', usecols=[1])
BPAT_load = pd.DataFrame(np.reshape(
    df_synth_load.values, (24*365, 1200), order='F'))
base = dt(2001, 1, 1)
arr = np.array([base + timedelta(hours=i) for i in range(24*365)])
BPAT_load.index = arr
BPAT_load = BPAT_load.resample('D').mean()
BPAT_load.drop([82, 150, 374, 377, 540, 616, 928, 940,
               974, 980, 1129, 1191], axis=1, inplace=True)

BPAT_load.columns = np.arange(0, len(BPAT_load.columns))
sequences = pd.DataFrame(pd.read_csv("random_sequences.csv").iloc[:,1])

BPAT_load = BPAT_load.loc[:, sequences.iloc[:,0]]  

BPAT_load = pd.DataFrame(np.reshape(BPAT_load.values, (365*len(BPAT_load.columns)), order='F'))

df_load = pd.read_excel('net_rev_data.xlsx', sheet_name='load', skiprows=[
                        0, 1])  # , usecols=[12]) ## column 12 = 2021
PF_load_y = df_load.loc[13, 'AVERAGE']

# Calculate daily BPAT proportions for demand and wind
load_ratio = BPAT_load/BPAT_load.mean()

PF_load = pd.DataFrame(PF_load_y*load_ratio)
PF_load = pd.DataFrame(np.reshape(
    PF_load.values, (365, int(len(PF_load)/365)), order='F')).sum(axis=0)
PF_load_ens = pd.DataFrame(np.reshape(PF_load.values, (int(len(PF_load)/20), 20), order='F'))

crac_costs_loc0 = (PF_load_ens * loc0_crac).mean(axis=0)
crac_costs_loc_all = (PF_load_ens * loc_all_crac).mean(axis=0)
crac_costs_no_loc = (PF_load_ens * no_loc_crac).mean(axis=0)
crac_costs_ins = (PF_load_ens * ins_crac).mean(axis=0)

crac_no_loc_range = (PF_load_ens * no_loc_crac).T
crac_loc0_range = (PF_load_ens * loc0_crac).T
crac_loc_all_range = (PF_load_ens * loc_all_crac).T
crac_loc_ins_range = (PF_load_ens * ins_crac).T

pv_crac_no_loc2 = pv_calc(crac_no_loc_range, discount=discount,
                          time_horizon=time_horizon)
pv_crac_loc02 = pv_calc(crac_loc0_range, discount=discount,
                        time_horizon=time_horizon)
pv_crac_loc_all2 = pv_calc(crac_loc_all_range, discount=discount,
                           time_horizon=time_horizon)
pv_crac_ins2 = pv_calc(crac_loc_ins_range, discount=discount,
                       time_horizon=time_horizon)

pv_crac_costs_loc0 = pv_calc(crac_costs_loc0, discount=discount,
                             time_horizon=time_horizon)
pv_crac_costs_loc_all = pv_calc(crac_costs_loc_all, discount=discount,
                                time_horizon=time_horizon)
pv_crac_costs_no_loc = pv_calc(crac_costs_no_loc, discount=discount,
                               time_horizon=time_horizon)
pv_crac_costs_ins = pv_calc(crac_costs_ins, discount=discount,
                            time_horizon=time_horizon)

################### difference between expected and repaid ####################
# discount repaid
pv_repaid_all = pv_calc(loc_all_repaid, rate=rates_tr,
                        discount=discount, time_horizon=50)

loc_used_e0 = used_tf0.copy()
loc_used0 = loc_used_e0.mean(axis = 1)

loc_used_e_all = used_tf_all.copy()
loc_used_all = loc_used_e_all.mean(axis = 1) 

loc_amort_loc0, loc_pv_loc0 = pv_calc(loc_used_e0, rate=rates_loc[1],
                                      modelled=True, time_horizon=50,
                                      discount=discount)
pv_repaid_loc_loc0 = pv_calc(loc0_repaid, rate=rates_loc[1],
                             discount=discount, time_horizon=50)

loc_amort_loc_all, loc_pv_loc_all = pv_calc(loc_used_e_all,
                                            rate=rates_loc[1], modelled=True,
                                            time_horizon=50, discount=discount)
pv_repaid_loc_loc_all = pv_calc(loc_all_repaid, rate=rates_loc[1],
                                discount=discount, time_horizon=50)

loc_pv_loc0.columns = pv_repaid_loc_loc0.columns
foregone_expanding0 = (loc_pv_loc0 - pv_repaid_loc_loc0).mean(axis=1)

loc_pv_loc_all.columns = pv_repaid_loc_loc_all.columns

foregone_expanding_all = amort_all_pv - loc_all_repaid.mean(axis = 1)

print(f"Foregone, if don't repay: ${round(foregone_expanding0.iloc[-1],0):,}")
print(f"Foregone, if repay all: ${round(foregone_expanding_all.iloc[-1],0):,}")
print()

foregone_expanding02 = loc_pv_loc0 - pv_repaid_loc_loc0
foregone_expanding_all2 = amort_all_pv1.iloc[:20,:] - pv_repaid_loc_loc_all

########################## interest rate subsidy ##############################
# these are different because you may defer more/less based on
# how much is repaid
bpa_wacc = pd.read_excel('../Hist_data/BPA_debt.xlsx', sheet_name='WAI')
ba_int_rate = bpa_wacc[bpa_wacc['Name'] == 'US Treasury ']['Average'].values[0]
nf_int_rate = bpa_wacc[bpa_wacc['Name'] ==
                       'Non-federal Total']['Average'].values[0]

interest_subsidy_nf0 = (amort0_pv_nf1 - amort0_pv1).iloc[:20,:].mean(axis=1)
interest_subsidy_nf_all = (amort_all_pv_nf1 - amort_all_pv1).iloc[:20,:].mean(axis=1)

interest_subsidy_nf02 = (amort0_pv_nf1 - amort0_pv1).iloc[:20,:]
interest_subsidy_nf_all2 = (amort_all_pv_nf1 - amort_all_pv1).iloc[:20,:]


# putting into a table for organizational purposes
strats = ['Strategy 1', 'Strategy 2a', 
          'Strategy 2b', 'Strategy 3']  

cracs = [pv_crac_costs_no_loc.iloc[-1, 0],
         pv_crac_costs_loc0.iloc[-1, 0],
         pv_crac_costs_loc_all.iloc[-1, 0],
         pv_crac_costs_ins.iloc[-1, 0]]  

subs = [0,
        interest_subsidy_nf0.iloc[-1],
        interest_subsidy_nf_all.iloc[-1],
        0]

foregone = [0,
            foregone_expanding0.iloc[-1],
            foregone_expanding_all.iloc[-1],
            0]

prems = [0,
         0,
         0,
         disc_prems.iloc[-1, 0]]

repaid = [0,
          0,
          pv_repaid_all.iloc[-1, :].mean(),
          0]

opp_costs = [0,
             opp_cost_tf0.iloc[-1, 0],
             opp_cost_tf_all.iloc[-1, 0],
             pv_prem.iloc[-1, 0]]

# tariff adjustments
# interest rate subsidy
# foregone repayment
# Strategy 1
sum_no_loc_costs = pv_crac_costs_no_loc.iloc[:, 0]  # + pv_res_no_loc.iloc[:,0]

# Strategy 2a
sum_loc0_costs = pv_crac_costs_loc0.iloc[:, 0] + \
    foregone_expanding0 +  \
    + interest_subsidy_nf0 + \
    opp_cost_tf0.iloc[:, 0]  # pv_res_loc.iloc[:,0] +

# Strategy 2b
sum_loc_all_costs = pv_crac_costs_loc_all.iloc[:, 0] + \
    foregone_expanding_all +  \
    + interest_subsidy_nf_all + \
    opp_cost_tf_all.iloc[:, 0] + \
    pv_repaid_all.mean(axis=1)  
    
# Strategy 3
sum_ins_costs = pv_crac_costs_ins.iloc[:, 0] + \
    pv_prem.iloc[:, 0]  # + pv_res_ins.iloc[:,0]


###############################################################################
################################# TABLE 3 #####################################
###################  distinguishing by different party ########################

tot_costs_bpa_no_loc = repaid[0]
tot_costs_bpa_loc0 = repaid[1]
tot_costs_bpa_loc_all = repaid[2]
tot_costs_bpa_ins = repaid[3] + opp_costs[3]

tot_costs_govt_no_loc = subs[0] + foregone[0] + opp_costs[0]
tot_costs_govt_loc0 = subs[1] + foregone[1] + opp_costs[1]
tot_costs_govt_loc_all = subs[2] + foregone[2] + opp_costs[2]
tot_costs_govt_ins = subs[3] + foregone[3]

# to customers
print(f"Costs to Customers, Strategy 1:  {round(cracs[0],-6):,}")
print(f"Costs to Customers, Strategy 2a:  {round(cracs[1], -6):,}")
print(f"Costs to Customers, Strategy 2b:  {round(cracs[2], -6):,}")
print(f"Costs to Customers, Strategy 3:  {round(cracs[3], -6):,}")
print()

# to BPA
print(f"Costs to BPA, Strategy 1:  {round(tot_costs_bpa_no_loc,-6):,}")
print(f"Costs to BPA, Strategy 2a:  {round(tot_costs_bpa_loc0, -6):,}")
print(f"Costs to BPA, Strategy 2b:  {round(tot_costs_bpa_loc_all, -6):,}")
print(f"Costs to BPA, Strategy 3:  {round(tot_costs_bpa_ins, -6):,}")
print()

# to government
print(f"Costs to Gov't', Strategy 1:  {round(tot_costs_govt_no_loc,-6):,}")
print(f"Costs to Gov't, Strategy 2a:  {round(tot_costs_govt_loc0, -6):,}")
print(f"Costs to Gov't, Strategy 2b:  {round(tot_costs_govt_loc_all, -6):,}")
print(f"Costs to Gov't, Strategy 3:  {round(tot_costs_govt_ins, -6):,}")
print()


# total
print(f"Strategy 1 Costs: {round(sum_no_loc_costs.iloc[-1],-6):,}")
print(f"Strategy 2a Costs: {round(sum_loc0_costs.iloc[-1],-6):,}")
print(f"Strategy 2c Costs: {round(sum_loc_all_costs.iloc[-1],-6):,}")
print(f"Strategy 3 Costs: {round(sum_ins_costs.iloc[-1],-6):,}")
print()

strats = pd.Series(['0. No Risk Risk Management', '1. Reserves + Surcharges', 
          'Unlimited Line of Credit + Reserves + Surcharges', 
          '2a. No Repayment', '2b. Full Repayment', 
          '3. Index Insurance + Reserves + Surcharges'])

tariff_surcharges = pd.Series([np.nan, 
                     round(cracs[0],-6)/pow(10,6), 
                     np.nan, 
                     round(cracs[1], -6)/pow(10,6), 
                     round(cracs[2], -6)/pow(10,6),
                     round(cracs[3], -6)/pow(10,6)])

loc_prems = pd.Series([np.nan, 
             round(tot_costs_bpa_no_loc,-6)/pow(10,6),
             np.nan, 
             round(tot_costs_bpa_loc0, -6)/pow(10,6),
             round(tot_costs_bpa_loc_all, -6)/pow(10,6),
             round(tot_costs_bpa_ins, -6)/pow(10,6)])

foregone_oc = pd.Series([np.nan, 
               round(tot_costs_govt_no_loc,-6)/pow(10,6),
               np.nan,  
               round(tot_costs_govt_loc0,-6)/pow(10,6), 
               round(tot_costs_govt_loc_all, -6)/pow(10,6), 
               round(tot_costs_govt_ins, -6)/pow(10,6)])

avg_totals = pd.Series([np.nan, 
               round(sum_no_loc_costs.iloc[-1],-6)/pow(10,6), 
               np.nan, 
               round(sum_loc0_costs.iloc[-1],-6)/pow(10,6),
               round(sum_loc_all_costs.iloc[-1],-6)/pow(10,6),
               round(sum_ins_costs.iloc[-1],-6)/pow(10,6)])

table3 = pd.concat([strats, 
                    tariff_surcharges, 
                    loc_prems, 
                    foregone_oc, 
                    avg_totals], axis = 1)

table3.to_csv(f"Results/table3_{discount}.csv")

###############################################################################
############################ FIG 7: Cumulative costs ##########################
###############################################################################
# one panel with cumulative costs, including reserves, CRAC, line of credit, etc
# SUM TOTAL OF ALL COSTS OF RISK MANAGEMENT TO ALL PARTIES
# cumulative costs

title_font = 10 
label_font = 10
tick_font = 12
leg_font = 12
marker_font = 14

fig, (ax3, ax5, ax1, ax2, ax4) = plt.subplots(nrows=5, sharex=True)
#plt.subplots_adjust(hspace=.3)

# A) Foregone repayment
ax1.plot(foregone_expanding0.index, foregone_expanding0,
         linestyle=':', linewidth=3, label='No Repayment', color = color_loc0)  
ax1.plot(foregone_expanding_all.index, foregone_expanding_all, color = color_loc_all,
         linestyle='--', linewidth=3, label='Maximum Repayment')  # ', Expanding')
ax1.set_xlabel("Ensemble Year", fontsize=label_font)
ax1.set_ylabel("Average Present Value, \nForegone Repayments \n($M)",
               fontsize=label_font)
ax1.set_yticks([0, 20*pow(10, 6), 40*pow(10, 6), 60*pow(10, 6)],
               ['0', '20', '40', '60'],
               fontsize=tick_font)
ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
               ['1', '3', '5', '7', '9', '11', '13',
               '15', '17', '19'], fontsize=tick_font)
ax1.legend(fontsize=leg_font, frameon=False, loc='upper left')

# C) Opportunity Cost 
ax5.plot(foregone_expanding0.index, opp_cost_tf_all.iloc[:,0],
         linestyle='--', linewidth=3, label='Line of Credit', color='#8B3D88')  
ax5.plot(foregone_expanding_all.index, pv_prem.iloc[:,0], color=color_ins,#'#37A794',
         linestyle='-.', linewidth=3, label='Index Insurance')  # ', Expanding')
ax5.set_ylabel("Average Present Value, \nOpportunity Cost \n($M)",
               fontsize=label_font)
ax5.set_yticks([0, 100*pow(10, 6), 200*pow(10, 6), 300*pow(10, 6), 400*pow(10, 6)],
               ['0', '100', '200', '300', '400'],
               fontsize=tick_font)
ax5.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
               ['1', '3', '5', '7', '9', '11', '13',
               '15', '17', '19'], fontsize=tick_font)
ax5.legend(fontsize=leg_font, frameon=False, loc='upper left')
ax5.set_xlabel("Ensemble Year", fontsize=label_font)

# B) Interest Rate Subsidy
ax2.plot(interest_subsidy_nf0.index, interest_subsidy_nf0, linewidth=3,
         color='#8B3D88', linestyle=':')#'a')
## fully overlapping as lines on the page, so showing one ( <250k differences at all times)
#ax2.plot(interest_subsidy_nf_all.index, interest_subsidy_nf_all, linewidth = 3,
#         color = '#37A794', linestyle = '--', label = 'Strategy 2b')
ax2.legend(fontsize=leg_font, frameon=False)
#ax2.set_xlabel("Ensemble Year", fontsize=label_font)
ax2.set_ylabel("Average Present Value, \nInterest Rate Subsidy \n($M)",
               fontsize=label_font)
ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
               ['1', '3', '5', '7', '9', '11', '13',
               '15', '17', '19'], fontsize=tick_font)
ax2.set_yticks([0, 8*pow(10, 6), 16*pow(10,6), 24*pow(10, 6)],
               ['0', '8', '16','24'], fontsize = tick_font)#'5', '10', '15', '20'], fontsize=tick_font)
ax2.legend(fontsize=leg_font, frameon=False, loc='upper left')
ax2.set_xlabel("Ensemble Year", fontsize=label_font)

# C) mean CRAC * load
ax3.plot(pv_crac_costs_no_loc.index, pv_crac_costs_no_loc, label='Strategy 1',
         color = color_no_loc, linestyle=':', linewidth=3)
ax3.plot(pv_crac_costs_no_loc.index, pv_crac_costs_loc_all, label='Strategies 2a and 2b',
         color='#8B3D88', linestyle='--', linewidth=3)
ax3.plot(pv_crac_costs_no_loc.index, pv_crac_costs_ins, label='Strategy 3',
         color = color_ins, linewidth=3, linestyle='-.')
ax3.set_yticks([0, 5*pow(10, 6), 10*pow(10, 6), 15*pow(10, 6), 20*pow(10, 6)],
               ['0', '5', '10', '15', '20'], fontsize=tick_font)
ax3.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
               ['1', '3', '5', '7', '9', '11', '13',
               '15', '17', '19'], fontsize=tick_font)
#ax3.set_xlabel("Ensemble Year", fontsize=label_font)
ax3.set_ylabel("Average Present Value, \nTariff Surcharges \n($M)",
               fontsize=label_font)
ax3.legend(fontsize=leg_font, frameon=False)

ax4.plot(sum_no_loc_costs.index, sum_no_loc_costs, linewidth=3,
         color = color_no_loc, linestyle=':', label='Strategy 1, Total')
ax4.plot(sum_no_loc_costs.index, sum_loc0_costs, linewidth=3,
         linestyle='--', color = color_loc0, label='Strategy 2a, Total')
ax4.plot(sum_no_loc_costs.index, sum_loc_all_costs, linewidth=2,
         linestyle='--', color = color_loc_all, label='Strategy 2b, Total')
ax4.plot(sum_no_loc_costs.index, sum_ins_costs, linewidth=3,
         color = color_ins, linestyle='-.', label='Strategy 3, Total')
# ax4.plot(np.arange(1,21), sum_ins_cap_costs, linewidth = 3,
#         color = 'purple', alpha = 0.6, linestyle = '-', label = 'Strategy 4, Total')
ax4.legend(fontsize=leg_font, frameon=False, loc = 'upper left')
ax4.set_xlabel("Ensemble Year", fontsize=label_font)
ax4.set_ylabel("Average Present \nValue Cost ($M)", fontsize=label_font)
ax4.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
               ['1', '3', '5', '7', '9', '11', '13',
               '15', '17', '19'], fontsize=tick_font)
ax4.set_yticks([0, 200*pow(10, 6), 
                400*pow(10, 6),  600*pow(10, 6)],
               ['0', '200', '400', '600'], fontsize=tick_font)

###########################################################################
######################## FIG 8: Cost Range ################################
###########################################################################
# Strategy 1
sum_no_loc_costs2 = pv_crac_no_loc2

opp_cost_tf2 = pd.concat([opp_cost_tf0]*590, axis=1)
opp_cost_tf2.columns = pv_crac_loc02.columns

# Strategy 2a
sum_loc0_costs2 = pv_crac_loc02 + \
    foregone_expanding02 +  \
    interest_subsidy_nf02 + \
    opp_cost_tf2

# Strategy 2c
sum_loc_all_costs2 = pv_crac_loc_all2 + \
    foregone_expanding_all2 +  \
    interest_subsidy_nf_all2 + \
    opp_cost_tf2 + \
    pv_repaid_all
    
# Strategy 3
pv_prem2 = pd.concat([pv_prem]*590, axis=1)
pv_prem2.columns = pv_crac_ins2.columns
sum_ins_costs2 = pv_crac_ins2 + \
    pv_prem2

sum_no_loc_costs2.to_csv(f"Results/sum_no_loc_costs2_{discount}.csv")
sum_loc0_costs2.to_csv(f"Results/sum_loc0_costs2_{discount}.csv")
sum_loc_all_costs2.to_csv(f"Results/sum_loc_all_costs2_{discount}.csv")
sum_ins_costs2.to_csv(f"Results/sum_ins_costs2_{discount}.csv")

label_font = 14
tick_font = 14
leg_font = 14
alpha = .07

fig, (ax1, ax2, ax4, ax5) = plt.subplots(ncols=4, nrows=1,
                                         sharey=True, sharex=True)

plt.subplots_adjust(hspace=.3)

ax1.plot(sum_no_loc_costs2.index, sum_no_loc_costs2, linewidth=1,
          color='gray', alpha=alpha)
ax1.plot(sum_no_loc_costs2.index, sum_no_loc_costs2.mean(axis=1), linewidth=3,
          color = color_no_loc, linestyle= (0, (1, 1)), label='Strategy 1, Mean')
ax1.set_ylabel("Present Value Cost ($M)", fontsize=label_font)

ax2.plot(sum_no_loc_costs2.index, sum_loc0_costs2, linewidth=1,
         color='gray', alpha=alpha)
ax2.plot(sum_no_loc_costs2.index, sum_loc0_costs2.mean(axis=1), linewidth=3,
         linestyle='--', color = color_loc0, label='Strategy 2a, Mean')

ax4.plot(sum_no_loc_costs2.index, sum_loc_all_costs2, linewidth=1,
         color='gray', alpha=alpha)
ax4.plot(sum_no_loc_costs2.index, sum_loc_all_costs2.mean(axis=1), linewidth=3,
         linestyle='--', color = color_loc_all, label='Strategy 2b, Mean')

ax5.plot(sum_no_loc_costs2.index, sum_ins_costs2, linewidth=1,
         color='gray', alpha=alpha)
ax5.plot(sum_no_loc_costs2.index, sum_ins_costs2.mean(axis=1), linewidth=3,
         color = color_ins, linestyle='-.', label='Strategy 3, Mean')

ax1.legend(fontsize=leg_font, frameon=False, loc = 'upper left')
ax1.set_xlabel("Ensemble Year", fontsize=label_font)
ax1.set_xticks([0, 4, 9, 14, 19],
                ['1', '5', '10', '15', '20'], fontsize=tick_font)
ax1.set_yticks([0, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#ax1.set_ylim(0, 1250*pow(10, 6))
ax1.set_ylim(0, 1500*pow(10, 6))

ax2.legend(fontsize=leg_font, frameon=False, loc = 'upper left')
ax2.set_xlabel("Ensemble Year", fontsize=label_font)
ax2.set_xticks([0, 4, 9, 14, 19],
               ['1', '5', '10', '15', '20'], fontsize=tick_font)
ax2.set_yticks([0, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#ax2.set_ylim(0, 1250*pow(10, 6))
ax2.set_ylim(0, 1500*pow(10, 6))

ax4.legend(fontsize=leg_font, frameon=False, loc = 'upper left')
ax4.set_xlabel("Ensemble Year", fontsize=label_font)
ax4.set_xticks([0, 4, 9, 14, 19],
               ['1', '5', '10', '15', '20'], fontsize=tick_font)
ax4.set_yticks([0, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#ax4.set_ylim(0, 1250*pow(10, 6))
ax4.set_ylim(0, 1500*pow(10, 6))

ax5.legend(fontsize=leg_font, frameon=False, loc = 'upper left')
ax5.set_xlabel("Ensemble Year", fontsize=label_font)
ax5.set_xticks([0, 4, 9, 14, 19],
               ['1', '5', '10', '15', '20'], fontsize=tick_font)
ax5.set_yticks([0, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#ax5.set_ylim(0, 1250*pow(10, 6))
ax5.set_ylim(0, 1500*pow(10, 6))

print("Min/max CRACs:")
print(f"Strategy 1: {pv_crac_no_loc2.iloc[-1,:].min()}; {int(pv_crac_no_loc2.iloc[-1,:].max()):,}")
print(f"Strategy 2a: {pv_crac_loc02.iloc[-1,:].min()}; {int(pv_crac_loc02.iloc[-1,:].max()):,}")
print(f"Strategy 2b: {pv_crac_loc_all2.iloc[-1,:].min()}; {int(pv_crac_loc_all2.iloc[-1,:].max()):,}")
print(f"Strategy 3: {pv_crac_ins2.iloc[-1,:].min()}; {int(pv_crac_ins2.iloc[-1,:].max()):,}")
print()

print("Min/max subsidy:")
print(f"Strategy 2a: {interest_subsidy_nf02.iloc[-1,:].min(axis = 0)}; {int(interest_subsidy_nf02.iloc[-1,:].max(axis = 0)):,}")
print(f"Strategy 2b: {interest_subsidy_nf_all2.iloc[-1,:].min(axis = 0)}; {int(interest_subsidy_nf_all2.iloc[-1,:].max(axis = 0)):,}")
print()

print("Min/max foregone:")
print(f"Strategy 2a: {int(foregone_expanding02.iloc[-1,:].min(axis = 0)):,}; {int(foregone_expanding02.iloc[-1,:].max(axis = 0)):,}")
print(f"Strategy 2b: {int(foregone_expanding_all2.iloc[-1,:].min(axis = 0)):,}; {int(foregone_expanding_all2.iloc[-1,:].max(axis = 0)):,}")
print()

print("Maximum repaid: , minimum repaid: ")
print(f"Strategy 2a: {pv_repaid_all.iloc[-1, :].min(axis = 0)}; {int(pv_repaid_all.iloc[-1, :].max(axis = 0)):,}")

print()
print(f"Minimum 1: {int(sum_no_loc_costs2.iloc[-1,:].min()):,}, Maximum 1: {int(sum_no_loc_costs2.iloc[-1,:].max()):,}")
print(f"Minimum 2a: {int(sum_loc0_costs2.iloc[-1,:].min()):,}, Maximum 2a: {int(sum_loc0_costs2.iloc[-1,:].max()):,}")
print(f"Minimum 2b: {int(sum_loc_all_costs2.iloc[-1,:].min()):,}, Maximum 2b: {int(sum_loc_all_costs2.iloc[-1,:].max()):,}")
print(f"Minimum 3: {int(sum_ins_costs2.iloc[-1,:].min()):,}, Maximum 3: {int(sum_ins_costs2.iloc[-1,:].max()):,}")

###############################################################################
############################### TABLE 2 #######################################
###############################################################################

# CRAC mean values
loc0_crac_mean = loc0_crac.mean().mean()
loc_all_crac_mean = loc_all_crac.mean().mean()
no_loc_crac_mean = no_loc_crac.mean().mean()
ins_crac_mean = ins_crac.mean().mean()

# comparison to MidC prices
PF = pd.read_excel('net_rev_data.xlsx', sheet_name='PF_rates',
                   skiprows=np.arange(13, 31), usecols=[0, 7])
PF = PF.iloc[:11, 1].mean()

pct_ens_no_loc = (no_loc_crac.max(axis=1) > 0).sum()/len(no_loc_crac)
pct_ens_loc0 = (loc0_crac.max(axis=1) > 0).sum()/len(no_loc_crac)
pct_ens_loc_all = (loc_all_crac.max(axis=1) > 0).sum()/len(no_loc_crac)
pct_ens_ind = (ins_crac.max(axis=1) > 0).sum()/len(no_loc_crac)

print(f"Strategy 1 CRAC mean: {round((no_loc_crac_mean),4)}")
print(
    f"Strategy 1 CRAC as % of PF prices: {round((no_loc_crac_mean/PF.mean().mean())*100,2)}%")

print(f"Strategy 2a CRAC mean: {round((loc0_crac_mean),4)}")
print(
    f"Strategy 2a CRAC as % of PF prices: {round((loc0_crac_mean/PF.mean().mean())*100,2)}%")

print(f"Strategy 2b CRAC mean: {round((loc_all_crac_mean),4)}")
print(
    f"Strategy 2b CRAC as % of PF prices: {round((loc_all_crac_mean/PF.mean().mean())*100,2)}%")

print(f"Strategy 3 CRAC mean: {round((ins_crac_mean),4)}")

print(
    f"Strategy 3 CRAC as % of PF prices: {round((ins_crac_mean/PF.mean().mean())*100,2)}%")
print()

print(f"Strategy 1 Ensembles with CRAC: {round(pct_ens_no_loc*100,0)}%")
print(f"Strategy 2a Ensembles with CRAC: {round(pct_ens_loc0*100,0)}%")
print(f"Strategy 2b Ensembles with CRAC: {round(pct_ens_loc_all*100,0)}%")
print(f"Strategy 3 Ensembles with CRAC: {round(pct_ens_ind*100,0)}%")

# average NR
print(f"Strategy 0 Avg NR: {round(no_tools.mean(),-5):,}")
print(f"Strategy 1 Avg NR: {round(nr_no_loc.mean(),-5):,}")
print(f"Strategy 2a Avg NR: {round(nr_loc0.mean(),-1):,}")
print(f"Strategy 2c Avg NR: {round(nr_loc_all.mean(),-1):,}")
print(f"Strategy 3 Avg NR: {round(nr_ins.mean(),-5):,}")
print()

# Expected Shortfalls
print(
    f"Expected Shortfall Strategy 0: ${no_tools[no_tools <= np.percentile(no_tools, 5)].mean():,}")
print(
    f"Expected Shortfall Strategy 1: ${nr_no_loc[nr_no_loc <= np.percentile(nr_no_loc, 5)].mean():,}")
print(
    f"Expected Shortfall Strategy 2a: ${nr_loc0[nr_loc0 <= np.percentile(nr_loc0, 5)].mean():,}")
print(
    f"Expected Shortfall Strategy 2b: ${nr_loc_all[nr_loc_all <= np.percentile(nr_loc_all, 5)].mean():,}")
print(
    f"Expected Shortfall Strategy 3: ${nr_ins[nr_ins <= np.percentile(nr_ins, 5)].mean():,}")
print()

# 95% VaR
print(f"Strategy 0 95% VaR: {round(np.percentile(no_tools,5),-5):,}")
print(f"Strategy 1 95% VaR: {round(np.percentile(nr_no_loc,5),-5):,}")
print(f"Strategy 2a 95% VaR: {round(np.percentile(nr_loc0,5),-5):,}")
print(f"Strategy 2c 95% VaR: {round(np.percentile(nr_loc_all,5),-5):,}")
print(f"Strategy 3 95% VaR: {round(np.percentile(nr_ins,5),-5):,}")
print()
print()

avg_nrs = pd.Series ([round(no_tools.mean()/pow(10,6)),
                      round(nr_no_loc.mean()/pow(10,6)),
                      np.nan, 
                      round(nr_loc0.mean()/pow(10,6)), 
                      round(nr_loc_all.mean()/pow(10,6)), 
                      round(nr_ins.mean()/pow(10,6))])

cvars = pd.Series([round(no_tools[no_tools <= np.percentile(no_tools, 5)].mean()/pow(10,6)),
                    round(nr_no_loc[nr_no_loc <= np.percentile(nr_no_loc, 5)].mean()/pow(10,6)),
                    np.nan,
                    round(nr_loc0[nr_loc0 <= np.percentile(nr_loc0, 5)].mean()/pow(10,6)),
                    round(nr_loc_all[nr_loc_all <= np.percentile(nr_loc_all, 5)].mean()/pow(10,6)),
                    round(nr_ins[nr_ins <= np.percentile(nr_ins, 5)].mean()/pow(10,6))])

ens_crac_pcts = pd.Series([np.nan, 
                        round(pct_ens_no_loc*100),
                        np.nan, 
                        round(pct_ens_loc0*100),
                        round(pct_ens_loc_all*100), 
                        round(pct_ens_ind*100)])

mean_surcharge = pd.Series([np.nan, 
                            
                            [round((no_loc_crac_mean),2),
                             round((no_loc_crac_mean/PF.mean().mean())*100,2)],
                            
                            np.nan, 
                            
                            [round((loc0_crac_mean),2),
                             round((loc0_crac_mean/PF.mean().mean())*100,2)],
                            
                            [round((loc_all_crac_mean),2),
                             round((loc_all_crac_mean/PF.mean().mean())*100,2)],
                            
                            [round((ins_crac_mean),4),
                             round((ins_crac_mean/PF.mean().mean())*100,2)]
                            
                            ])

table2 = pd.concat([strats, 
                    avg_nrs, 
                    cvars,
                    ens_crac_pcts,
                    mean_surcharge], axis = 1)













