# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 09:35:53 2022

@author: rcuppari
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

###############################################################################
############################ FIG. 1: AUTOCORRELATIONS #########################
###############################################################################

tda = pd.read_csv("Hist_data/TDA6ARF_daily.csv")
tda['date'] = pd.to_datetime(tda.iloc[:,0])
tda['year'] = tda['date'].dt.year
ann_tda = tda.groupby('year').mean()

hist_temp = pd.read_csv("Hist_data/all_hist_temp_wind.csv")
seattle = hist_temp[hist_temp['NAME'] == 'SEATTLE TACOMA INTERNATIONAL AIRPORT, WA US']
boise = hist_temp[hist_temp['NAME'] == 'BOISE AIR TERMINAL, ID US']

seattle['TAVG'] = pd.to_numeric(seattle['TAVG'], errors = 'coerce')
boise['TAVG'] = pd.to_numeric(boise['TAVG'], errors = 'coerce')

ann_seat = seattle[['Year', 'TAVG']].groupby('Year').mean()
ann_boise = boise[['Year', 'TAVG']].groupby('Year').mean()

## stochastic 
stoch_flows = pd.read_csv("CAPOW_data/Synthetic_streamflows_FCRPS.csv").iloc[:,1:]
names = pd.read_excel("CAPOW_data/BPA_name.xlsx")
stoch_flows.columns = names.iloc[0,:]
stoch_tda = stoch_flows['Dalls ARF']
stoch_temp = pd.read_csv("CAPOW_data/synthetic_weather_data.csv")
stoch_boise = stoch_temp['BOISE_T']
stoch_seat = stoch_temp['SEATTLE_T']

ann_stoch_tda = stoch_tda.groupby(stoch_tda.index // 365).mean()
ann_stoch_boise = stoch_boise.groupby(stoch_boise.index // 365).mean()
ann_stoch_seat = stoch_seat.groupby(stoch_seat.index // 365).mean()

## shuffled 
sequences = pd.DataFrame(pd.read_csv("random_sequences.csv").iloc[:,1])
shuff_tda = ann_stoch_tda[sequences.iloc[:,0]]  
shuff_seat = ann_stoch_seat[sequences.iloc[:,0]]  
shuff_boi = ann_stoch_boise[sequences.iloc[:,0]]  

from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox
from statsmodels.stats.diagnostic import acorr_breusch_godfrey as acorr_bg

lag = 10 

ljungbox_tda = ljungbox(ann_tda.iloc[:,0], lags = lag, boxpierce = True)
ljungbox_seat = ljungbox(ann_seat.iloc[:,0], lags = lag, boxpierce = True)
ljungbox_boi = ljungbox(ann_boise.iloc[:,0], lags = lag, boxpierce = True)

ljungbox_stoch_tda = ljungbox(ann_stoch_tda, lags = lag, boxpierce = True)
ljungbox_stoch_seat = ljungbox(ann_stoch_seat, lags = lag, boxpierce = True)
ljungbox_stoch_boi = ljungbox(ann_stoch_boise, lags = lag, boxpierce = True)

ljungbox_shuff_tda = ljungbox(shuff_tda, lags = lag, boxpierce = True)
ljungbox_shuff_seat = ljungbox(shuff_seat, lags = lag, boxpierce = True)
ljungbox_shuff_boi = ljungbox(shuff_boi, lags = lag, boxpierce = True)

from statsmodels.graphics.tsaplots import plot_acf

fig, axs = plt.subplots(3,4, sharex = True)
plot_acf(ann_tda, lags = lag, ax = axs[0,0], label = 'Historical')
axs[0,0].set_ylim(-0.5, 1.05)
axs[0,0].set_title("")

plot_acf(ann_stoch_tda, lags = lag, ax = axs[0,1], label = 'Stochastic')
axs[0,1].set_ylim(-0.5, 1.05)
axs[0,1].set_title("")

plot_acf(shuff_tda, lags = lag, ax = axs[0,2], label = 'Stochastic')
axs[0,2].set_ylim(-0.5, 1.05)
axs[0,2].set_title("")

axs[0,3].plot(ljungbox_tda['bp_pvalue'], label = 'Historical')
axs[0,3].plot(ljungbox_stoch_tda['bp_pvalue'], label = 'Stochastic')
axs[0,3].plot(ljungbox_shuff_tda['bp_pvalue'], label = 'Shuffled')
axs[0,3].set_title("")
axs[0,3].legend(frameon = False, fontsize = 13) 
axs[0,3].set_ylim(0, 1.05)

plot_acf(ann_boise.iloc[:,0], lags = lag, ax = axs[1,0], label = 'Historical')
axs[1,0].set_title("")
axs[1,0].set_ylim(-0.5, 1.05)

plot_acf(ann_stoch_boise, lags = lag, ax = axs[1,1], label = 'Stochastic')
axs[1,1].set_title("")
axs[1,1].set_ylim(-0.5, 1.05)

plot_acf(shuff_boi, lags = lag, ax = axs[1,2], label = 'Shuffled')
axs[1,2].set_title("")
axs[1,2].set_ylim(-0.5, 1.05)

axs[1,3].plot(ljungbox_boi['bp_pvalue'], label = 'Historical')
axs[1,3].plot(ljungbox_stoch_boi['bp_pvalue'], label = 'Stochastic')
axs[1,3].plot(ljungbox_shuff_boi['bp_pvalue'], label = 'Shuffled')
axs[1,3].set_ylim(0, 1.05)

plot_acf(ann_seat.iloc[:,0], lags = lag, ax = axs[2,0], label = 'Historical')
axs[2,0].set_title("")
axs[2,0].set_ylim(-0.75, 1.05)
axs[2,0].set_xlabel("Lag", fontsize = 14)

plot_acf(ann_stoch_seat, lags = lag, ax = axs[2,1], label = 'Stochastic')
axs[2,1].set_title("")
axs[2,1].set_ylim(-0.75, 1.05)
axs[2,1].set_xlabel("Lag", fontsize = 14)

plot_acf(shuff_seat, lags = lag, ax = axs[2,2], label = 'Shuffled')
axs[2,2].set_title("")
axs[2,2].set_ylim(-0.75, 1.05)
axs[2,2].set_xlabel("Lag", fontsize = 14)

axs[2,3].plot(ljungbox_seat['bp_pvalue'], label = 'Historical')
axs[2,3].plot(ljungbox_stoch_seat['bp_pvalue'], label = 'Stochastic')
axs[2,3].plot(ljungbox_shuff_seat['bp_pvalue'], label = 'Shuffled')
axs[2,3].set_title("")
axs[2,3].set_xlabel("Lag", fontsize = 14)
axs[2,3].set_ylim(0, 1.05)

# Add column headers
col_headers = ['Historical', 'Stochastic \n(1,180 years)', \
               'Shuffled \n(11,800 years)', 'Box-Pierce P-Values']
fig.text(0.155, 0.96, col_headers[0], ha='center', va = 'center', fontsize = 14)
fig.text(0.4, 0.965, col_headers[1], ha='center', va = 'center', fontsize = 14)
fig.text(0.64, 0.965, col_headers[2], ha='center', va = 'center', fontsize = 14)
fig.text(0.89, 0.96, col_headers[3], ha='center', va = 'center', fontsize = 14)

# Add row headers
row_headers = ['Streamflow, \nThe Dalles', 'Temperature, \nBoise (ID)', \
               'Temperature, \nSeattle (OR)']
fig.text(0.02, 0.82, row_headers[0], ha='center', va='center', rotation='vertical', fontsize = 14)
fig.text(0.02, 0.5, row_headers[1], ha='center', va='center', rotation='vertical', fontsize = 14)
fig.text(0.02, 0.18, row_headers[2], ha='center', va='center', rotation='vertical', fontsize = 14)

plt.subplots_adjust(top=0.94, left=0.06, bottom=0.05, right = 0.99)

###############################################################################
########################## FIG. 2: SENSITIVITY ANALYSIS #######################
###############################################################################

## discount 0
no_loc_costs_disc0 = pd.read_csv("Results/sum_no_loc_costs2_0.csv")
loc0_costs_disc0 = pd.read_csv("Results/sum_loc0_costs2_0.csv")
loc_all_costs_disc0 = pd.read_csv("Results/sum_loc_all_costs2_0.csv")
ins_costs_disc0 = pd.read_csv("Results/sum_ins_costs2_0.csv")

summary_disc0 = pd.read_csv("Results/table3_0.csv")

## discount 5
no_loc_costs_disc5 = pd.read_csv("Results/sum_no_loc_costs2_5.csv")
loc0_costs_disc5 = pd.read_csv("Results/sum_loc0_costs2_5.csv")
loc_all_costs_disc5 = pd.read_csv("Results/sum_loc_all_costs2_5.csv")
ins_costs_disc5 = pd.read_csv("Results/sum_ins_costs2_5.csv")

summary_disc5 = pd.read_csv("Results/table3_5.csv")

label_font = 14
tick_font = 14
leg_font = 14
alpha = .07

## color coordination
color_no_loc = 'darkorange'
color_loc0 = 'gold'
color_loc_all = 'olivedrab'
color_ins = '#71AFE2'

fig, axs = plt.subplots(ncols=4, nrows=2, 
                        sharey=True, sharex=True)

plt.subplots_adjust(hspace=.3)

############### discount rate 0%
axs[0,0].plot(no_loc_costs_disc0.index, no_loc_costs_disc0, linewidth=1,
          color='gray', alpha=alpha)
axs[0,0].plot(no_loc_costs_disc0.index, no_loc_costs_disc0.mean(axis=1), linewidth=3,
          color = color_no_loc, linestyle= (0, (1, 1)), label='Strategy 1, Mean')
axs[0,0].set_ylabel("Present Value Cost ($M)", fontsize=label_font)

axs[0,1].plot(no_loc_costs_disc0.index, loc0_costs_disc0, linewidth=1,
         color='gray', alpha=alpha)
axs[0,1].plot(no_loc_costs_disc0.index, loc0_costs_disc0.mean(axis=1), linewidth=3,
         linestyle='--', color = color_loc0, label='Strategy 2a, Mean')

axs[0,2].plot(no_loc_costs_disc0.index, loc_all_costs_disc0, linewidth=1,
         color='gray', alpha=alpha)
axs[0,2].plot(no_loc_costs_disc0.index, loc_all_costs_disc0.mean(axis=1), linewidth=3,
         linestyle='--', color = color_loc_all, label='Strategy 2b, Mean')

axs[0,3].plot(no_loc_costs_disc0.index, ins_costs_disc0, linewidth=1,
         color='gray', alpha=alpha)
axs[0,3].plot(no_loc_costs_disc0.index, ins_costs_disc0.mean(axis=1), linewidth=3,
         color = color_ins, linestyle='-.', label='Strategy 3, Mean')

axs[0,0].legend(fontsize=leg_font, frameon=False, loc = 'upper left')
axs[0,0].set_xlabel("Ensemble Year", fontsize=label_font)
axs[0,0].set_xticks([0, 4, 9, 14, 19],
                ['1', '5', '10', '15', '20'], fontsize=tick_font)
axs[0,0].set_yticks([0, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#axs[0,0].set_ylim(0, 1250*pow(10, 6))
axs[0,0].set_ylim(0, 1500*pow(10, 6))

axs[0,1].legend(fontsize=leg_font, frameon=False, loc = 'upper left')
axs[0,1].set_xlabel("Ensemble Year", fontsize=label_font)
axs[0,1].set_xticks([0, 4, 9, 14, 19],
               ['1', '5', '10', '15', '20'], fontsize=tick_font)
axs[0,1].set_yticks([0, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#axs[0,1].set_ylim(0, 1250*pow(10, 6))
axs[0,1].set_ylim(0, 1500*pow(10, 6))

axs[0,2].legend(fontsize=leg_font, frameon=False, loc = 'upper left')
axs[0,2].set_xlabel("Ensemble Year", fontsize=label_font)
axs[0,2].set_xticks([0, 4, 9, 14, 19],
               ['1', '5', '10', '15', '20'], fontsize=tick_font)
axs[0,2].set_yticks([0, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#axs[0,2].set_ylim(0, 1250*pow(10, 6))
axs[0,2].set_ylim(0, 1500*pow(10, 6))

axs[0,3].legend(fontsize=leg_font, frameon=False, loc = 'upper left')
axs[0,3].set_xlabel("Ensemble Year", fontsize=label_font)
axs[0,3].set_xticks([0, 4, 9, 14, 19],
               ['1', '5', '10', '15', '20'], fontsize=tick_font)
axs[0,3].set_yticks([0, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#axs[0,3].set_ylim(0, 1250*pow(10, 6))
axs[0,3].set_ylim(0, 1500*pow(10, 6))


############### discount rate 5%
axs[1,0].plot(no_loc_costs_disc0.index, no_loc_costs_disc5, linewidth=1,
          color='gray', alpha=alpha)
axs[1,0].plot(no_loc_costs_disc0.index, no_loc_costs_disc5.mean(axis=1), linewidth=3,
          color = color_no_loc, linestyle= (0, (1, 1)), label='Strategy 1, Mean')
axs[1,0].set_ylabel("Present Value Cost ($M)", fontsize=label_font)

axs[1,1].plot(no_loc_costs_disc0.index, loc0_costs_disc5, linewidth=1,
         color='gray', alpha=alpha)
axs[1,1].plot(no_loc_costs_disc0.index, loc0_costs_disc5.mean(axis=1), linewidth=3,
         linestyle='--', color = color_loc0, label='Strategy 2a, Mean')

axs[1,2].plot(no_loc_costs_disc0.index, loc_all_costs_disc5, linewidth=1,
         color='gray', alpha=alpha)
axs[1,2].plot(no_loc_costs_disc0.index, loc_all_costs_disc5.mean(axis=1), linewidth=3,
         linestyle='--', color = color_loc_all, label='Strategy 2b, Mean')

axs[1,3].plot(no_loc_costs_disc0.index, ins_costs_disc5, linewidth=1,
         color='gray', alpha=alpha)
axs[1,3].plot(no_loc_costs_disc0.index, ins_costs_disc5.mean(axis=1), linewidth=3,
         color = color_ins, linestyle='-.', label='Strategy 3, Mean')

axs[1,0].legend(fontsize=leg_font, frameon=False, loc = 'upper left')
axs[1,0].set_xlabel("Ensemble Year", fontsize=label_font)
axs[1,0].set_xticks([1, 4, 9, 14, 19],
                ['1', '5', '10', '15', '20'], fontsize=tick_font)
axs[1,0].set_yticks([1, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#axs[1,0].set_ylim(0, 1250*pow(10, 6))
axs[1,0].set_ylim(0, 1500*pow(10, 6))

axs[1,1].legend(fontsize=leg_font, frameon=False, loc = 'upper left')
axs[1,1].set_xlabel("Ensemble Year", fontsize=label_font)
axs[1,1].set_xticks([1, 4, 9, 14, 19],
               ['1', '5', '10', '15', '20'], fontsize=tick_font)
axs[1,1].set_yticks([1, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#axs[1,1].set_ylim(0, 1250*pow(10, 6))
axs[1,1].set_ylim(0, 1500*pow(10, 6))

axs[1,2].legend(fontsize=leg_font, frameon=False, loc = 'upper left')
axs[1,2].set_xlabel("Ensemble Year", fontsize=label_font)
axs[1,2].set_xticks([1, 4, 9, 14, 19],
               ['1', '5', '10', '15', '20'], fontsize=tick_font)
axs[1,2].set_yticks([1, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
#axs[1,2].set_ylim(0, 1250*pow(10, 6))
axs[1,2].set_ylim(0, 1500*pow(10, 6))

axs[1,3].legend(fontsize=leg_font, frameon=False, loc = 'upper left')
axs[1,3].set_xlabel("Ensemble Year", fontsize=label_font)
axs[1,3].set_xticks([1, 4, 9, 14, 19],
               ['1', '5', '10', '15', '20'], fontsize=tick_font)
axs[1,3].set_yticks([1, 250*pow(10, 6), 500*pow(10, 6),
                750*pow(10, 6), 1000*pow(10, 6), 1250*pow(10, 6), 1500*pow(10, 6)],
                ['0', '250', '500', '750', '1000', '1250', '1500'],
                fontsize=tick_font)
axs[1,3].set_ylim(0, 1500*pow(10, 6))

###############################################################################
############################### TABLE 2 OF SM #################################
###############################################################################
 
## combination of the following data
## drop the first 'unnamed' row
## discount 0%
table3_disc0 = pd.read_csv("Results/table3_0.csv").iloc[:,1:]

## discount 5% 
table3_disc5 = pd.read_csv("Results/table3_5.csv").iloc[:,1:]

full_sm_table3 = pd.DataFrame(index = table3_disc0.index, 
                              columns = table3_disc0.columns)

for col in table3_disc0.columns[1:]:
    print(col)
    for row in table3_disc0.index:
        full_sm_table3.loc[row, col] = f"{table3_disc0.loc[row,col]}, {table3_disc5.loc[row,col]}"

full_sm_table3.iloc[:,0] = table3_disc0.iloc[:,0]
full_sm_table3.columns = ['Strategy', 
                          'Tariff Surcharge Costs (BPA Customers) ($)', 
                          'Line of Credit Repayments and Insurance Premiums (BPA) ($M)',
                          'Foregone Repayments and Opportunity Cost (Government and Taxpayers) ($M)', 
                          'Average Total Cost ($M)']























