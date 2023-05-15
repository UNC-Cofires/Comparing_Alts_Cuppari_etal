# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:37:36 2022

@author: rcuppari
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 
from cost_calculations import pv_calc
from cost_calculations import expected_repay
from cost_calculations import plot_cum_pv
from cost_calculations import compound_interest
from cost_calculations import used
from cost_calculations import repaid
from cost_calculations import read_reserves
from cost_calculations import crac
from cost_calculations import opp_cost
from density_plots import density_scatter_plot
from density_plots import multiple_pdf

##############################################################################
##### SCRIPT USED TO MAKE ALL PLOTS IN MAIN TEXT AND SUPPLEMENTARY INFO ######
##### NOTE: figure 1 made in ArcGIS Online, and figure 2 made as PPT file ####
##############################################################################

# discount rate being used -- modify based on plots to make 
discount = 5 
strike_pct = 15 

###############################################################################
########################### read in all data ##################################

## annual net revenue results for strategies with and without full reserves 
# strategy 1
nr_no_loc = pd.read_csv("Results/ann_net_rev_no_loc2.csv").iloc[:,2]
nr_no_loc_cap = pd.read_csv("Results/ann_net_rev_no_loc_cap.csv").iloc[:,2]
# strategy 2
nr_loc = pd.read_csv("Results/ann_net_rev_loc2.csv").iloc[:,2]
nr_loc_cap = pd.read_csv("Results/ann_net_rev_loc_cap.csv").iloc[:,2]
# strategy 3
nr_ins = pd.read_csv("Results/ann_net_rev_index_no_loc15.csv").iloc[:,2]
nr_ins_cap = pd.read_csv("Results/ann_net_rev_index_cap_no_loc20.csv").iloc[:,2]


## net payouts -- the min value (the only negative one) is the premium
premium = -pd.read_csv("Results/net_payouts10.csv").iloc[:,1].min()
premium_cap = -pd.read_csv("Results/net_payouts20.csv").iloc[:,1].min()

## CRAC 
no_loc_crac, no_loc_ttp, crac_no_loc = crac("no_loc2")
loc_crac, loc_ttp, crac_loc = crac('loc2')
ins_crac, ins_ttp, crac_ind = crac('index_no_loc15')
ins_res_crac, ins_res_ttp, crac_ind_res = crac('index_cap_no_loc20')
no_loc_crac, no_loc_ttp, crac_no_loc_res = crac("no_loc_cap")
loc_cap_crac, loc_cap_ttp, crac_loc_res = crac('loc_cap')

## How much BA was used/repaid
loc_repaid, loc2 = repaid('loc2')
ins_repaid, ins_repaid2 = repaid('index_no_loc15')
ins_res_repaid, ins_res_repaid2 = repaid('index_cap_no_loc20')
no_loc_repaid, no_loc_repaid2 = repaid("no_loc2")
loc_repaid, loc_repaid2 = repaid("loc2")

## read in reserves held
ins_reserves = read_reserves('index_no_loc15')
ins_cap_reserves = read_reserves('index_cap_no_loc20') ## NOTE: with reserve cap use higher strike 
no_loc_cap_reserves = read_reserves("no_loc_cap_low") 
no_loc_reserves = read_reserves("no_loc2") 
ba_reserves = read_reserves("loc2") 
ba_cap_reserves = read_reserves("loc_cap") 

###############################################################################
######################### FIG 3 Index Performance #############################
###############################################################################

## previously had this embedded in the index making code, but have now saved 
## the predicted outputs so that they can be read in here


rev_above_index = pd.read_csv("Results/rev_above_index.csv").iloc[:,1]
rev_below_index = pd.read_csv("Results/rev_below_index.csv").iloc[:,1]

pred_above = pd.read_csv("Results/index_pred_above.csv").iloc[:,1]
pred_below = pd.read_csv("Results/index_pred_below.csv").iloc[:,1]

tda_index_above = pd.read_csv("Results/tda_index_above.csv").iloc[:,1]
tda_index_below = pd.read_csv("Results/tda_index_below.csv").iloc[:,1]

xmin = -800*pow(10,6)
xmax = 500*pow(10,6)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols = 1)
ax1.scatter(rev_above_index, tda_index_above, label = "Normal/Wet Years", 
            color = 'cornflowerblue')
ax1.scatter(rev_below_index, tda_index_below, label = "Dry Years (payout)", color = 'brown')
ax1.set_ylabel("Dalles Based Index Value ($)",fontsize=14)
ax1.set_xlabel("Net Revenues ($M)",fontsize=14)
#ax1.annotate('MAE below 15th% flow: $' + str(round(mae_tda_below/pow(10,6),2)) + " M",(-750*pow(10,6), 300*pow(10,3)),fontsize=16)
#ax1.annotate('MAE above 15th% flow: ' + str(round(mae_tda_above/pow(10,6),2)),(300*pow(10,6), -400*pow(10,6)),fontsize=16)
ax1.axhspan(xmin, 0, 0, .615, color = 'red', alpha = .1, 
                  label = 'Negative Net Revenues')
ax1.hlines(0, xmin, 0, color = 'black', linestyles = '--', linewidth = 3)#, label = '$0 Index Value')
ax1.vlines(0, xmin, 0, color = 'black', linestyles = '--', linewidth = 3)#, label = '$0 Net Revenues')
ax1.set_xticks([-800*pow(10,6),-600000000,-400000000,-200000000,0,200000000,400000000, \
                600000000, 800000000],
            ['-800', '-600','-400','-200','0','200','400', '600', '800'],
            fontsize=14)
ax1.set_yticks([-800*pow(10,6),-600000000,-400000000,-200000000,0,200000000,400000000, \
                600000000, 800000000],
            ['-800', '-600','-400','-200','0','200','400', '600', '800'],
            fontsize=14)
ax1.plot([xmin, 0], [xmin, 0], linestyle = '-.', color = 'black',
          label = '1:1 line')
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(xmin, xmax) 
#ax1.set_ylim(0, 100*pow(10,3))
ax1.legend(fontsize = 14, frameon = False, loc = 'lower right') 

ax2.scatter(rev_above_index, pred_above, label = 'Normal/Wet Years', 
            color = 'cornflowerblue')
ax2.scatter(rev_below_index, pred_below, alpha = .8, label = 'Dry Years (payout)', color = 'brown')
ax2.hlines(0, xmin, 0, color = 'black', linestyles = '--', linewidth = 3)#, label = '$0 Index Value')
ax2.vlines(0, xmin, 0, color = 'black', linestyles = '--', linewidth = 3)#, label = '$0 Net Revenues')
ax2.axhspan(xmin, 0, 0, .615, color = 'red', alpha = .1, 
                  label = 'Negative Net Revenues')
ax2.set_xlabel("Net Revenues ($M)", fontsize = 14)
ax2.set_ylabel("Index Value ($M)", fontsize = 14)
ax2.set_xticks([-800*pow(10,6), -600000000,-400000000,-200000000,0,200000000,400000000, \
                600000000, 800000000],
            ['-800','-600','-400','-200','0','200','400', '600', '800'],
            fontsize=14)
ax2.set_yticks([-800*pow(10,6),-600000000,-400000000,-200000000,0,200000000,400000000, \
                600000000, 800000000],
            ['-800', '-600','-400','-200','0','200','400', '600', '800'],
            fontsize=14)
ax2.plot([xmin, 0], [xmin, 0], linestyle = '-.', color = 'black',
          label = '1:1 line')
#ax2.annotate('MAE below 15th% flow: $' + str(round(mae_below/pow(10,6),2)) + " M",(-750*pow(10,6), 300*pow(10,6)),fontsize=16)
#ax2.annotate('MAE above 15th% flow: ' + str(round(mae_above/pow(10,6),2)),(300*pow(10,6), -400*pow(10,6)),fontsize=16)
ax2.legend(frameon = False, fontsize=14, loc = 'lower right')
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(xmin, xmax) 

###############################################################################
####################### FIG 4 Historical Rates/BA #############################
###############################################################################

## hist reserves & ba
res_ba = pd.read_excel('net_rev_data.xlsx', sheet_name = 'res_ba')
res_ba.set_index('Unnamed: 0', inplace = True)
res_ba = res_ba.T

res = np.array(res_ba.loc[2010:2021,'start_res'])
res = pd.to_numeric(res)

## historical repayment 
repaid = pd.read_excel('BPA_Historical_Data/Debt/BPA_debt.xlsx', sheet_name = 'Repayment', usecols=[0,8,9,10,11,12,13,14,15,16]).iloc[0:4,:]
repaid.set_index('Unnamed: 0', inplace = True )
repaid2 = repaid.T

## hist BA and CRAC
ba = res_ba.loc[:2022, 'BA']
ba_add = res_ba.loc[:2022,'addition']
ba_add.fillna(0, inplace = True)

for i in np.arange(2000, 2023): 
    if ba_add[i] !=0: 
        ba_add[i] = ba.max()

crac = res_ba.loc[:2022,'CRAC']
crac.replace('Y', ba.max(), inplace = True)
crac.replace('n', 0, inplace = True)
crac[2000] = 0


## remove years that don't have both us treasury wacc + non-federal 
bpa_wacc = pd.read_excel('../Hist_data/BPA_debt.xlsx').iloc[:,:-3]
bpa_wacc = bpa_wacc[(bpa_wacc['Name'] == 'Non-federal Total') | (bpa_wacc['Name'] == 'US Treasury ') |
                    (bpa_wacc['Name'] == 'Federal Spread') | (bpa_wacc['Name'] == 'Treasury Spread') |
                    (bpa_wacc['Name'] == 'Federal appropriations')]
bpa_wacc.set_index('Name', inplace = True)
wacc2 = bpa_wacc.T#*100*100 ## convert from decimal to percent and then percent to basis points (100 basis pt = 1%)
wacc2 = wacc2.iloc[1:,:]
wacc2.replace(0, np.nan, inplace=True)


fig, axs = plt.subplots(2,1)
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
         label = 'Non-federal Rates', linewidth = 4)
axs[1].plot(wacc2['Federal appropriations'], color = 'maroon', linestyle = ':', \
         label = 'Appropriations Rates', linewidth = 4)  ## 
axs[1].plot(wacc2['US Treasury '], color = '#77AADD', alpha = .8, \
         linestyle = '--', linewidth = 4, label = 'Line of Credit Rates')
axs[1].set_ylabel("Weighted Average Interest Rate (%)", fontsize = 16)
axs[1].set_xlabel("Year", fontsize = 16)
#axs[1].set_xticks([2000, 2005, 2010, 2015, 2020],
#                 ['2000', '2005', '2010', '2015', '2020'],fontsize = 14) 
axs[1].set_yticks([0, .01, .02, .03, .04, .05,  .06, .07],
           ['0', '1', '2', '3', '4', '5', '6', '7'], fontsize = 14)
# axs[0].plot(costs['Op Rev'], linestyle = '-.', color = "#90C987", linewidth = 4, label = 'Operating Revenues') 
# axs[0].plot(costs['Tot Exp'], color = "#E8601C", linewidth = 4, label = 'Expenses', alpha = .8)
axs[1].legend(fontsize = 14, frameon = False)
axs[1].set_xlabel("Year", fontsize = 16)
#axs[0].set_ylabel("$B", fontsize = 16)
#axs[0].set_yticks([2.5*pow(10,6), 3*pow(10,6), 3.5*pow(10,6), 4*pow(10,6), 4.5*pow(10,6)], 
#           ['2.5', '3', '3.5', '4', '4.5'], fontsize = 14)
axs[1].set_xticks([2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022],
                  ['2000', '2002', '2004', '2006', '2008', '2010', 
                   '2012', '2014', '2016', '2018', '2020', '2022'], fontsize = 14)
plt.setp(axs[1].get_xticklabels(), visible=True)
plt.setp(axs[0].get_xticklabels(), visible=True)


###############################################################################
####################### FIG 5 Dot density plots ###############################
###############################################################################

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
tda['TDA flow'] = "Dry"

strike = np.percentile(tda['Dalls ARF'], 5)
for i in range(0,len(tda)):
    if tda.iloc[i,0] > strike:
        tda.iloc[i,1] = "Normal"


all_data=pd.concat([nr_no_loc, nr_loc, nr_ins, \
                   nr_ins_cap, \
                    tda['TDA flow']],axis=1)

all_data.columns=['No LOC', 'LOC', 'Ins', \
                  'Ins Res', 'TDA flow']
all_data.dropna(inplace=True)

slim = all_data[['No LOC','LOC',  'Ins', 'Ins Res', \
                 #'CFD', 'CFD$200',
                 'TDA flow']]
    
full_cols=[['No LOC', 'No LOC X', 'LOC', 'LOC X', \
            'Ins', 'Ins X', \
            'INS Res', 'INS Low X', 
            #'CFD', 'CFD X', 'CFD200',' CFD 200 Cap X', \
                 'TDA flow']]

points=pd.DataFrame(index=slim.index,columns=full_cols)
    
density_scatter_plot(slim, tda, points, same_plot = True, var = 99) 

###############################################################################
######################## FIG 6 Opportunity Costs ##############################
###############################################################################

## remove years that don't have both us treasury wacc + non-federal 
bpa_wacc = pd.read_excel('../Hist_data/BPA_debt.xlsx').iloc[:,:-3]
bpa_wacc = bpa_wacc[(bpa_wacc['Name'] == 'Non-federal Total') | (bpa_wacc['Name'] == 'US Treasury ') |
                    (bpa_wacc['Name'] == 'Federal Spread') | (bpa_wacc['Name'] == 'Treasury Spread') | \
                    (bpa_wacc['Name'] == 'Federal appropriations')]
bpa_wacc.set_index('Name', inplace = True)
wacc2 = bpa_wacc.T#*100*100 ## convert from decimal to percent and then percent to basis points (100 basis pt = 1%)
wacc2 = wacc2.iloc[1:,:]


#################### calculate the pv/value differences #######################
## need to read in the repayment 
t_rates = pd.read_csv('../Hist_data/bonds_30year.csv')
t_rates['date'] = pd.to_datetime(t_rates['DATE'])
t_rates['year'] = t_rates['date'].dt.year
t_rates = t_rates[['year', 'DGS30']]
t_rates = t_rates[t_rates['DGS30'] != '.']
t_rates['DGS30'] = pd.to_numeric(t_rates['DGS30'])

ann_bond = t_rates.groupby('year').mean()

## expected_interest gives what each year should repay, given usage in a given year
## and years prior, and a 20 year repayment period 
## used should be a df with columns for ensembles and rows for years 
## ASSUMPTION: not taking into consideration the pre-start used, just for
## what is used within the modelling

loc_used_e, loc_used = used('loc2')

## CUMULATIVE NET PRESENT VALUE
## calculate pv expected based on the average non-federal return 
## given the used value 
## rates should be a column with 20 (1/ensemble year)

non_fed = wacc2['Non-federal Total'].mean()#/100/100
rates_non_fed = [non_fed] * 20

app = wacc2['Federal appropriations'].mean()#/100/100
rates_app = [app] * 20

loc_tr = wacc2['US Treasury '].mean()
rates_loc = [loc_tr]*20

###############################################################################
############################# FIG 6 ?? ##################################
###############################################################################
################################# FIG 6 #######################################
## two panel plot with opportunity costs for 
## reserves (across three strategies)
## premium (across high/low res)

## when it's used, you amortize and then continue piling on top future
## amortized amounts
loc_amort_nf, loc_pv_nf = pv_calc(loc_used_e, rate = non_fed, modelled = True,
                                      time_horizon = 50, discount = discount)
loc_pv_nf2 = pd.DataFrame(np.reshape(loc_pv_nf.values, (20*59), order='F'))

## calculate pv expected based on the average Treasury return for BPA
treas = wacc2['US Treasury '].mean()#/100/100
rates_tr = [treas] * 20

loc_amort_tr, loc_pv_tr = pv_calc(loc_used_e, rate = treas, modelled = True,
                                        time_horizon = 50, discount = discount)
loc_pv_tr2 = pd.DataFrame(np.reshape(loc_pv_tr.values, (20*59), order='F'))

## calculate pv expected based on the average 30 year rate
thirty_rate = ann_bond.mean()/100
rates30 = [thirty_rate] * 20

## calculate pv expected based on the average fed appropriations rates
loc_amort_app, loc_pv_app = pv_calc(loc_used_e, rate = app, modelled = True,
                                        time_horizon = 50, discount = discount)
loc_pv_app2 = pd.DataFrame(np.reshape(loc_pv_app.values, (20*59), order='F'))

## calculate pv expected based on the average BA Treasury rates
loc_amort_loc, loc_pv_loc = pv_calc(loc_used_e, rate = loc_tr, modelled = True,
                                        time_horizon = 50, discount = discount)
loc_pv_loc2 = pd.DataFrame(np.reshape(loc_pv_loc.values, (20*59), order='F'))

###############################################################################
################### difference between expected and repaid ####################
###############################################################################
## discount repaid
pv_repaid_loc_app = pv_calc(loc_repaid, rate = app, discount = discount) 
pv_repaid_loc_nf = pv_calc(loc_repaid, rate =  non_fed, discount = discount) 
pv_repaid_loc_loc = pv_calc(loc_repaid, rate =  loc_tr, discount = discount)

## if the rate were 0, just purely repaid
diff_loc_nf =  pv_repaid_loc_nf - loc_pv_nf 
diff_loc_app = pv_repaid_loc_app - loc_pv_app
diff_loc_loc = pv_repaid_loc_loc - loc_pv_loc

foregone_expanding = diff_loc_loc.mean(axis = 1)
repaid_means = loc_repaid.mean(axis = 1)

######## going to use treasury rates as the alternative, safe investment ######
        ######### use bond rates to get opp costs: alt. earnings ########

## pv of premium payments
t_rates2000s = t_rates[t_rates['year'] >= 2000]
avg_tr = t_rates2000s['DGS30'].mean()
rates_tr = pd.DataFrame(np.repeat(avg_tr/100,20))

premium_ = pd.DataFrame(np.repeat(premium, 20))

## "rates" here doesn't matter because pv calc just uses the discount rate
## it was for the amortization function
disc_prems = pv_calc(premium_, discount = discount)

premium_cap_ = pd.DataFrame(np.repeat(premium_cap, 20))
disc_prems_cap = pv_calc(premium_cap_, discount = discount)

tot_gov_cost_nf = abs(foregone_expanding) + (loc_pv_nf.mean(axis = 1) - loc_pv_loc.mean(axis = 1))
tot_gov_cost_app = abs(foregone_expanding) + (loc_pv_app.mean(axis = 1) - loc_pv_loc.mean(axis = 1))

tot_gov_cost_avg = (tot_gov_cost_nf + tot_gov_cost_app)/2


## start by finding the difference in reserves held per strategy 
## with and without reserves held
diff_res_loc = (ba_reserves - ba_cap_reserves).mean(axis = 1)
diff_res_no_loc = (no_loc_reserves - no_loc_cap_reserves).mean(axis = 1)
diff_res_ins = (ins_reserves - ins_cap_reserves).mean(axis = 1)

## this is very simplistic, but for now we take the difference in 
## reserves held and just calculate compound interest 
## with the reserves rate: 0.67% 
## versus the tr rate 
## NOTE: this is not discounted
opp_res_loc = opp_cost(money = diff_res_loc, time_horizon = 20, 
                       safe_rate = 0.67, alt_rate = avg_tr)
opp_res_no_loc = opp_cost(money = diff_res_no_loc, time_horizon = 20, 
                       safe_rate = 0.67, alt_rate = avg_tr)
opp_res_ins = opp_cost(money = diff_res_ins, time_horizon = 20, 
                       safe_rate = 0.67, alt_rate = avg_tr)

## put another way... 
ins_cap_reserves2 = ins_cap_reserves.mean(axis = 1)
ins_reserves2 = ins_reserves.mean(axis = 1)

opp_cost_res_ins = pd.DataFrame(opp_cost(money = ins_reserves2, time_horizon = 20, 
                       safe_rate = 0.67, alt_rate = avg_tr))
opp_cost_res_ins_cap = pd.DataFrame(opp_cost(money = ins_cap_reserves2, time_horizon = 20, 
                       safe_rate = 0.67, alt_rate = avg_tr))
diff_res_cap_ins = opp_cost_res_ins - opp_cost_res_ins_cap 

## make premiums into a list because I trapped myself in my code
opp_prem = opp_cost(money = np.repeat(premium,20), time_horizon = 20, 
                       safe_rate = 0.67, alt_rate = avg_tr)
opp_prem_res = opp_cost(money = np.repeat(premium_cap,20), time_horizon = 20, 
                       safe_rate = 0.67, alt_rate = avg_tr)

## so now we discount this, and sum it to get the cumulative PV
pv_res_loc = pv_calc(opp_res_loc, discount = discount)
pv_res_no_loc = pv_calc(opp_res_no_loc, discount = discount)
pv_res_ins = pv_calc(opp_res_ins, discount = discount)

pv_prem = pv_calc(opp_prem, discount = discount)
pv_prem_cap = pv_calc(opp_prem_res, discount = discount)

label_font = 14
tick_font = 12
leg_font = 14
    
fig, (ax1, ax2) = plt.subplots(2, 1)
## A) Reserves: opportunity costs for 120 days CoH versus 90 days CoH
ax1.plot(np.arange(1,21), pv_res_no_loc, color = '#F1932D', alpha = .8, 
         linewidth = 3, linestyle = '-.', label = 'Strategy 1')
ax1.plot(np.arange(1,21), pv_res_loc, color = '#73B761', 
         linewidth = 3, linestyle = '--', label = 'Strategy 2')
ax1.plot(np.arange(1,21), pv_res_ins, color = '#71AFE2',
         linewidth = 3, linestyle = ':', label = 'Strategy 3')
ax1.legend(fontsize = leg_font, frameon = False)
ax1.set_xlabel("Ensemble Year", fontsize = label_font)
ax1.set_ylabel("PV Opportunity Costs \n of Reserves ($M)", fontsize = label_font)
ax1.set_yticks([0, 500*pow(10,6), 1*pow(10,9),1.5*pow(10,9), 2*pow(10,9)], 
               ["0", '500', '1000', '1500', "2000"], fontsize = tick_font)
ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              ['0', '2', '4', '6', '8', '10', '12', 
                '14', '16', '18', '20'], fontsize = tick_font)


## B) Insurance: opportunity cost of premium, a simple line
ax2.plot(np.arange(1,21), pv_prem, color = '#8B3D88', 
         linewidth = 3, linestyle = '--', label = 'Strategy 3 Premium (10th% Strike)')
ax2.plot(np.arange(1,21), pv_prem_cap, color = '#DD6B7F',
         linewidth = 3, linestyle = '-.', label = 'Strategy 4 Premium (20th% Strike)')
ax2.set_yticks([0, 100*pow(10,6), 200*pow(10,6), 300*pow(10,6), 
                400*pow(10,6), 500*pow(10,6)], 
               ["0", "100", "200", "300", "400", "500"], fontsize = tick_font)
ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              ['0', '2', '4', '6', '8', '10', '12', 
                '14', '16', '18', '20'], fontsize = tick_font)
ax2.set_xlabel("Ensemble Year", fontsize = label_font)
ax2.set_ylabel("PV Opportunity Costs ($M)", fontsize = label_font)
ax2.legend(fontsize = leg_font, frameon = False)

fig.align_ylabels()

###############################################################################
################################# FIG 7 #######################################
###############################################################################
## one panel with cumulative costs, including reserves, CRAC, line of credit, etc
## SUM TOTAL OF ALL COSTS OF RISK MANAGEMENT TO ALL PARTIES
## cumulative costs
## BUT do not consider opportunity costs 

loc_crac_ens = loc_crac.mean()
no_loc_crac_ens = no_loc_crac.mean()
ins_crac_ens = ins_crac.mean()
ins_cap_ba_crac_ens = ins_res_crac.mean()

## total costs of CRAC -- $$ adjustments * PF sales 
from datetime import datetime as dt
from datetime import timedelta  
df_synth_load = pd.read_csv('CAPOW_data/Sim_hourly_load.csv', usecols=[1])
BPAT_load = pd.DataFrame(np.reshape(df_synth_load.values, (24*365,1200), order='F'))
base = dt(2001, 1, 1)
arr = np.array([base + timedelta(hours=i) for i in range(24*365)])
BPAT_load.index=arr
BPAT_load = BPAT_load.resample('D').mean()
BPAT_load.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
BPAT_load = pd.DataFrame(np.reshape(BPAT_load.values, (365*1188), order='F'))

df_load = pd.read_excel('net_rev_data.xlsx',sheet_name='load',skiprows=[0,1])#, usecols=[12]) ## column 12 = 2021
PF_load_y = df_load.loc[13, 'AVERAGE'] 

# Calculate daily BPAT proportions for demand and wind
load_ratio = BPAT_load/BPAT_load.mean()

PF_load = pd.DataFrame(PF_load_y*load_ratio)
PF_load = pd.DataFrame(np.reshape(PF_load.values, (365, 1188), order='F')).sum(axis=0)
PF_load = PF_load.iloc[:1180]
PF_load_ens = pd.DataFrame(np.reshape(PF_load.values, (59, 20), order='F'))

crac_costs_loc = (PF_load_ens * loc_crac).mean(axis = 0)
crac_costs_no_loc = (PF_load_ens * no_loc_crac).mean(axis = 0)
crac_costs_ins = (PF_load_ens * ins_crac).mean(axis = 0)
crac_costs_ins_cap = (PF_load_ens * ins_res_crac).mean(axis = 0)

pv_crac_costs_loc = pv_calc(crac_costs_loc, discount = discount)
pv_crac_costs_no_loc = pv_calc(crac_costs_no_loc, discount = discount)
pv_crac_costs_ins = pv_calc(crac_costs_ins, discount = discount)
pv_crac_costs_ins_cap = pv_calc(crac_costs_ins_cap, discount = discount)

interest_subsidy_nf = loc_pv_nf.mean(axis = 1) - loc_pv_loc.mean(axis = 1)
interest_subsidy_app = loc_pv_app.mean(axis = 1) - loc_pv_loc.mean(axis = 1)

## let's count the interest rate subsidy as the difference between 
## the loc and the non-federal (seems more fair than just choosing the 
## higher appropriations, and more realistic since non-federal would be a more 
## viable alternative)
foregone_repaid = loc_pv_loc.mean(axis = 1) - pv_repaid_loc_loc.mean(axis = 1)
foregone_repaid[foregone_repaid < 0] = 0 ## can't have a neg foregone payment

## tariff adjustments
 ## interest rate subsidy
 ## foregone repayment

sum_no_loc_costs = pv_crac_costs_no_loc.iloc[:,0]  #+ pv_res_no_loc.iloc[:,0]

    #- interest_subsidy_nf + \

sum_loc_costs = pv_crac_costs_loc.iloc[:,0] + \
                foregone_repaid +  \
                pv_repaid_loc_loc.iloc[:,0] #pv_res_loc.iloc[:,0] +

sum_ins_costs = pv_crac_costs_ins.iloc[:,0] + \
                disc_prems.iloc[:,0] #+ pv_res_ins.iloc[:,0]

sum_ins_cap_costs = pv_crac_costs_ins_cap.iloc[:,0] + disc_prems_cap.iloc[:,0] 
                ## not including the opportunity costs here 

## putting into a table for organizational purposes
strats = ['Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4']

cracs = [pv_crac_costs_no_loc.iloc[-1,0], 
         pv_crac_costs_loc.iloc[-1,0],
         pv_crac_costs_ins.iloc[-1,0], 
         pv_crac_costs_ins_cap.iloc[-1,0]]

subs = [0, 
        interest_subsidy_nf.iloc[-1], 
        0, 
        0]

foregone = [0, 
            foregone_repaid.iloc[-1], 
            0, 
            0]

prems = [0, 
         0, 
         disc_prems.iloc[-1,0], 
         disc_prems_cap.iloc[-1,0]]

repaid = [0, 
          pv_repaid_loc_loc.iloc[-1,0], 
          0, 
          0]

opp_costs = [pv_res_no_loc.iloc[-1,0], pv_res_loc.iloc[-1,0], 
             pv_res_ins.iloc[-1,0], 0]

###############################################################################
#################### distinguishing by different party ########################

tot_costs_bpa_no_loc = repaid[0] + prems[0] - subs[0] #+ opp_costs[0]
tot_costs_bpa_loc = repaid[1] + prems[1] - subs[1] # + opp_costs[1]
tot_costs_bpa_ins = repaid[2] + prems[2] - subs[2] #+ opp_costs[2]
tot_costs_bpa_res = repaid[3] + prems[3] - subs[3] #+ opp_costs[3]

print(f"Total Strategy 1 Value: {round(sum_no_loc_costs.iloc[-1],-6):,}")
print(f"Total Strategy 2 Value: {round(sum_loc_costs.iloc[-1],-6):,}")
print(f"Total Strategy 3 Value: {round(sum_ins_costs.iloc[-1],-6):,}")
print(f"Total Strategy 4 Value: {round(sum_ins_cap_costs.iloc[-1],-6):,}")
print() 

print(f"Costs to BPA, Strategy 1:  {round(tot_costs_bpa_no_loc,-6):,}")
print(f"Costs to BPA, Strategy 2:  {round(tot_costs_bpa_loc, -6):,}")
print(f"Costs to BPA, Strategy 3:  {round(tot_costs_bpa_ins, -6):,}")
print(f"Costs to BPA, Strategy 4:  {round(tot_costs_bpa_res, -6):,}")
print() 

## to customers
print(f"Costs to Customers, Strategy 1:  {round(cracs[0],-6):,}")
print(f"Costs to Customers, Strategy 2:  {round(cracs[1], -6):,}")
print(f"Costs to Customers, Strategy 3:  {round(cracs[2], -6):,}")
print(f"Costs to Customers, Strategy 4:  {round(cracs[3], -6):,}")
print()

## to government
print(f"Costs to Gov't', Strategy 1:  {round(foregone[0],-6):,}")
print(f"Costs to Gov't, Strategy 2:  {round(foregone[1], -6):,}")
print(f"Costs to Gov't, Strategy 3:  {round(foregone[2], -6):,}")
print(f"Costs to Gov't, Strategy 4:  {round(foregone[3], -6):,}")
print()

## to opp costs
print(f"Res Opp Costs, Strategy 1:  {round(opp_costs[0],-6):,}")
print(f"Res Opp Costs, Strategy 2:  {round(opp_costs[1], -6):,}")
print(f"Res Opp Costs, Strategy 3:  {round(opp_costs[2], -6):,}")
print()

print(f"Strategy 1 Avg NR: {round(nr_no_loc.mean(),-5):,}")
print(f"Strategy 2 Avg NR: {round(nr_loc.mean(),-1):,}")
print(f"Strategy 3 Avg NR: {round(nr_ins.mean(),-5):,}")
print(f"Strategy 4 Avg NR: {round(nr_ins_cap.mean(),-5):,}")
print(f"Cap Avg NR: {round(nr_no_loc_cap.mean(),-5):,}")
print() 

print(f"Strategy 1 Res Avg NR: {round(nr_no_loc_cap.mean(),-5):,}")
print(f"Strategy 2 Res Avg NR: {round(nr_loc_cap.mean(),-5):,}")
#print(f"Strategy 3 Res Avg NR: {round(nr_ins.mean(),-5):,}")
print(f"Strategy 4 Res Avg NR: {round(nr_ins_cap.mean(),-5):,}")
print()

print(f"Strategy 1 Res 99% VaR: {round(np.percentile(nr_no_loc_cap,1),-5):,}")
print(f"Strategy 2 Res 99% VaR: {round(np.percentile(nr_loc_cap,1),-1):,}")
#print(f"Strategy 3 Res 95% VaR: {round(np.percentile(nr_ins,5),-5):,}")
print(f"Strategy 4 Res 99% VaR: {round(np.percentile(nr_ins_cap,1),-1):,}")
print()

print(f"Strategy 1 95% VaR: {round(np.percentile(nr_no_loc,5),-5):,}")
print(f"Strategy 2 95% VaR: {round(np.percentile(nr_loc,5),-1):,}")
print(f"Strategy 3 95% VaR: {round(np.percentile(nr_ins,5),-5):,}")
print()

print(f"Strategy 1 99% VaR: {round(np.percentile(nr_no_loc,1),-5):,}")
print(f"Strategy 2 99% VaR: {round(np.percentile(nr_loc,1),-1):,}")
print(f"Strategy 3 99% VaR: {round(np.percentile(nr_ins,1),-5):,}")
print(f"Strategy 4 Res 99% VaR: {round(np.percentile(nr_ins_cap,1),-1):,}")

## make df to print to csv
loc_crac_mean = loc_crac.mean().mean()
no_loc_crac_mean = no_loc_crac.mean().mean()
ins_crac_mean = ins_crac.mean().mean()
ins_cap_ba_crac_mean = ins_res_crac.mean().mean()

table2 = pd.DataFrame([[nr_no_loc.mean(), np.percentile(nr_no_loc, 1), 61, no_loc_crac_mean], 
                  [ nr_loc.mean(), np.percentile(nr_loc, 1), 27, loc_crac_mean], 
                  [nr_ins.mean(), np.percentile(nr_ins, 1), 44, ins_crac_mean], 
                  [nr_ins_cap.mean(), np.percentile(nr_ins_cap, 1), 40, ins_cap_ba_crac_mean]], 
                      columns = ['Average Net Revenues ($M)', '99th% VaR ($M)', 
                                 'Percent Ensembles with Tariff Surcharges',
                                 'Average CRAC Adjustment ($/MWh)'])

table2.index = ["Strategy 1", 'Strategy 2', 'Strategy 3', 'Strategy 4']

table3 = pd.DataFrame(
                    [[opp_costs[0], nr_loc_cap.mean(), np.percentile(nr_no_loc_cap, 1)],
                     
                      [opp_costs[1], nr_no_loc_cap.mean(), np.percentile(nr_loc_cap, 1)],
                      
                      [opp_costs[2],  nr_ins_cap.mean(), np.percentile(nr_ins_cap, 1)]], 
                    columns = ['Opportunity Cost ($M)', 'Average Net Reveues ($M)', 
                               '99th% VaR ($M)'])
table3.index = ["Strategy 1", 'Strategy 2', 'Strategy 3']

table4 = pd.DataFrame(
                    [[tot_costs_bpa_no_loc, cracs[0], foregone[0], sum_no_loc_costs.iloc[-1]],
                     
                      [tot_costs_bpa_loc, cracs[1], foregone[1], sum_loc_costs.iloc[-1]],
                      
                      [tot_costs_bpa_ins, cracs[2], foregone[2], sum_ins_costs.iloc[-1]], 
                      
                      [tot_costs_bpa_res, cracs[3], foregone[3], sum_ins_cap_costs.iloc[-1]]],
                    
                    columns = ['PV Cost to BPA ($M)', 
                               'PV Cost to BPA Customers (Tariff  Surcharges) ($M)', 
                               'PV Cost to Government (Taxpayers) ($M)', 
                               'Total PV Cost ($M)'])
                     
table4.index = ["Strategy 1", 'Strategy 2', 'Strategy 3', 'Strategy 4']

# with pd.ExcelWriter('Results//tables_disc' + str(discount) + '.xlsx' ) as writer:
#     table2.to_excel(writer, sheet_name='table2')
#     table3.to_excel(writer, sheet_name='table3')
#     table4.to_excel(writer, sheet_name='table4')



###############################################################################

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, sharex = True)
plt.subplots_adjust(hspace = .3)
ax1.plot(loc_pv_app.index, interest_subsidy_app, linewidth = 3, 
         color = 'cyan', linestyle = ':', label = 'Subsidy Compared to \nCongressional Appropriations Rates')
ax1.plot(loc_pv_nf.index, interest_subsidy_nf, linewidth = 3, 
         color = 'cornflowerblue', linestyle = '-.', label = 'Subsidy Compared to \nNon-federal rates')
#ax1.plot(loc_pv_tr.index, loc_pv_tr.mean(axis = 1), linewidth = 3, 
#         linestyle = '--', color = 'cyan', alpha = .6, label = 'Line of Credit Rates')
ax1.legend(fontsize = leg_font, frameon = False)
ax1.set_xlabel("Ensemble Year", fontsize = label_font)
ax1.set_ylabel("Cumulative PV, \nInterest Rate Subsidy ($M)", fontsize = label_font)
ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              ['0', '2', '4', '6', '8', '10', '12', 
                '14', '16', '18', '20'], fontsize = tick_font)
ax1.set_yticks([0, 150*pow(10,6), 300*pow(10,6), 450*pow(10,6), 600*pow(10,6)],
              ['0', '150', '300', '450', '600'], fontsize = tick_font)

ax2.plot(pv_repaid_loc_loc.index, loc_pv_loc.mean(axis = 1), color = '#37A794', 
        linestyle = '--', linewidth = 4, label = 'Full repaidment')#', Expanding')
ax2.plot(pv_repaid_loc_loc.index, pv_repaid_loc_loc.mean(axis = 1), 
        linestyle = ':', linewidth = 4, label = 'Repayment', color = 'orchid')#', Expanding')
#ax2.bar(diff_avg_loc.index, foregone_expanding, label ='Foregone Payments', color = '#EE6677', alpha = .8)
ax2.set_xlabel("Ensemble Year", fontsize = label_font)
ax2.set_ylabel("Cumulative PV ($M)", fontsize = label_font)
ax2.set_yticks([0, .5*pow(10,9), 1*pow(10,9), 1.5*pow(10,9), 2*pow(10,9)],
              ['0', '500', '1000', '1500', '2000'],
              fontsize = tick_font)
ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              ['0', '2', '4', '6', '8', '10', '12', 
                '14', '16', '18', '20'], fontsize = tick_font)
ax2.legend(fontsize = leg_font, frameon = False)

## C) mean CRAC * load 
ax3.plot(np.arange(1,21), pv_crac_costs_no_loc, label = 'Strategy 1', 
         color = '#F1932D', linestyle = '-.', linewidth = 3)
ax3.plot(np.arange(1,21), pv_crac_costs_loc, label = 'Strategy 2', 
         color = '#73B761', linestyle = '--', linewidth = 3)
ax3.plot(np.arange(1,21), pv_crac_costs_ins, label = 'Strategy 3', 
         color = '#71AFE2', linewidth = 3, linestyle = ':')
ax3.plot(np.arange(1,21), pv_crac_costs_ins_cap, label = 'Strategy 4', 
         color = 'purple', alpha = 0.6, linestyle = '-', linewidth = 3)
ax3.set_yticks([0, 5*pow(10,6), 10*pow(10,6), 15*pow(10,6), 20*pow(10,6)], 
               ['0', '5', '10', '15', '20'], fontsize = tick_font)
ax3.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              ['0', '2', '4', '6', '8', '10', '12', 
                '14', '16', '18', '20'], fontsize = tick_font)
ax3.set_xlabel("Ensemble Year", fontsize = label_font)
ax3.set_ylabel("Cumulative PV, \nTariff Surcharges ($M)", fontsize = label_font)
ax3.legend(fontsize = leg_font, frameon = False)

ax4.plot(np.arange(1,21), sum_no_loc_costs, linewidth = 3, 
         color = '#F1932D', linestyle = '-.', label = 'Strategy 1, Total')
ax4.plot(np.arange(1,21), sum_loc_costs, linewidth = 3, 
         linestyle = '--', color = '#73B761', label = 'Strategy 2, Total')
ax4.plot(np.arange(1,21), sum_ins_costs, linewidth = 3, 
         color = '#71AFE2', linestyle = ':', label = 'Strategy 3, Total')
ax4.plot(np.arange(1,21), sum_ins_cap_costs, linewidth = 3, 
         color = 'purple', alpha = 0.6, linestyle = '-', label = 'Strategy 4, Total')
ax4.legend(fontsize = leg_font, frameon = False)
ax4.set_xlabel("Ensemble Year", fontsize = label_font)
ax4.set_ylabel("Cumulative PV Cost ($M)", fontsize = label_font)
ax4.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
              ['0', '2', '4', '6', '8', '10', '12', 
                '14', '16', '18', '20'], fontsize = tick_font)
ax4.set_yticks([0, .5*pow(10,9), 1*pow(10,9), 1.5*pow(10,9), 2*pow(10,9)],
              ['0', '500', '1000', '1500', '2000'], fontsize = tick_font)

fig.savefig("figure7.png", dpi = 1200)
fig.align_ylabels()


###########################################################################
###################### S1: different strikes ##############################
###########################################################################
## read in relevant inputs 
sq_nr = pd.read_csv("Results/ann_net_rev_avg_no_crac.csv").iloc[:,1] ## used to design index 
fcrps = pd.read_csv("CAPOW_data/Synthetic_streamflows_FCRPS.csv").iloc[:,1:]
names = pd.read_excel("CAPOW_data//BPA_name.xlsx")
fcrps.columns = names.iloc[0,:]

years=pd.DataFrame(np.arange(0,1200))
df=pd.DataFrame({'year':years.values.repeat(365)})
fcrps.loc[:,'year']=df['year']

drop = [82,150,374,377,540,616,928,940,974,980,1129,1191]
fcrps2 = fcrps[~fcrps.year.isin(drop)]
fcrps2.reset_index(inplace = True)
ann_fcrps = fcrps2.groupby(np.arange(len(fcrps2.index)) // 365).mean()
tda_merge = pd.concat([ann_fcrps['Dalls ARF'], sq_nr], axis = 1)

wet = tda_merge[tda_merge['Dalls ARF'] > np.percentile(tda_merge['Dalls ARF'], 15)]
dry = tda_merge[tda_merge['Dalls ARF'] < np.percentile(tda_merge['Dalls ARF'], 15)]

title_font = 12 
label_font = 10
tick_font = 10
leg_font = 10
marker_font = 14

wet5 = tda_merge[tda_merge['Dalls ARF'] > np.percentile(tda_merge['Dalls ARF'], 5)]
dry5 = tda_merge[tda_merge['Dalls ARF'] < np.percentile(tda_merge['Dalls ARF'], 5)]

wet10 = tda_merge[tda_merge['Dalls ARF'] > np.percentile(tda_merge['Dalls ARF'], 10)]
dry10 = tda_merge[tda_merge['Dalls ARF'] < np.percentile(tda_merge['Dalls ARF'], 10)]

wet20 = tda_merge[tda_merge['Dalls ARF'] > np.percentile(tda_merge['Dalls ARF'], 20)]
dry20 = tda_merge[tda_merge['Dalls ARF'] < np.percentile(tda_merge['Dalls ARF'], 20)]


label_font = 10
tick_font = 10
leg_font = 10

fig, (ax2, ax3, ax4) = plt.subplots(3, 1)
ax2.scatter(wet10.iloc[:,1], wet10['Dalls ARF'], s = marker_font, label = "Normal/Wet Years, 10th% Strike")
ax2.scatter(dry10.iloc[:,1], dry10['Dalls ARF'], s = marker_font, label = "Dry Years, 10th% Strike")
#ax2.set_title("10th% Strike", fontsize = title_font)
ax2.legend(frameon = False, fontsize = leg_font)
ax2.set_yticks([0, 100*pow(10,3), 200*pow(10,3), 300*pow(10,3)], \
               ['0', '100', '200', '300'], fontsize = tick_font)
ax2.set_xticks([-6*pow(10,8), -4*pow(10,8), -2*pow(10,8), 0*pow(10,8), 2*pow(10,8), 4*pow(10,8)],
               ['-600', '-400', '-200', '0', '200', '400'], fontsize = tick_font)    
#ax2.set_xlabel("Net Revenues ($M)", fontsize = label_font) 
ax2.set_ylabel("Dalles Flow ('000 cfs')", fontsize = label_font)

ax3.scatter(wet.iloc[:,1], wet['Dalls ARF'], s = marker_font, label = "Normal/Wet Years, 15th% Strike")
ax3.scatter(dry.iloc[:,1], dry['Dalls ARF'], s = marker_font, label = "Dry Years, 15th% Strike")
#ax3.set_title("15th% Strike", fontsize = title_font)
ax3.legend(frameon = False, fontsize = leg_font)
ax3.set_yticks([0, 100*pow(10,3), 200*pow(10,3), 300*pow(10,3)], \
               ['0', '100', '200', '300'], fontsize = tick_font)
ax3.set_xticks([-6*pow(10,8), -4*pow(10,8), -2*pow(10,8), 0*pow(10,8), 2*pow(10,8), 4*pow(10,8)],
               ['-600', '-400', '-200', '0', '200', '400'], fontsize = tick_font)    
#ax3.set_xlabel("Net Revenues ($M)", fontsize = label_font) 
ax3.set_ylabel("Dalles Flow ('000 cfs')", fontsize = label_font)

ax4.scatter(wet20.iloc[:,1], wet20['Dalls ARF'], s = marker_font, label = "Normal/Wet Years, 20th% Strike")
ax4.scatter(dry20.iloc[:,1], dry20['Dalls ARF'], s = marker_font, label = "Dry Years, 20th% Strike")
#ax4.set_title("20th% Strike", fontsize = title_font)
ax4.legend(frameon = False, fontsize = leg_font)
ax4.set_yticks([0, 100*pow(10,3), 200*pow(10,3), 300*pow(10,3)], \
               ['0', '100', '200', '300'], fontsize = tick_font)
ax4.set_xticks([-6*pow(10,8), -4*pow(10,8), -2*pow(10,8), 0*pow(10,8), 2*pow(10,8), 4*pow(10,8)],
               ['-600', '-400', '-200', '0', '200', '400'], fontsize = tick_font)    
ax4.set_xlabel("Net Revenues ($M)", fontsize = label_font) 
ax4.set_ylabel("Dalles Flow ('000 cfs')", fontsize = label_font)


###############################################################################
########################## S2: index performance ##############################
###############################################################################

ins = pd.read_csv("Results/ann_net_rev_index_no_loc15.csv").iloc[:,2]
loc = pd.read_csv("Results/ann_net_rev_loc2.csv").iloc[:,2]
res_ind = pd.read_csv("Results/ann_net_rev_index_cap_no_loc20.csv").iloc[:,2]
no_loc = pd.read_csv("Results/ann_net_rev_no_loc2.csv").iloc[:,2]

no_loc_cap = pd.read_csv("Results/ann_net_rev_no_loc_cap.csv").iloc[:,2]
loc_cap = pd.read_csv("Results/ann_net_rev_loc_cap.csv").iloc[:,2]

premium = -pd.read_csv("Results/net_payouts10.csv").iloc[:,1].min()
premium_cap = -pd.read_csv("Results/net_payouts20.csv").iloc[:,1].min()

axis_fonts = 12
ticks_fonts = 10
legend_fonts = 10

fig, ax = plt.subplots(1,3)
x = np.linspace(-600000000, 650000000)
y = np.linspace(-600000000, 650000000)
ax[0].scatter(loc, ins, color = '#5289C7', linewidth = 2, alpha = .8)
ax[0].fill_between(x, y, 650000000, color = '#4EB265', alpha = .25, 
                  label = 'Insurance Improves \nNet Revenues')
ax[0].vlines(0, -600000000, 650000000, linestyle = '--', color = 'red', linewidth = 4)
ax[0].plot(x,y, color = 'black', linewidth = 4, linestyle = '--')
ax[0].set_xlabel("Annual Net Revenues ($M), Expanding Line of Credit", fontsize = axis_fonts)
ax[0].set_ylabel("Annual Net Revenues ($M), \nWith Insurance", fontsize = axis_fonts)
ax[0].set_xticks([-6*pow(10,8), -4*pow(10,8), -2*pow(10,8), 0, 
            2*pow(10,8), 4*pow(10,8)],
            ['-600', '-400', '-200', '0', '200','400'], fontsize = ticks_fonts)
ax[0].set_yticks([-6*pow(10,8), -4*pow(10,8), -2*pow(10,8), 0, 
            2*pow(10,8), 4*pow(10,8)],
            ['-600', '-400', '-200', '0', '200','400'], fontsize = ticks_fonts)
ax[0].set_ylim(-600000000, 650000000)
ax[0].set_xlim(-600000000, 650000000)
ax[0].legend(frameon = False, loc = 'lower right', fontsize = legend_fonts)
ax[0].grid(True)


ax[1].scatter(no_loc, ins, color = '#5289C7', linewidth = 2, alpha = .8)
ax[1].fill_between(x, y, 650000000, color = '#4EB265', alpha = .25, 
                  label = 'Insurance Improves \nNet Revenues')
ax[1].vlines(0, -600000000, 650000000, linestyle = '--', color = 'red', linewidth = 4)
ax[1].plot(x,y, color = 'black', linewidth = 4, linestyle = '--')
ax[1].set_xlabel("Annual Net Revenues ($M), No Line of Credit", fontsize = axis_fonts)
ax[1].set_ylabel("Annual Net Revenues ($M), \nWith Insurance", fontsize = axis_fonts)
ax[1].set_xticks([-6*pow(10,8), -4*pow(10,8), -2*pow(10,8), 0, 
            2*pow(10,8), 4*pow(10,8)],
            ['-600', '-400', '-200', '0', '200','400'], fontsize = ticks_fonts)
ax[1].set_yticks([-6*pow(10,8), -4*pow(10,8), -2*pow(10,8), 0, 
            2*pow(10,8), 4*pow(10,8)],
            ['-600', '-400', '-200', '0', '200','400'], fontsize = ticks_fonts)
ax[1].set_ylim(-600000000, 650000000)
ax[1].set_xlim(-600000000, 650000000)
ax[1].legend(frameon = False, loc = 'lower right', fontsize = legend_fonts)
ax[1].grid(True)

ax[2].scatter(no_loc, res_ind, color = '#5289C7', linewidth = 1, alpha = .8)
ax[2].fill_between(x, y, 650000000, color = '#4EB265', alpha = .25, 
                  label = 'Reserve Cap + Insurance \nImproves Net Revenues')
ax[2].plot(x,y, color = 'black', linewidth = 4, linestyle = '--')
ax[2].vlines(0, -600000000, 500000000, linestyle = '--', color = 'red', linewidth = 4)
ax[2].set_xlabel("Annual Net Revenues ($M), No Line of Credit", fontsize = axis_fonts)
ax[2].set_ylabel("Annual Net Revenues ($M), \nInsurance and Reserve Cap", fontsize = axis_fonts)
ax[2].set_xticks([-6*pow(10,8), -4*pow(10,8), -2*pow(10,8), 0, 
            2*pow(10,8), 4*pow(10,8)],
            ['-600', '-400', '-200', '0', '200','400'], fontsize = ticks_fonts)
ax[2].set_yticks([-6*pow(10,8), -4*pow(10,8), -2*pow(10,8), 0, 
            2*pow(10,8), 4*pow(10,8)],
            ['-600', '-400', '-200', '0', '200','400'], fontsize = ticks_fonts)
ax[2].set_ylim(-600000000, 650000000)
ax[2].set_xlim(-600000000, 650000000)
ax[2].grid(True)
ax[2].legend(frameon = False, loc = 'lower right', fontsize = legend_fonts)















