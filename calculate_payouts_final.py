# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 08:16:23 2022

@author: rcuppari
"""
import numpy as np	
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score    
from sklearn.linear_model import LinearRegression

########################### set up for index ##################################
## set tda strike and give a name 
strike_name = '05'
strike_pct = .05
strike_pct2 = .5

## retrieve BPA net revenues **without** considering CRAC increases 
suffix = 'no_crac_no_capex_cheap'
update = ''#'AVERAGE'
res_cap = ((608.6*pow(10,6))/120)*90

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
ann_fcrps = fcrps2.groupby(np.arange(len(fcrps2.index)) // 365).mean()

## corresponding annual net revenues
ann_nr = pd.read_csv("Results/ann_net_rev_" + suffix + ".csv").iloc[:,1]

## read in mid-c price (marginal price) 
midc=pd.read_csv('CAPOW_Data/MidC_daily_prices_new.csv').iloc[:,1]
midc=pd.DataFrame(np.reshape(midc.values, (365,1200), order='F'))
midc.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
midc=pd.DataFrame(np.reshape(midc.values, (365*1188), order='F'))
midc.columns=['midc']
ann_midc = midc.groupby(np.arange(len(midc)) // 365).mean()

###############################################################################
# #################### now look at our index equation ###########################
# ## differentiate between "wet" and "dry" years with our strike pct
strike = np.quantile(ann_fcrps['Dalls ARF'], strike_pct)
strike2 = np.quantile(ann_fcrps['Dalls ARF'], strike_pct2)

################################# TEST ########################################

wet_dry = pd.concat([ann_nr,ann_fcrps],axis=1)
wet = wet_dry[wet_dry['Dalls ARF'] >= strike]
dry = wet_dry[wet_dry['Dalls ARF'] < strike]

X = pd.concat([ann_fcrps, ann_midc], axis = 1)
## if we design a regression with just flow at the Dalles, we get *okay* results
X = sm.add_constant(X)

X_train, X_test, y_train, y_test=tts(X, ann_nr, test_size=.2, random_state=1) #3
est_tda = sm.OLS(y_train, X_train[['Dalls ARF', 'const']], hasconst = True)
est_tda2 = est_tda.fit()

pred_tda = est_tda2.predict(X[['Dalls ARF', 'const']])
est_tda2.summary()
np.corrcoef(pred_tda, ann_nr)**2


## so let's make it more interesting...(this is obviously clean code after 
## way to much time playing around with different variables)
reg_inputs = ['const', 'Dalls ARF', 'midc']

est=sm.OLS(y_train, X_train[reg_inputs], hasconst = True)
est2=est.fit()
print(est2.summary()) 
## yay! Way better
print(est2.rsquared_adj) 

pred = est2.predict(X_test[reg_inputs])
pred2 = est2.predict(X_train[reg_inputs])
pred_all = est2.predict(X[reg_inputs])

strike = np.percentile(X['Dalls ARF'], strike_pct*100)
X_below = X[X['Dalls ARF'] < strike]
X_below = sm.add_constant(X_below)

pred_below = est2.predict(X_below[['const', 'Dalls ARF', 'midc']])
rev_below_index = ann_nr[ann_nr.index.isin(X_below.index)] 
print(f"r2 for overall regression, below strike: {np.corrcoef(rev_below_index, pred_below)}")
#rev_below_index.to_csv("Results/rev_below_index"+ str(strike_pct) + ".csv")

## we can also use LOOCV to evaluate model 
cv = LeaveOneOut()
#build multiple linear regression model
model = LinearRegression()

scores = cross_val_score(model, X[reg_inputs], ann_nr,
                         scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

rme = np.sqrt(np.mean(np.absolute(scores)))
mae_below = np.mean(np.absolute(scores))
print('Root Mean Error, Below Index: ' + str(rme))
print('MAE, Below Index: ' + str(mae_below.mean()))
print('%MAE, Below Index: ' + str(mae_below.mean()/rev_below_index.mean()))


# ## BUT creating a linear index based on flow does not work very well for neg years -- 
# ## there seem to be two separate linear components based on neg versus pos NR
# plt.scatter(pred_all, ann_nr)
# plt.plot([ann_nr.min(), ann_nr.max()], [ann_nr.min(), ann_nr.max()], color = 'red')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")

pred_all_below = est2.predict(X_below[reg_inputs])
np.corrcoef(pred_all_below, rev_below_index)**2

## so, let's customize our index based on a strike with Dalles flow and a unique 
## index based on points below that strike

X_above = X[~X.index.isin(X_below.index)]
X_above = sm.add_constant(X_above)

rev_above_index = ann_nr[~ann_nr.index.isin(X_below.index)] 
#rev_above_index.to_csv("Results/rev_above_index"+ str(strike_pct) + ".csv")

## How does the TDA only index fare? 
## spoiler: mediocre -- it does way better on the full dataset than on the low years
est_tda = sm.OLS(rev_below_index, X_below[['Dalls ARF', 'const']], hasconst = True)
est_tda = est_tda.fit()
print(est_tda.summary())

est_tda_above = sm.OLS(rev_above_index, X_above[['Dalls ARF', 'const']], hasconst = True)
est_tda_above = est_tda_above.fit()
print(est_tda_above.summary())

## let's save results for overall TDA 
tda_index_below = est_tda2.predict(X_below[['Dalls ARF', 'const']])
#tda_index_below.to_csv("Results/tda_index_below"+ str(strike_pct) + ".csv")
print(f"r2 for tda only, below strike: {np.corrcoef(tda_index_below, rev_below_index)}")
tda_index_above = est_tda2.predict(X_above[['Dalls ARF', 'const']])
#tda_index_above.to_csv("Results/tda_index_above"+ str(strike_pct) + ".csv")

## first above the strike
scores = cross_val_score(model, X_above[['Dalls ARF', 'const']], rev_above_index, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
mae_tda_above = np.mean(np.absolute(scores))

## then below the strike
scores = cross_val_score(model, X_below[['Dalls ARF', 'const']], rev_below_index, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
mae_tda_below = np.mean(np.absolute(scores))

## can confirm that it performs poorly below the strike
print("MAE with TDA regression > strike: " + str(mae_tda_above))
print("MAE with TDA regression < strike: " + str(mae_tda_below))

## fully?
scores = cross_val_score(model, X[['Dalls ARF', 'const']], ann_nr, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
np.mean(np.absolute(scores))

###############################################################################
########## now let's regress differently based on above/below strike ##########
##########                and with our fuller index                  ##########
X4 = X_below[reg_inputs]

X_train,X_test,y_train,y_test = tts(X4, rev_below_index, test_size=.2, random_state = 4)
est_ind = sm.OLS(y_train, X_train, has_const = True)
est_ind2 = est_ind.fit()
print(est_ind2.summary()) 

pred_test = est_ind2.predict(X_test)
np.corrcoef(pred_test, y_test)**2

scores = cross_val_score(model, X4, rev_below_index, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)

rme = np.sqrt(np.mean(np.absolute(scores)))
mae_below = np.mean(np.absolute(scores))
print('Root Mean Error, Below Index: ' + str(rme))
print('MAE, Below Index: ' + str(mae_below.mean()))
print('%MAE, Below Index: ' + str(mae_below.mean()/rev_below_index.mean()))


pred_below = est_ind2.predict(X4)
#pred_below.to_csv("Results/index_pred_below"+ str(strike_pct) + ".csv")

np.corrcoef(pred_below, rev_below_index)**2

#########################################################################
## for above the strike, we are going to leave it as the first regression
pred_above = pred_all[~pred_all.index.isin(X_below.index)]
#pred_above.to_csv("Results/index_pred_above"+ str(strike_pct) + ".csv")

## making a pretty plot to see it all... 
xmin = -800*pow(10,6)
xmax = 500*pow(10,6)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols = 1)
ax1.scatter(rev_above_index, tda_index_above, label = "Years Above Strike", 
            color = 'cornflowerblue')
ax1.scatter(rev_below_index, tda_index_below, label = "Years Below Strike (payout)", color = 'brown')
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

ax2.scatter(rev_above_index, pred_above, label = 'Years Above Strike', 
            color = 'cornflowerblue')
ax2.scatter(rev_below_index, pred_below, alpha = .8, label = 'Years Below Strike (payout)', color = 'brown')
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

## input needs to have "strike" column to compare to strike (column #2)
## a column for which the mean should be calculated (column #1)
## and then other columns in order of usage in above regression 
## and if slope == True, the price data to use to calculate slope in col 2
## mean is input if using slope, because the mean value is what the 1 slope will be
## I cap payouts based on the 95th% of simulated losses

################################################################################
# ################# the juicy bit: putting our index into practice ###############

## set cap for payouts to BPA
cap = np.percentile(ann_nr, 1)
## set up the inputs to the model 
inputs = X[['const', 'Dalls ARF', 'midc']]

from wang_transform import wang_slim
from wang_transform import id_payouts_index2
from wang_transform import id_payouts_index

payouts = id_payouts_index(inputs = inputs, strike = strike, cap = cap, \
                            model = est_ind2)

# # ## now use the wang transform to get the premiums and final payouts for each one
prem_ind_fixed, whole_ind_fixed = wang_slim(payouts, contract = 'put', from_user = True)
#whole_ind_fixed.to_csv("Results/net_payouts2" + strike_name + ".csv")

## calculate NR with index insurance payouts
import BPA_net_rev_func as calc
nr_ind_fixed, index_fix = calc.net_rev_full_inputs(pd.DataFrame(whole_ind_fixed), custom_redux = 0, 
                                                    name = 'index_no_loc2' + strike_name, 
                                                    excel = True, 
                                                    y = 'no_loc', repay_ba = 'no', 
                                                    infinite = False)


        