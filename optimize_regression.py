# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 09:53:47 2022

@author: rcuppari
"""

import numpy as np	
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import statsmodels.formula.api as smf
from itertools import combinations

########## optimize regression to maximize 95% VaR, minimize CRAC, 
########## and maximize probability of Treasury Payment 
########## but because CRAC/TTP are dependent on the sequence of years, 
########## will need to re-run the BPA model for each candidate 
########## however, do know that reduction in 95% VaR is very well 
########## correlated with those outcomes, so use the objective of maximizing
########## VaR and annual net revenues and use then feed the top candidates into 
########## the model 

# metrics include: 95% VaR, TTP, average annual net revenue
# can use the existing regression equation as the benchmark to beat: 
# >= 125% VaR, > 95% TTP, >= .8 average SQ net rev 
# df to input must have all variables to be analyzed 
# monthly = True will include monthly average values 
# otherwise, just annual timestep will be used 

## have two companion functions also: 
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def combination_ann_regressions(df, y = outcome): 
    ## make sure data is matching! 
    regression_output = pd.DataFrame()

    X_train,X_test,y_train,y_test=tts(x,y,test_size=.3,random_state=1)

    for subset in powerset(x.columns): 
        print(subset)
        if ((len(subset) > 0) & (len(subset <= 4))): 
            model = sm.OLS(y_train, X_train[list(subset)]).fit()
    
    ## check the fit on test data 
            predicted = model.predict(X_test[list(subset)])
            pred_r2 = (predicted.corr(y_test)**2)
    ## store regression inputs, p-values, and adj rsquared 
            pval = list(model.pvalues)
            r2 = np.float(model.rsquared_adj)
            series = list(subset)
            
            new = {'inputs':[series], 'r2': r2, 'pval':[pval], 'test_r2': pred_r2} 
            new = pd.DataFrame(new)
            regression_output = pd.concat([regression_output, new], axis = 0)
   
    return regression_output


def optim_reg(ann_df, outcome, mon_df = 0, metric = 'net_rev'):

    ## if mon_df is non-zero, means we need to clean up the dataframe 
    ## that should have a column with month, one with year, and then the 
    ## variable values. Need it to become a separate variable for each month's
    ## value
    if mon_df != 0: 
        months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

        monthly_vals = pd.DataFrame()    
        for m, month in enumerate(months): 
            m = m+1
            subset_mon = mon_df[mon_df['month']==m]
            ## get rid of the month and year columns now, since we 
            ## don't need to duplicate those
            subset_mon = subset_mon.iloc[:, 2:]
            print(month)
            
            mon_names = []
            for i in range(0, len(subset_mon.columns)): 
                new_name = subset_mon.columns[i] + '_' + month
                # print(new_name)
                mon_names.append(new_name)

            subset_mon.columns = mon_names
            subset_mon.reset_index(drop = True, inplace = True)
            monthly_vals = pd.concat([monthly_vals, subset_mon], axis = 1)
        
        ## now want to combine these new variables with the annual 
        ## values
    df = pd.concat([df, monthly_vals], axis = 1)

    ## now need to get all possible combinations of variables <=4 
    ## and use those regressions to predict net revenues 
    ## then track how high the 95% VaR can go 
    
    vars_to_model = list(df)
    print(vars_to_model)
    
    combi = []
    output = pd.DataFrame()
    modelNumber = 1

## really want to run this for every possible combination of <5 variables
## more than that and we are probably overfitting 
    for i in range(1,len(vars_to_model)):
        combi = (list(combinations(vars_to_model, i)))
        for c in combi:
            print('— — — — — — -')
            variable_string = ‘PREGNANT ~ 1 ‘
            var_iter =1
            for j in list(c):
                variable_string +=’ + ‘ + str(j)
                if var_iter == len(list(c)):
                    try:
                        result = smf.logit(formula= variable_string, data=preg_data).fit()
                        coeffs = result.params
                        coeffs = pd.DataFrame({‘Variable’:coeffs.index, ‘Values’:coeffs.values})
                        predTable = result.pred_table()
                        prsq = result.prsquared
 row1 = [{‘Variable’:’modelNumber’, ‘Values’:modelNumber}
 , {‘Variable’:’pRSQ’, ‘Values’:prsq}
 , {‘Variable’:’precision’, ‘Values’:precision}
 , {‘Variable’:’recall’, ‘Values’:recall}
 , {‘Variable’:’accuracy’, ‘Values’:accuracy}
 , {‘Variable’:’truepos’, ‘Values’:tp}
 , {‘Variable’:’trueneg’, ‘Values’:tn}
 , {‘Variable’:’falsepos’, ‘Values’:fp}
 , {‘Variable’:’falseneg’, ‘Values’:fn}
 , {‘Variable’:’variableString’, ‘Values’:variable_string}
 , {‘Variable’:’NumVars’, ‘Values’:len(c)} 
 ]
 coeffs = coeffs.append(row1, ignore_index=True)
 output= output.append(coeffs.set_index(‘Variable’).T)
 modelNumber += 1
 except :
 pass
 var_iter +=1


## once we have narrowed the pool of candidate models 
## we can use them in the BPA model
## since we mostly care about downside risk, we are just 
## going to use the index insurance component, so that 
## we can speed up this process 

## but first we need the payouts and we need to price the 
## instrument to account for the annual premium 
## in this step, 
import BPA_net_rev_func
