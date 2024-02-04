# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 22:12:02 2021

@author: rcuppari
"""
import pandas as pd 
import scipy.stats as st
import numpy as np 

##########################################################################
######### wang transform function ########################################
############## Returns dataframe with net payout #########################
##########################################################################

## lam is set to 0.25 unless otherwise specified (risk adjustment)
## df should be dataframe with payout per year 
## modified slightly -- from user = False means that it is from the insurer's perspective
def wang_slim(payouts, lam = 0.25, contract = 'put', from_user = False):  
  if from_user == True: 
      ## switch the signs because you are feeding a negative value where the payout occurs
      ## all other values are zero
      payouts = -payouts
  if contract == 'put': 
      lam = -abs(lam)
  unique_pays = pd.DataFrame()
  unique_pays['unique'] = payouts.payout.unique()
  unique_pays['prob'] = 0 
  for j in range(len(unique_pays)):  
      count = 0
      val = unique_pays['unique'].iloc[j]
      for i in np.arange(len(payouts)): 
          if payouts['payout'].iloc[i] == val: 
              count += 1
    #  print(count)
      unique_pays['prob'].iloc[j] = count/len(payouts)
      
  unique_pays.sort_values(inplace=True, by='unique')
  dum = unique_pays['prob'].cumsum()  # asset cdf
  dum = st.norm.cdf(st.norm.ppf(dum) + lam)  # risk transformed payout cdf
  dum = np.append(dum[0], np.diff(dum))  # risk transformed asset pdf
  prem = (dum * unique_pays['unique']).sum()
  print(prem)
  payouts.sort_index(inplace=True)

  if from_user == False: 
     ## want the insurer's perspective
      whole = (prem - payouts['payout'])

  else: 
#      payouts.sort_index(inplace=True)
     whole = (payouts['payout'] - prem)

  
  return prem, whole

def id_payouts_index(inputs, strike, cap, model): 
    payouts = pd.DataFrame(index = np.arange(0, len(inputs)),columns=['payout']) 
    payouts['payout'] = 0
    
    for i in np.arange(len(inputs)):
        if inputs['Dalls ARF'].iloc[i] < strike: 
            ## model will predict a value that represents losses essentially
            payouts.iloc[i,0] = model.predict(inputs.iloc[i,:])
       #     print(payouts.iloc[i,0])
            ## constrain so that if predicted value is > 0, BPA does not get paid
            if payouts.iloc[i,0] > 0:
                payouts.iloc[i,0] = 0
            ## cap payouts
            if payouts.iloc[i,0] < cap: 
                payouts.iloc[i,0] = cap     
            ## NOTE: these are left negative to work within the wang transform function
    return payouts

def id_payouts_index2(inputs, strike, cap, model): 
    payouts = pd.DataFrame(index = np.arange(0, len(inputs)),columns=['payout']) 
    payouts['payout'] = 0
    
    for i in np.arange(len(inputs)):
        if model.predict(inputs.iloc[i,:])[0] < strike: 
            ## model will predict a value that represents losses essentially
            payouts.iloc[i,0] = model.predict(inputs.iloc[i,:])
            ## cap payouts
            if payouts.iloc[i,0] < cap: 
                payouts.iloc[i,0] = cap     
            ## NOTE: these are left negative to work within the wang transform function
        else: 
            payouts.iloc[i,0] = 0

    return payouts

