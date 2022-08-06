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

def wang_slim(payouts, lam = 0.25, from_user = False, premOnly = False):  
 # if from_user == True: 
 #     payouts = -payouts
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
 # if from_user == False: 
  prem = (dum * unique_pays['unique']).sum()
  print(prem)
  if premOnly == False:
      payouts.sort_index(inplace=True)
      whole = (payouts['payout'] - prem)
  #else: 
  #    prem = ((dum * unique_pays['unique']).sum())
  #    print(prem)
  #    if premOnly == False:
  #        payouts.sort_index(inplace=True)
  #        whole = (prem - payouts['payout'])
  
  return prem, whole












