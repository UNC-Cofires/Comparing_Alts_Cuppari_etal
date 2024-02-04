# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 22:29:34 2023

@author: rcuppari
"""

######################################################
######### BPA cost calculation functions #############
######################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

## 30 year time horizon, based on the payback period for Treasury
def expected_repay(df, rate, time_horizon = 30): 
    ## used is the principal in a given year 
    ## for each payment, want to store the theoretical repayment 
    amortized = pd.DataFrame(0, index = np.arange(0,len(df)), columns=np.arange(0,len(df.columns)))    
    
    for r in range(0, len(df)): 
        for c in range(0, len(df.columns)):
            ## each year should be the cumulative owed (so stacking the previous
            ## amortized value since 20 yr ensemble < 30 yr payback)
            i = rate
            num = i * pow((1 + i), time_horizon)
            deno = pow((1 + i), time_horizon) - 1
            new_oblig = df.iloc[r,c] * (num/deno)
            if r == 0: ## no amortized amount in yr 1
                amortized.iloc[r,c] = 0 
            else: 
                amortized.iloc[r,c] = amortized.iloc[r-1,c] + new_oblig
 
    return amortized

## pv_calc gives the net present value of the interest + principal payments     
## need input of discount rate and "used", which indicates whether we are asking
## for the pv of the modelled repayments (False) or the expected repayments (True)
## NOTE: this is cumulative
## also note, I confusing left these rates as decimals down here, NOT PERCENTS
def pv_calc(df, discount = 5, rate = 'na', modelled = False, 
            time_horizon = 30): 
    if modelled == True:
        amortized = expected_repay(df, rate, time_horizon = time_horizon)
#        print(amortized.mean().mean())
    else: 
        amortized = df ## when inputting repaid values, we just need the pv
        ## because the "amortization" is just what's being repaid

    try:        
        pv = pd.DataFrame(0, index = np.arange(len(df)), columns = np.arange(len(df.columns)))
            
        for r in range(0, len(amortized)):
            for c in range(0, len(amortized.columns)): 
                new_yr = amortized.iloc[r,c]/pow((1 + discount/100),r)

                
                if r == 0: 
                    pv.iloc[r, c] = new_yr 
                else: 
                    pv.iloc[r, c] = pv.iloc[(r-1), c] + new_yr
                
    except: 
        pv = pd.DataFrame(index = np.arange(len(df)), columns = ['npv'])
        for r in range(0, len(df)):
            new_yr = df[r]/pow((1 + discount/100),r)
            
            if r == 0: 
                pv.iloc[r,0] = new_yr
            else: 
                pv.iloc[r,0] = pv.iloc[r-1,0] + new_yr
    if modelled == True:             
        return amortized, pv
    else: 
        return pv 

def repaid(redux, color = False):
    Repaid_e=pd.DataFrame()
    
    for e in range (1,60):
        Repaid_e=pd.concat([Repaid_e, pd.read_excel('Results/BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=['repaid'])], axis = 1)

#    Repaid_e.fillna(0, inplace = True)
    Repaid_e = Repaid_e #* pow(10, -2)
    repaid_avg = Repaid_e.mean(axis = 1)

    if color != False: 
        plt.figure()
        plt.plot(Repaid_e, color = color, linewidth = 2, alpha = .7)
        plt.plot(repaid_avg, color = 'black', linewidth = 4, label = 'Average Repayment')
        plt.xlabel("Year", fontsize = 20)
        plt.ylabel("$M", fontsize = 20)
        plt.yticks([0, 25000000, 50000000, 75000000, 100000000, 125000000, 150000000, 175000000], 
                    ['0', '25', '50', '75', '100', '125', '150', '175'], fontsize = 18)
        plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 
                    ['1', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20'], fontsize = 18)
        plt.show()
    
    return Repaid_e, repaid_avg 

def crac(redux): 
    # CRAC ensemble  
    CRAC_e=pd.DataFrame()
    for e in range (1,60):
        CRAC_e=pd.concat([CRAC_e, pd.read_excel('Results/BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[6])], axis=1)
        #Qc=(100/PF_rates_avg)*CRAC_e.T
    Qc=CRAC_e.T
    Qc.reset_index(inplace=True, drop=True)

    #CRAC distribution
    count=np.sum(CRAC_e.any())
    percent1=100*count/59  #BAU=11.86% 
    
    mean = Qc.mean().mean()    
#    print(redux)
#    print('Percent of CRAC ensembles: %.2f' % percent1 )
#    print('Mean CRAC: %.2f' % mean )


    TTP_e=pd.DataFrame()
    for e in range (1,60):
        TTP_e=pd.concat([TTP_e, pd.read_excel('Results/BPA_net_rev_stoc_y' + redux + '.xlsx', sheet_name='ensemble' + str(e), usecols=[2])], axis=1)
    count=sum(sum(sum([TTP_e != 1]).values)) #0% for both BAU and minus 10% and minus20%
    percent2=100-100*count/1188  
    #print(redux)
    #print ('TPP: %.2f' % percent2 )
    
    return Qc, count, mean

def read_reserves(suffix):
    Reserves_e=pd.DataFrame()
    for e in range (1,60):
        Reserves_e=(pd.concat([Reserves_e, pd.read_excel('Results/BPA_net_rev_stoc_y' + suffix + '.xlsx', sheet_name='ensemble' + str(e), usecols=[1])['Reserves']], axis=1)) 
    return Reserves_e 


def compound_interest(principal, rate, time): 
    # Calculates compound interest
    ## rate here is in % form so we divide by 100 
    Amount = principal * (pow((1 + rate / 100), time))
    return Amount



def pv_opp_cost(money, time_horizon, safe_rate, alt_rate, discount = 5):
    opp_cost2 = []
    cum_opp_cost = 0
    
    for i in range(0, len(money)): 
        val_safe = compound_interest(money[i], safe_rate, time_horizon)
        val_alt = compound_interest(money[i], alt_rate, time_horizon)
    
        opp_cost_ann = (val_alt - val_safe)/pow((1+discount/100),i)
        #print(opp_cost_ann)
       
        if i > 0: 
            opp_cost2.append(opp_cost_ann + opp_cost2[i-1])
        else: 
            opp_cost2.append(opp_cost_ann)
            
        cum_opp_cost += opp_cost_ann   
        
    return opp_cost2
    
    
def plot_cum_pv(df, label, color = 'blue', linestyle = 'solid', loc = 'upper left'): 
    plt.plot(df.index, df, color = 'grey', alpha = .6, linewidth = 1)
    plt.plot(df.index, df.mean(axis = 1), color = color, linewidth = 2, label = label)
    plt.xlabel("Ensemble Year", fontsize = 18) 
    plt.xticks([0,2,4,6,8,10,12,14,16,18,20], 
           ['0', '2', '4', '6', '8', '10', '12',
            '14', '16', '18', '20'], fontsize = 16)
    plt.ylabel("$M", fontsize = 18)
    plt.yticks([0, 500000000, 1000000000, 1500000000, 2000000000, 
            2500000000, 3000*pow(10,6), 3500*pow(10,6), 
            4000*pow(10,6), 4500*pow(10,6)],
           ['0', '500', '1000', '1500', '2000', '2500', '3000', '3500', '4000','4500'], 
           fontsize = 16)
    plt.legend(frameon = False, fontsize = 16, loc = loc)
    plt.show()

def TF_used(prefix, num_ensem = 59):
    used_e = pd.DataFrame()
    
    for e in range (1,60):
        used_e=pd.concat([used_e, pd.read_excel('Results/BPA_net_rev_stoc_y' + prefix + '.xlsx', sheet_name='ensemble' + str(e), usecols=['used_BA'])], axis=1) #usecols=['TF2'])], axis=1)   
    return used_e


def TF_used2(prefix, num_ensem = 59):
    used_e = pd.DataFrame()
    tf1 = pd.DataFrame()
    tf2 = pd.DataFrame()
    for e in range (1,60):
        tf1 = pd.concat([tf1, pd.read_excel('Results/BPA_net_rev_stoc_y' + prefix + '.xlsx', sheet_name='ensemble' + str(e), usecols=['TF1'])], axis = 1)
        tf2 = pd.concat([tf2, pd.read_excel('Results/BPA_net_rev_stoc_y' + prefix + '.xlsx', sheet_name='ensemble' + str(e), usecols=['TF2'])], axis = 1)
        used_e = pd.concat([used_e, pd.read_excel('Results/BPA_net_rev_stoc_y' + prefix + '.xlsx', sheet_name='ensemble' + str(e), usecols=['used_BA'])], axis = 1)
               
    return tf1, tf2, used_e


## need to use what was USED, to show what should have come back  
def used(redux, folder = 'Results'):
    used_e=pd.DataFrame(index = np.arange(0,20), columns = np.arange(0,59))
    used_e.iloc[0,:] = 0
    
    ba_e = pd.DataFrame()
    add_ba_e = pd.DataFrame()
    
    for e in range (1,60):
        ba_e = pd.concat([ba_e, pd.read_excel(f'{folder}/BPA_net_rev_stoc_y{redux}.xlsx', sheet_name='ensemble' + str(e), usecols=['BA'])], axis = 1)
        add_ba_e = pd.concat([add_ba_e, pd.read_excel(f'{folder}/BPA_net_rev_stoc_y{redux}.xlsx', sheet_name='ensemble' + str(e), usecols=['add_BA'])], axis = 1)
        
    ba_e2 = ba_e.copy()
    add_ba_e.columns = ba_e.columns
    ## first need to account for adding BA (which might be added even after it is used)
    for i in range(1,len(ba_e)): 
        ba_e2.iloc[i,:] = ba_e.iloc[i,:] - add_ba_e.iloc[i,:]
        used_e.iloc[i,:] = ba_e2.iloc[i-1,:] - ba_e2.iloc[i,:]

    used_e[used_e < 0] = 0
                
    used_e.fillna(0, inplace = True)
    used_avg = used_e.mean(axis = 1)
    
    return used_e, used_avg 


