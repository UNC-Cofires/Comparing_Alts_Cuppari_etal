# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:29:19 2019

@author: sdenaro
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta  
import numpy as np
import numpy.matlib as matlib
import seaborn as sns
from sklearn import linear_model

    ###########################################################################
    ## this function calculates BPA net revenues 
    ## and evaluates the performance of their risk mitigation tools
    ## while allowing the addition of other contracts, given payouts for the 
    ## contract (abbreviation for saving), the loading (used to calculate the annual premium)
    ## and any customer reductions (default = 0, should set as decimal, e.g., .1)
    ## note that in the original runs, some years failed. These are hard coded 
    ## in the text year, but are activated only if drop = True 
    ## if excel = True, save the full sheets with TTP, CRAC, etc. 
    ## otherwise just save the annual net revenues
    ###########################################################################

## numbers have been updated to reflect the 2022 Annual Report 
#######################################################################################################################
## this time I have already calculated the net payout with the premium an can input it directly (no separate loading)
## df needs to have the net payment to BPA from counterparty (i.e. positive when below tda strike and negative 
## when above the strike)
## note: when using marginal price to scale payments, don't want to use v :) 
## p2 = % pos net revenues used to repay BA (debt optimization)
## p = % pos net revenues used to replenish reserves
## repay_ba says whether or not there is a line of credit ("yes" or "no")
sequences = pd.DataFrame(pd.read_csv("random_sequences.csv").iloc[:,1])

def net_rev_full_inputs(df_payout, custom_redux = 0, name = '', excel = True, y = 'AVERAGE', infinite = False, 
                        p = 10, p2 = 32, res_cap = 608691000, repay_ba = 'yes',
                        ba_int_rate = 0.042, time_horizon = 50, sequence = 'no'): ## default interest rate is from hist avg
    #Set Preference Customers reduction percent (number)
    

    # Yearly firm loads (aMW)
    df_load=pd.read_excel('net_rev_data.xlsx',sheet_name='load',skiprows=[0,1])#, usecols=[12]) ## column 12 = 2021
    PF_load_y = df_load.loc[13, y] - custom_redux*df_load.loc[13 ,y]
    IP_load_y = df_load.loc[3, y] - custom_redux* df_load.loc[3, y]
    ET_load_y = df_load.loc[14, y]

    # Hourly hydro generation from FCRPS stochastic simulation
    df_hydro = pd.read_csv('CAPOW_data/new_BPA_hydro_daily.csv')
    df_hydro = df_hydro['gen']

    #df_hydro=pd.read_csv('CAPOW_data/new_BPA_hydro_daily2.csv', usecols=([1]))
    BPA_hydro = df_hydro/24
    BPA_hydro[BPA_hydro>45000]=45000
    #Remove CAISO bad_years
    BPA_hydro = pd.DataFrame(np.reshape(BPA_hydro.values, (365,1200), order='F'))
    BPA_hydro.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
        
    if len(sequence) > 1:
        BPA_hydro.columns = np.arange(0, len(BPA_hydro.columns))
        BPA_hydro = BPA_hydro.loc[:, sequences.iloc[:,0]]            

    BPA_hydro = pd.DataFrame(np.reshape(BPA_hydro.values, (365*int(len(BPA_hydro.columns))), order='F'))    
    
    length = int(len(BPA_hydro)/365 + 1)

    # Yearly resources other than hydro (aMW)
    df_resources = pd.read_excel('net_rev_data.xlsx',sheet_name='BP Net Resources',skiprows=[0,1])#, usecols=[12])
    Nuc_y = df_resources.loc[7, y]
    Wind_y = df_resources.loc[8, y]
    Purch_y = df_resources.loc[10, y]

    # Yearly costs and monthly rates (Oct-Sep)
    costs = pd.read_excel('net_rev_data.xlsx',sheet_name='costs', skiprows = [0])#,skiprows=[0,3,4,5])
    #costs = 2229980 * pow(10,3) 
    costs = costs.loc[2,'AVERAGE'] * pow(10,3) ## 1 for the 2.2B
    costs_y = np.repeat(costs, length)
    
    # Yearly borrowing authority outlays
    ba_capex = 0#pd.read_excel('net_rev_data.xlsx',sheet_name='costs', header = 1).loc[4, 'AVERAGE']
    
    PF_rates = pd.read_excel('net_rev_data.xlsx',sheet_name='PF_rates',skiprows=np.arange(13,31))
    PF_rates = PF_rates[['month', y]]
    IP_rates = pd.read_excel('net_rev_data.xlsx',sheet_name='IP_rates',skiprows=np.arange(13,31))
    IP_rates = IP_rates[['month', y]]
        
    #load BPAT hourly demand and wind and convert to daily
    df_synth_load = pd.read_csv('CAPOW_data/Sim_hourly_load.csv', usecols=[1])
    BPAT_load = pd.DataFrame(np.reshape(df_synth_load.values, (24*365,1200), order='F'))
    base = dt(2001, 1, 1)
    arr = np.array([base + timedelta(hours=i) for i in range(24*365)])
    BPAT_load.index=arr
    BPAT_load = BPAT_load.resample('D').mean()
    BPAT_load.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
    
    if len(sequence) > 1:
        BPAT_load.columns = np.arange(0, len(BPAT_load.columns))
        BPAT_load = BPAT_load.loc[:, sequences.iloc[:,0]]  

    BPAT_load = pd.DataFrame(np.reshape(BPAT_load.values, (365*int(len(BPAT_load.columns))), order='F'))

    df_synth_wind = pd.read_csv('CAPOW_data/wind_power_sim.csv', usecols=[1])
    BPAT_wind = pd.DataFrame(np.reshape(df_synth_wind.values, (24*365,1200), order='F'))
    BPAT_wind.index = arr
    BPAT_wind = BPAT_wind.resample('D').mean()
    BPAT_wind.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
    
    if len(sequence) > 1:
        BPAT_wind.columns = np.arange(0, len(BPAT_wind.columns))
        BPAT_wind = BPAT_wind.loc[:, sequences.iloc[:,0]]  

    BPAT_wind = pd.DataFrame(np.reshape(BPAT_wind.values, (365*int(len(BPAT_wind.columns))), order='F'))

    # Calculate daily BPAT proportions for demand and wind
    load_ratio = BPAT_load/BPAT_load.mean()
    wind_ratio = BPAT_wind/BPAT_wind.mean()

    # Derive daily BPA loads and other resources
    PF_load = pd.DataFrame(PF_load_y*load_ratio)
    PF_load_avg = (np.reshape(PF_load.values, (365, length - 1), order='F')).sum(axis=0).mean()
    IP_load = pd.DataFrame(IP_load_y*load_ratio)
    IP_load_avg = (np.reshape(IP_load.values, (365, length - 1), order='F')).sum(axis=0).mean()
    ET_load = pd.DataFrame(ET_load_y*load_ratio)
    Purch = pd.DataFrame(Purch_y*load_ratio)
    Wind = pd.DataFrame(Wind_y*wind_ratio)
    Nuc = pd.DataFrame(data=np.ones(len(Wind))*Nuc_y, index=Wind.index)
    
    # STOCHASTIC MIdC and California daily prices
    MidC=pd.read_csv('CAPOW_data/MidC_daily_prices_new.csv').iloc[:, 1:]
    MidC=MidC.iloc[:,0]
    MidC=pd.DataFrame(np.reshape(MidC.values, (365,1200), order='F'))
    # reshuffle 
#    MidC[[0, 121, 826, 212]]=MidC[[826, 212, 0, 121]]
    MidC.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
    
    if len(sequence) > 1:
        MidC.columns = np.arange(0, len(MidC.columns))
        MidC = MidC.loc[:, sequences.iloc[:,0]]  
        
    MidC=pd.DataFrame(np.reshape(MidC.values, (365*int(len(MidC.columns))), order='F'))
    
    CAISO=pd.read_csv('CAPOW_data/CAISO_daily_prices.csv').iloc[:, 1:]
    #reshuffle
#    CAISO[['0', '121', '826', '212']]=CAISO[['826', '212', '0', '121']]

    if len(sequence) > 1:
        CAISO.columns = np.arange(0, len(CAISO.columns))
        CAISO = CAISO.loc[:, sequences.iloc[:,0]]  

    CAISO = pd.DataFrame(np.reshape(CAISO.values, (365*len(CAISO.columns)), order='F'))
    
    Wholesale_Mkt=pd.concat([MidC,CAISO], axis=1)
    Wholesale_Mkt.columns=['MidC','CAISO']
                       

    # Extra regional discount and Transmission Availability
    ExR=0.71
    TA=1000

    ## interest rate for losses under the independent strategy (Strategy 1)
    bpa_wacc = pd.read_excel('../Hist_data/BPA_debt.xlsx', sheet_name = 'WAI')
    nf_int_rate = bpa_wacc[bpa_wacc['Name'] == 'Non-federal Total']['Average'].values[0]

    ##Calculate revenue
    start = pd.read_excel('net_rev_data.xlsx', sheet_name='res_ba')

    start_res = 0#start.loc[0, y]*pow(10,6)
 
    starting_BA = 0#start.loc[2, y]*pow(10,9)
#    print('STARTING BA: ' + str(starting_BA))
    new_BA = 0 
       
    if infinite == False:
        Treas_fac1 = 0   # Treasury facility (1)
        Treas_fac2 = 0   # Treasury facility (2)
    else: 
        Treas_fac1 = 320*pow(10,6)   # Treasury facility (1)
        Treas_fac2 = 430*pow(10,6)   # Treasury facility (2)
            
    # before contributed to BA from transmission -- now isolating contribution
    # from power 
    trans_BA = 0 # 9.782*pow(10,6)*0.4 #40 percent contribution to BA from transmission line
    Used_TF = 0
    Used_TF2 = pd.DataFrame(index=np.arange(1, length))
    Used_TF2.loc[:,0] = 0 #used TF over the 20 year enesemble    
    
    trans_losses = 3*(Wind + BPA_hydro + Nuc)/100; #consider 3% transmission losses, total BPA resources
    BPA_res = pd.DataFrame(data=(Wind + BPA_hydro + Purch + Nuc)-trans_losses) 
    #Calculate Surplus/Deficit given BPA resources BP_res
    SD = pd.DataFrame(data = BPA_res - (PF_load + IP_load + ET_load))

    months=pd.date_range('2001-1-1','2001-12-31', freq='D').strftime('%B').tolist()
    months= np.transpose(matlib.repmat(months, 1, length-1))
    PF_load['month'] = months
    IP_load['month'] = months

    #initialize
    BPA_rev_d = pd.DataFrame(index = PF_load.index)
    BPA_Net_rev_y = pd.DataFrame(index = np.arange(1, length))
    BPA_Net_rev_y2 = pd.DataFrame(index = np.arange(1, length))
    PF_rev = pd.DataFrame(index = PF_load.index)
    IP_rev = pd.DataFrame(index = IP_load.index)
    P = pd.DataFrame(index = SD.index)
    SS = pd.DataFrame(index = SD.index)
    Reserves = pd.DataFrame(index = np.arange(1, length))
    Reserves.loc[1,0] = 0 #start_res
    TF = pd.DataFrame(index = np.arange(1, length), columns=['TF1','TF2'])
    TF.loc[:,:] = 0
    repaid = pd.DataFrame(index = np.arange(1, length), columns=['repaid'])
    repaid.loc[:,:] = 0
    TTP = pd.DataFrame(index = np.arange(1, length), columns=['TTP'])
    TTP.loc[:] = True
    Remaining_BA = pd.DataFrame(index = np.arange(1, length))
    Remaining_BA.loc[1,0] = starting_BA
    new_BA_y = pd.DataFrame(index = np.arange(1, length))
    new_BA_y.loc[1,0] = 0
    
    amortized = pd.DataFrame(index = np.arange(1, 21 + time_horizon), 
                             columns = [0]) ## will only use if p2 = 'all'

    CRAC = 0
    CRAC_y = pd.DataFrame(index = np.arange(1, length))
    CRAC_y.loc[:,0]=0
    CRAC_rev = pd.DataFrame(index = np.arange(1, length))
    CRAC_rev.loc[:,0] = 0

    avg_customer_prices = pd.DataFrame(index = PF_load.index)

    #Create DataFrame list to hold results
    Result_ensembles_y = {} 
    Result_ensembles_d = {} 

    e = 1 ## ensemble number

    
## not including financial reserves distribution (aka RDC) because 
## up to administrator how used -- may not go to ratepayers 
###############################################

## also not including new revenue financing policy 
    for i in SD.index:
            #daily simulation
            # Calculate revenue from Obligations
            
            ## pull the rates month by month 
            RatePF = PF_rates[str(y)][PF_rates['month']==months[i,0]].values 
            ## when the tariff adjustment is triggered, add that on top of the monthly rate
            #RatePF += CRAC
            ## revenues from each segment are load * the rate * 24 because 
            ## these are hourly 
            PF_rev.loc[i,0] = PF_load.loc[i,0]*RatePF*24
            
            ## same for industrial customers
            RateIP = IP_rates[str(y)][IP_rates['month']==months[i,0]].values 
            #RateIP += CRAC
            IP_rev.loc[i,0] = IP_load.loc[i,0]*RateIP*24
            
            weight_PF = PF_load.loc[i,0]/(IP_load.loc[i,0] + PF_load.loc[i,0])
            avg_customer_prices.loc[i,0] = weight_PF*RatePF + (1 - weight_PF)*RateIP
        
       # Calculate Surplus/Deficit revenue
            if SD.loc[i,0] < 0:
               ## if you have a deficit (SD < 0), then you 
               ## buy on either CAISO or Mid-C, based on where
               ## is cheaper, and the opposite with a surplus
               if Wholesale_Mkt.loc[i,'CAISO'] > Wholesale_Mkt.loc[i,'MidC']:
                       P.loc[i,0] = SD.loc[i,0]*Wholesale_Mkt.loc[i,'MidC']*24
               else:
                       P.loc[i,0] = SD.loc[i,0]*Wholesale_Mkt.loc[i,'CAISO']*24
               SS.loc[i,0] = 0
            else:
                   P.loc[i,0] = 0
                   if Wholesale_Mkt.loc[i,'CAISO'] > Wholesale_Mkt.loc[i,'MidC']:
                       Ex = min(SD.loc[i,0],TA)
                   else:
                       Ex = 0
                   SS.loc[i,0] = ExR*(Ex* Wholesale_Mkt.loc[i,'CAISO']*24) + (SD.loc[i,0]-Ex)*Wholesale_Mkt.loc[i,'MidC']*24
           
           ## daily revenue is just the sum of PF and IP revenue with
           ## any secondary sales, plus any purchases (which are negative $)
            BPA_rev_d.loc[i,0] = PF_rev.loc[i,0] + IP_rev.loc[i,0] + SS.loc[i,0] + P.loc[i,0]
        #   print(BPA_rev_d.loc[i,0])
        
        ## yearly simulation aggregates the revenues and then subtracts
        ## any annual expenses and takes into consideration risk management tools
            if ((i+1)/365).is_integer():
                year = int((i+1)/365)
                print(f"{name}_{year}")# + ' payout: ' + str(df_payout.iloc[year-1,0]))
                bol = year%2 == 0 
                PF_load_i = PF_load.iloc[(year-1)*365:year*365,0].sum()
                IP_load_i = IP_load.iloc[(year-1)*365:year*365,0].sum()
                BPA_Net_rev_y.loc[year,0] = (BPA_rev_d.loc[i-364:i,0]).sum() + df_payout.iloc[year-1,0] - costs_y[year-1]
                
             
                if year%20 == 0:
                    Result_ensembles_y['ensemble' + str(e)] = pd.DataFrame(data= np.stack([Reserves.loc[year-19:year,0],TTP.loc[year-19:year,'TTP'],
                                                                                           Remaining_BA.loc[year-19:year,0], TF.loc[year-19:year,'TF1'], 
                                                                                           TF.loc[year-19:year,'TF2'], CRAC_y.loc[year-19:year,0],
                                                                                           BPA_Net_rev_y.loc[year-19:year,0], new_BA_y.loc[year-19:year,0], 
                                                                                           repaid.loc[year-19:year,'repaid'], Used_TF2.loc[year-19:year,0]], axis=1), 
                                                                        columns=['Reserves','TTP','BA','TF1','TF2','CRAC','Net_Rev','add_BA', 'repaid', 'used_BA'])
                    
                    Result_ensembles_y['ensemble' + str(e)].reset_index(inplace=True, drop=True)
                    Result_ensembles_d['ensemble' + str(e)]=pd.DataFrame(np.stack([BPA_rev_d.iloc[(year-19)*365:year*365,0],PF_rev.iloc[(year-19)*365:year*365,0], 
                                                                       IP_rev.iloc[(year-19)*365:year*365,0] ,  P.iloc[(year-19)*365:year*365,0], SS.iloc[(year-19)*365:year*365,0] ],axis=1),
                                                                       columns=['Rev_gross','PF_rev','IP_rev','P','SS'])
                    Result_ensembles_d['ensemble' + str(e)].reset_index(inplace=True, drop=True)
                    #initialize new ensemble
                    e+=1
                    Reserves.loc[year+1,0] = start_res
                    #print('CRAC reset')
                    print(f"Cum Used TF: ${Used_TF:,}")
                    Used_TF = 0    #initialize treasury facility
#                    Remaining_BA.loc[year+1,0] = starting_BA  #Initialize Remaining borrowing authority 
                    new_BA_y.loc[year+1,0] = 0
                    Used_TF2.loc[year+1, 0] = 0

                    costs_y[year:] = np.repeat(costs_y[0], len(costs_y[year:]))

    print(name)
    print()
    print(f"mean NR1: {BPA_Net_rev_y.iloc[:,0].mean():,}")
    #print(f"mean NR2: {BPA_Net_rev_y2.iloc[:,0].mean():,}")
    print(f"95% VaR1: {np.percentile(BPA_Net_rev_y.iloc[:,0], 5):,}")
#    print(f"95% VaR2: {np.percentile(BPA_Net_rev_y2.iloc[:,0], 5):,}")

#Save results
    Results_d=pd.DataFrame(np.stack([BPA_rev_d[0],PF_rev[0],IP_rev[0],P[0],SS[0], \
                                     BPA_hydro[0],PF_load[0],IP_load[0],SD[0],BPA_res[0], \
                                     Wholesale_Mkt['MidC'],Wholesale_Mkt['CAISO'], \
                                     avg_customer_prices[0]],axis=1),
                        columns=['Rev_gross','PF_rev','IP_rev','P','SS','BPA_hydro', \
                                 'PF_load','IP_load','Surplus/Deficit','BPA_resources', \
                                 'MidC','CAISO', 'avg_prices'])
    
    if excel == True: 
        with pd.ExcelWriter('Results//BPA_net_rev_stoc_d' + name + '.xlsx' ) as writer:
            Results_d.to_excel(writer, sheet_name='Results_d')
            for e in range (1,60):
                Result_ensembles_d['ensemble' + str(e)].to_excel(writer, sheet_name='ensemble' + str(e))
    
        with pd.ExcelWriter('Results//BPA_net_rev_stoc_y' + name + '.xlsx' ) as writer:
            for e in range (1,60):
                Result_ensembles_y['ensemble' + str(e)].to_excel(writer, sheet_name='ensemble' + str(e))
            #costs_y.to_excel(writer,sheet_name='Costs_y')

    if excel == 'long': 
        SD.to_csv("Results//Long//SD_" + str(name) + '.csv')
        SS.to_csv("Results//Long//SS_" + str(name) + '.csv')

        CRAC_y.to_csv("Results//Long//CRAC_" + str(name) + '.csv')
        repaid.to_csv("Results//Long//repaid_" + str(name) + '.csv')
        Used_TF2.to_csv("Results//Long//used_tf2_" + str(name) + '.csv')
        
    BPA_rev_d.to_csv('Results//daily_net_rev_'+str(name)+'.csv')

#    BPA_NR = pd.concat([BPA_Net_rev_y, BPA_Net_rev_y2], axis = 1)
    BPA_Net_rev_y.to_csv('Results//ann_net_rev_'+str(name)+'.csv')


    return BPA_Net_rev_y, repaid

df_payout = pd.DataFrame([0]*1189*10)
bpa_wacc = pd.read_excel('../Hist_data/BPA_debt.xlsx', sheet_name = 'WAI')
ba_int_rate = bpa_wacc[bpa_wacc['Name'] == 'US Treasury ']['Average'].values[0]
nf_int_rate = bpa_wacc[bpa_wacc['Name'] == 'Non-federal Total']['Average'].values[0]
y = 'no_loc'
repay_ba = 'no'
infinite = False
custom_redux = 0
time_horizon = 50
name = 'no_tools'


nr_no_crac_no_loc, repay_no_crac_no_loc = net_rev_full_inputs(df_payout, custom_redux = 0, 
                                              name = 'no_tools_10k', excel = 'long',
                                              y = 'no_loc', repay_ba = 'no', 
                                              ba_int_rate = ba_int_rate, 
                                              infinite = False,
                                              sequence = sequences)






















