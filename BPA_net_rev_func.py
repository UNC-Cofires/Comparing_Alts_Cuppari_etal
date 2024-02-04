# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 18:09:28 2023

@author: rcuppari
"""

import pandas as pd
from datetime import datetime as dt
from datetime import timedelta  
import numpy as np
import numpy.matlib as matlib

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
## net payout is calculated including the premium (so net to BPA) 
## Note: this means that it is postive below the strike and negative above the strike\
## p2 = % pos net revenues used to repay BA
## p = % pos net revenues used to replenish reserves
## repay_ba says whether or not there is a line of credit ("yes" or "no")

# import random
# sequences = pd.DataFrame()
# for i in range(0, 10): 
#     sample = pd.DataFrame(random.sample(range(0, 1180), 1180))
#     sequences = pd.concat([sequences, sample])    

#sequences = sequences[~sequences.iloc[:,0].isin([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191])]
#sequences.to_csv("random_sequences.csv")

def net_rev_full_inputs(df_payout, custom_redux = 0, name = '', excel = True, 
                        y = 'AVERAGE', infinite = False, 
                        p = 10, p2 = 32, res_cap = 608691000, repay_ba = 'yes',
                        int_rate = 0.042, time_horizon = 50, drop = True, sequence = [0]):
    
    
    # Yearly firm loads (aMW)
    df_load=pd.read_excel('net_rev_data.xlsx',sheet_name='load',skiprows=[0,1])
    PF_load_y = df_load.loc[13, y] - custom_redux*df_load.loc[13 ,y]
    IP_load_y = df_load.loc[3, y] - custom_redux* df_load.loc[3, y]
    ET_load_y = df_load.loc[14, y]

    # Hourly hydro generation from FCRPS stochastic simulation
    df_hydro = pd.read_csv('CAPOW_data/new_BPA_hydro_daily.csv')
    df_hydro = df_hydro['gen']

    BPA_hydro = df_hydro/24
    BPA_hydro[BPA_hydro>45000]=45000
    #Remove CAISO bad_years
    BPA_hydro = pd.DataFrame(np.reshape(BPA_hydro.values, (365, int(len(df_hydro)/365)), order='F'))

    sim_years = len(BPA_hydro.columns)

    if drop == True:
        BPA_hydro.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191], axis=1, inplace=True)
    
    if len(sequence) > 1:
        BPA_hydro.columns = np.arange(0, len(BPA_hydro.columns))
        BPA_hydro = BPA_hydro.loc[:, sequences.iloc[:,0]]            

    BPA_hydro = pd.DataFrame(np.reshape(BPA_hydro.values, (365*int(len(BPA_hydro.columns))), order='F'))    
    
    length = int(len(BPA_hydro)/365 + 1)

    # Yearly resources other than hydro (aMW)
    df_resources = pd.read_excel('net_rev_data.xlsx',sheet_name='BP Net Resources',skiprows=[0,1])
    Nuc_y = df_resources.loc[7, y]
    Wind_y = df_resources.loc[8, y]
    Purch_y = df_resources.loc[10, y]

    # Yearly costs and monthly rates (Oct-Sep)
    costs = pd.read_excel('net_rev_data.xlsx',sheet_name='costs', skiprows = [0])#,skiprows=[0,3,4,5])
    #costs = 2229980 * pow(10,3) 
    costs = costs.loc[2,'AVERAGE'] * pow(10,3)
    costs_y = np.repeat(costs, length)
    
    ## Yearly borrowing authority outlays
    ## we set to zero (in contrast to Denaro et al) to focus purely on the costs of the borrowing
    ## authority as a risk management strategy 
    ba_capex = 0
    
    PF_rates = pd.read_excel('net_rev_data.xlsx',sheet_name='PF_rates',skiprows=np.arange(13,31))
    PF_rates = PF_rates[['month', y]]
    IP_rates = pd.read_excel('net_rev_data.xlsx',sheet_name='IP_rates',skiprows=np.arange(13,31))
    IP_rates = IP_rates[['month', y]]
        
    #load BPAT hourly demand and wind and convert to daily
    df_synth_load = pd.read_csv('CAPOW_data/Sim_hourly_load.csv', usecols=[1])
    BPAT_load = pd.DataFrame(np.reshape(df_synth_load.values, (24*365,sim_years), order='F'))
    base = dt(2001, 1, 1)
    arr = np.array([base + timedelta(hours=i) for i in range(24*365)])
    BPAT_load.index=arr
    BPAT_load = BPAT_load.resample('D').mean()
    
    if drop == True: 
        BPAT_load.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
           
    if len(sequence) > 1:
        BPAT_load.columns = np.arange(0, len(BPAT_load.columns))
        BPAT_load = BPAT_load.loc[:, sequences.iloc[:,0]]  

    BPAT_load = pd.DataFrame(np.reshape(BPAT_load.values, (365*int(len(BPAT_load.columns))), order='F'))

    df_synth_wind = pd.read_csv('CAPOW_data/wind_power_sim.csv', usecols=[1])
    BPAT_wind = pd.DataFrame(np.reshape(df_synth_wind.values, (24*365,sim_years), order='F'))
    BPAT_wind.index = arr
    BPAT_wind = BPAT_wind.resample('D').mean()
    
    if drop == True: 
        BPAT_wind.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)   
    
    if len(sequence) > 1:
        BPAT_wind.columns = np.arange(0, len(BPAT_wind.columns))
        BPAT_wind = BPAT_wind.loc[:, sequences.iloc[:,0]]  

    BPAT_wind = pd.DataFrame(np.reshape(BPAT_wind.values, (365*int(len(BPAT_wind.columns))), order='F'))

    # Calculate daily BPAT proportions for demand and wind
    load_ratio = BPAT_load/BPAT_load.mean()
    wind_ratio = BPAT_wind/BPAT_wind.mean()

    # Derive daily BPA loads and other resources
    PF_load = pd.DataFrame(PF_load_y*load_ratio.iloc[:(length-1)*365,:])
#    PF_load_avg = (np.reshape(PF_load.values, (365, length - 1), order='F')).sum(axis=0).mean()
    IP_load = pd.DataFrame(IP_load_y*load_ratio.iloc[:(length-1)*365,:])
#    IP_load_avg = (np.reshape(IP_load.values, (365, length - 1), order='F')).sum(axis=0).mean()
    ET_load = pd.DataFrame(ET_load_y*load_ratio.iloc[:(length-1)*365,:])
    Purch = pd.DataFrame(Purch_y*load_ratio.iloc[:(length-1)*365,:])
    Wind = pd.DataFrame(Wind_y*wind_ratio.iloc[:(length-1)*365,:])
    Nuc = pd.DataFrame(data=np.ones(len(Wind))*Nuc_y, index=Wind.index)
    
    # STOCHASTIC MIdC and California daily prices
    MidC=pd.read_csv('CAPOW_data/MidC_daily_prices_new.csv').iloc[:, 1:]
    MidC=MidC.iloc[:,0]
    MidC=pd.DataFrame(np.reshape(MidC.values, (365, int(len(MidC)/365)), order='F'))

    if drop == True: 
        MidC.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
    
    if len(sequence) > 1:
        MidC.columns = np.arange(0, len(MidC.columns))
        MidC = MidC.loc[:, sequences.iloc[:,0]]  

    MidC=pd.DataFrame(np.reshape(MidC.values, (365*int(len(MidC.columns))), order='F'))
    
    CAISO = pd.read_csv('CAPOW_data/CAISO_daily_prices.csv').iloc[:, 1:]

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

    start_res = start.loc[0, y]*pow(10,6)
 
    starting_BA = start.loc[2, y]*pow(10,9)
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
    BPA_Net_rev_y2 = pd.DataFrame(index = np.arange(1, length), columns = ['NR2'])
    PF_rev = pd.DataFrame(index = PF_load.index)
    IP_rev = pd.DataFrame(index = IP_load.index)
    P = pd.DataFrame(index = SD.index)
    SS = pd.DataFrame(index = SD.index)
    Reserves = pd.DataFrame(index = np.arange(1, length))
    Reserves.loc[1,0] = start_res
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

    ## ADD AMORTIZATION DFS
    amortized = pd.DataFrame(index = np.arange(1, 21 + time_horizon), 
                             columns = ['amort']) ## will only use if p2 != 0
    amortized.iloc[:,:] = 0 
    amort_schedule = pd.DataFrame(index = np.arange(1, 21 + time_horizon)) 
    amort_schedule.iloc[:,:] = 0 
    
    CRAC = 0
    CRAC_y = pd.DataFrame(index = np.arange(1, length))
    CRAC_y.loc[:,0]=0
    CRAC_rev = pd.DataFrame(index = np.arange(1, length))
    CRAC_rev.loc[:,0] = 0

    avg_customer_prices = pd.DataFrame(index = PF_load.index)

    #Create DataFrame list to hold results
    Result_ensembles_y = {} 
    Result_ensembles_d = {} 

    ## amortization set up
    num = int_rate * pow((1 + int_rate), time_horizon) ## 30 = debt repayment time horizon
    deno = pow((1 + int_rate), time_horizon) - 1
    res_add = 0
    
    e = 1

    def calculate_CRAC(NR_, tot_load, name):
        if NR_ > 5*pow(10,6):  
            if NR_ > 100*pow(10,6):
                NR1=100*pow(10,6)
                NR2=(NR_ - 100*pow(10,6))/2
            else: 
                NR1 = NR_
                NR2= 0
            X=min((NR1+NR2)/(tot_load*24) ,  300*pow(10,6)/(tot_load*24))
        else:
            X=0
        return X      

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
            RatePF += CRAC
            ## revenues from each segment are load * the rate * 24 because 
            ## these are hourly 
            PF_rev.loc[i,0] = PF_load.loc[i,0]*RatePF*24
            
            ## same for industrial customers
            RateIP = IP_rates[str(y)][IP_rates['month']==months[i,0]].values 
            RateIP += CRAC
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
                print(f"year: {year}")
#                print(str(name) + '_' + str(year) + ' payout: ' + str(df_payout.iloc[year-1,0]))
                bol = year%2 == 0
                PF_load_i = PF_load.iloc[(year-1)*365:year*365,0].sum()
                IP_load_i = IP_load.iloc[(year-1)*365:year*365,0].sum()
                tot_load_i = PF_load_i + IP_load_i
                BPA_Net_rev_y.loc[year,0] = (BPA_rev_d.loc[i-364:i,0]).sum() + \
                                            df_payout.iloc[year-1,0] - costs_y[year-1]
                Used_TF = Used_TF*bol
                
                ## a primary change from Denaro et al., 2022 -- adding the impact of 
                ## repaying the line of credit or replenishing reserves for calculating 
                ## the net revenues. Structure of the calculation of debt to be added
                ## (i.e., TF used) remains the same 
                                
                ## when have negative net revenues
                if BPA_Net_rev_y.loc[year,0] < 0:
                    ## absolute value of losses
                    losses = -BPA_Net_rev_y.loc[year,0]
                    ## try to use reserves to cover it (Res avail - absolute(losses))
                    Net_res1 = Reserves.loc[year,0] - losses
                    
                    ## reserves for next year set at 0 if all used or, 
                    ## if there were enough reserves to cover document the remainder of reserves 
                    Reserves.loc[year+1,0] = max(Net_res1, 0)
#                    print(f'Remaining Reserves {year}: {Reserves.loc[year+1,0]}')
                    
                    ## when reserves < losses... 
                    if int(Net_res1 < 0):
                        ## the thus far uncovered losses are the difference
                        BPA_Net_rev_y2.loc[year,'NR2'] = Net_res1 #BPA_Net_rev_y.loc[year,0] + Reserves.loc[year,0]
                        ## make losses absolute number here again
                        losses = -Net_res1    
                        ## compensate for what the reserves cover 
                        
                        #if there is BA AND TF
                        if repay_ba == 'yes':
                            print(f'year  {year}, reserves depleted to cover losses')

                            ## NOTE: next try adding below change to reset amount in Used_TF

                            if Used_TF <= 750*pow(10,6): 
#                            if (Remaining_BA.loc[year,0] - Used_TF) >= 750*pow(10,6): 
#                                print(f"year {year}, Rem_BA > Used TF")
                                ## losses here are a positive number (abs value) 
                                ## next year's TF is set to the minimum of losses 
                                ## or $320M (TF1) minus this year's existing TF
                                ## bol is because it is replenished every two years
                                TF.loc[year+1,'TF1'] = min(losses, Treas_fac1 - TF.TF1[year]*bol) ## TF1 used
                                print(f"year {year}, TF1: {TF.TF1[year]*bol}")
                                ## note how much of the TF is used 
                                Used_TF += TF.loc[year+1,'TF1']
                                Used_TF2.loc[year, 0] = TF.loc[year+1,'TF1']  
                                                                
                                if (Treas_fac1-TF.TF1[year]*bol - losses) < 0:
                                    CRAC += calculate_CRAC(losses, tot_load_i, name)
                                    #set max crac as +5$/MWh per ensemble
                                    CRAC = min(CRAC, 5)
                                    TF.loc[year+1,'TF2'] = min(losses, Treas_fac2-TF.TF2[year]*bol)
                                    Used_TF += TF.loc[year+1,'TF2']
                                    Used_TF2.loc[year, 0] += TF.loc[year+1,'TF2']
                                    
                                    ##if used is greater than available, than TTP is False 
                                    if (Treas_fac2-TF.TF2[year]*bol - losses) <= 0:
                                        TTP.loc[year] = False
                                        losses = -(Treas_fac2-TF.TF2[year]*bol - losses)
                                        #print(f"line 334 losses: {losses}")
                                        BPA_Net_rev_y2.loc[year,'NR2']  = -losses
                                        
                                        ## Want to add losses as costs for next year 
                                        ## because they don't just disappear into the night... 
                                        losses_to_pay = losses*(1 + nf_int_rate)
                                        costs_y[year] += losses_to_pay
                                        
                                    else: 
                                        BPA_Net_rev_y2.loc[year,'NR2']  = 0
                                else: 
                                    #losses = 0 #-(TF.loc[year+1, 'TF1'] - losses) 
                                    BPA_Net_rev_y2.loc[year,'NR2']  = 0
                                print(f"year {year}, Used TF: {Used_TF2.loc[year,0]}")
                                Remaining_BA.loc[year+1, 0] = max(0, Remaining_BA.loc[year, 0] - Used_TF2.loc[year,0])
                                
                              #  print(f"Used TF in {year}: {Used_TF}")
                            else:
                                print('Warning: depleted borrowing authority and deferred TP')
                                CRAC += calculate_CRAC(losses, tot_load_i, name)
                                #set max crac as +5$/MWh per ensemble
                                CRAC = min(CRAC, 5)
                                TTP.loc[year] = losses
                                Used_TF2.loc[year, 0] = 0 ## no TF to use 
                                Remaining_BA.loc[year+1,0] = Remaining_BA.loc[year,0]
                                TF.loc[year+1,'TF2'] = TF.loc[year,'TF2']
                                TF.loc[year+1,'TF1'] = TF.loc[year,'TF1']
                        else: 
                        ## if have no BA, then mitigated NR is just the losses post reserves
                        ## BUT, do want to add that to the costs for next year 
                        ## because they don't just disappear into the night... 
                            losses_to_pay = losses*(1 + nf_int_rate)
                            costs_y[year] += losses_to_pay
                            
                            CRAC += calculate_CRAC(losses, tot_load_i, name)
                            #set max crac as +5$/MWh per ensemble
                            CRAC = min(CRAC, 5)
                            Remaining_BA.loc[year+1,0] = 0
                            
                    else: 
                        ## all losses covered, so set net rev to 0  
                        BPA_Net_rev_y2.loc[year,'NR2']  = 0# BPA_Net_rev_y.loc[year,0] + losses           
                        ## and don't need to use the BA
                        Remaining_BA.loc[year+1,0] = Remaining_BA.loc[year,0]
                        Used_TF2.loc[year, 0] = 0 ## no TF to use 

                    
                    ## need to calculate the amortization of new debt 
                    ## and keep track of accumulating debt
                    ## new obligation going forward, to account for skipped year
                    ## but just going to load it all onto the next year
                    if Used_TF2.loc[year,0] > 0: 
                        new_oblig = Used_TF2.loc[year,0] * (num/deno) ## if used_tf = 0, then new_oblig should = 0
                    else: new_oblig = 0 
                    
                    if p2 != 0: ## if repay 0%, then no "deferral", just ignore
                        deferred = amortized.iloc[year%20-1, 0]                                    
                        ## and take deferred out of last year, otherwise it 
                        ## is double counted for the analysis 
                        amortized.iloc[year%20-1, 0] -= deferred 
                           
                        ## just add deferred obligation directly to next year
                        amortized.iloc[year%20, 0] += deferred * (1 + int_rate)
                        
                    ## then add new obligation for the long term
                    amortized.iloc[(year%20):(year%20 + time_horizon), 0] += new_oblig                                 
                  #  print(f"New debt obligation: {new_oblig}")
                    print()
                    
                else:## when net revs > 0 
                    ## no need for mitigation from BA/Reserves
                    if repay_ba == 'yes':
                 
                    
                        ## $608,691,000 is the Reserves cap
                        ## subtract TF1 because TF1 is part of the "cash on hand" and treated as part of reserves
                        Reserves.loc[year+1,0] = min(Reserves.loc[year,0] + 0.01*p*BPA_Net_rev_y.loc[year,0], \
                                                     res_cap - Treas_fac1)
                            
                        ## remove from net revenues whatever was contributed to reserves
                        res_add = max(0, (Reserves.loc[year+1,0] - Reserves.loc[year,0]))
    
                        ## remove from NR whatever was repaid to BA
                        BPA_Net_rev_y2.loc[year,'NR2']  = BPA_Net_rev_y.loc[year,0] - res_add

    
                        ## .01 is just for the post processing units sake ('000s instead of M) 
                        ## ASSUME WON'T USE RESERVES TO REPAY TREASURY 
                        ## by setting max, essentially only doing this during pos NR years
                  
                         ## add how much was repaid in each given year 
                        ## the curious thing about repayment is that it really just feeds back into the BA, since the BA is indefinite

                        
                        if p2 == 'all':                             
                            ## then, need to repay as much as possible (within BPA's remaining net revs) 
                            ## so, a minimum of the full amount, or whatever BPA has left after replenishing reserves
                            repaid.loc[year, 'repaid'] = min(amortized.iloc[year%20-1, 0], \
                                                             BPA_Net_rev_y2.loc[year,'NR2'] ) 
                            
                        elif p2 > 0: ## if 0% < p2 < 100%-p, pay up to amortized amount or p2% of net revenues
                            ## because using a percentage of the annual net revenues, use 
                            ## BPA_Net_rev_y instead of BPA_Net_rev_y2, which excludes res add (which is a % of NR)
                            repaid.loc[year,'repaid'] = min(amortized.iloc[year%20-1,0], 
                                                            .01*p2*BPA_Net_rev_y2.loc[year,'NR2'])                           
                        else: 
                            repaid.loc[year, 'repaid'] = 0
                               
                        ## remove from NR whatever was repaid to BA
                        BPA_Net_rev_y2.loc[year,'NR2']  = BPA_Net_rev_y2.loc[year,'NR2'] - repaid.loc[year, 'repaid']                    

                        ## if they can't repay the full debt service, then it needs to get kicked 
                        ## to the next year (essentially take out a new loan and re-amortize that)
                        if p2 != 0: 
                            deferred = max(0, \
                                           amortized.iloc[year%20-1,0] - repaid.loc[year, 'repaid'])
                            
                            if deferred > 0: 
                                print(f"deferred in {year}: ${round(deferred,2):,}")  
                                print()
                                amortized.iloc[year%20-1, 0] -= deferred
                                        
                                ## just add deferred obligation directly to next year, with some interest
                                amortized.iloc[(year%20), 0] += deferred*(1+int_rate)


                        ## let's assume the minimum repayment is the last three years' average 
                        ## -- then later we can compare the repaid column
                        ## with this value 

                        Remaining_BA.loc[year+1,0] = max(0, Remaining_BA.loc[year,0] + \
                                                         trans_BA - ba_capex + \
                                                         repaid.loc[year,'repaid'])

                    else: ## in case we set the BA to zero and keep it there
                        Remaining_BA.loc[year+1,0] = 0
                        Reserves.loc[year+1,0] = min(Reserves.loc[year,0] + 0.01*p*BPA_Net_rev_y.loc[year,0], res_cap)  

                    
                        ## remove from net revenues whatever was contributed to reserves
                        res_add = max(0, (Reserves.loc[year+1,0] - Reserves.loc[year,0]))
    #                    print(f'Added {res_add} in year {year}, \nRemaining Reserves: {Reserves.loc[year+1,0]}')
    
                        ## remove from NR whatever was repaid to BA
                        BPA_Net_rev_y2.loc[year,'NR2']  = BPA_Net_rev_y.loc[year,0] - res_add - repaid.loc[year, 'repaid']                    
                    
                ## assume CRAC never goes down. So, if CRAC triggered because of losses in previous years, 
                ## it remains for the following years 
                CRAC_y.loc[year+1] = CRAC 

#                print("pre-mitigation NR: $" + str(BPA_Net_rev_y.loc[year,0]))
#                print("mitigated NR: $" + str(BPA_Net_rev_y2.loc[year,0]))
                
                 ## replenish BA if it dips below the $1B threshold 
                if (infinite == True) & (Remaining_BA.loc[year+1,0] < 1.5*pow(10,9)): 
                    print(f"year {year}: adding $2B to BA")
                    Remaining_BA.loc[year+1,0] = Remaining_BA.loc[year,0] + 1.5*pow(10,9)
                    ## the cumulative BA added will be the sum of all years with added BA (always add $2B)
                    new_BA_y.loc[year+1,0] = 1.5*pow(10,9)  
                else: 
                    new_BA_y.loc[year+1,0] = 0
                
                #print(Remaining_BA.loc[year+1,0])
                #ensembles     
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
                    CRAC = 0 
                    CRAC_y.loc[year+1] = 0
                    Used_TF=0    #initialize treasury facility
                    Remaining_BA.loc[year+1,0] = starting_BA  #Initialize Remaining borrowing authority 
                    new_BA_y.loc[year+1,0] = 0
                    Used_TF2.loc[year+1, 0] = 0
                    amort_schedule = pd.concat([amort_schedule, amortized], axis = 1)
                    ## and then reset
                    amortized.iloc[:,:] = 0
                    ## reset costs moving forward (otherwise keeping adding from a higher baseline)
                    costs_y[year:] = np.repeat(costs_y[0], len(costs_y[year:]))


    print()
    print(f"mean NR1: {BPA_Net_rev_y.iloc[:,0].mean():,}")
    print(f"mean NR2: {BPA_Net_rev_y2.iloc[:,0].mean():,}")
    print(f"95% VaR1: {np.percentile(BPA_Net_rev_y.iloc[:,0], 5):,}")
    print(f"95% VaR2: {np.percentile(BPA_Net_rev_y2.iloc[:,0], 5):,}")

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
        
    BPA_rev_d.to_csv('Results//Long//daily_net_rev_'+str(name)+'.csv')
    amort_schedule.to_csv('Results//Long//amort_sched_'+str(name)+'.csv')

    print('completed ' + name)

    BPA_NR = pd.concat([BPA_Net_rev_y, BPA_Net_rev_y2], axis = 1)
    BPA_NR.to_csv('Results//Long//ann_net_rev_'+str(name)+'.csv')
    

    return BPA_NR, repaid


bpa_wacc = pd.read_excel('../Hist_data/BPA_debt.xlsx', sheet_name = 'WAI')
ba_int_rate = bpa_wacc[bpa_wacc['Name'] == 'US Treasury ']['Average'].values[0]
nf_int_rate = bpa_wacc[bpa_wacc['Name'] == 'Non-federal Total']['Average'].values[0]

## SAMPLE RUN: 
res_cap = ((608.6*pow(10,6))) ## cap on reserves held
repay_ba = 'yes' ## repayment of borrowing authority/line of credit?
name = 'test' ## name for saving files 
y = 'AVERAGE' ## which starting conditions to use (e.g., '2018', '2010'). Average = 2018-2022 average
custom_redux = 0 ## reduction in number of preference customers (i.e., in fixed demand)
p = 10 ## percent of positive net revenues that are used to replenish reserves each year 
p2 = 'all' ## percent of positive net revenues that are put towards repayment of the line of credit 
infinite = True ## forever expanding BA? 
excel = False ## save excel file?
time_horizon = 50 ## time horizon for debt amortization 
int_rate = ba_int_rate ## interest rate to calculate amortization payments
drop = True ## need to drop years that did not fully run from the original CAPOW data? 
sim_years = 1200 ## number of input years 
sequences = pd.DataFrame(pd.read_csv("random_sequences.csv").iloc[:,1]) ## shuffling data to create 1189 * 10 runs 

df_payout = pd.DataFrame([0]*sim_years) ## insurance payouts
BPA_NR, repaid = net_rev_full_inputs(df_payout, custom_redux = custom_redux, 
                                              name = name, res_cap = res_cap,  
                                              excel = excel, time_horizon = time_horizon,
                                              infinite = infinite, p2 = p2, p = p, 
                                              y = y, repay_ba = repay_ba, 
                                              int_rate = ba_int_rate, sequence = sequences,
                                              sim_years = sim_years)

