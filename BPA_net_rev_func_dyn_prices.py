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

## numbers have been updated to reflect the 2021 Annual Report 
#######################################################################################################################
## this time I have already calculated the net payout with the premium an can input it directly (no separate loading)
## df needs to have the net payment to BPA from counterparty (i.e. positive when below tda strike and negative 
## when above the strike)
## note: when using marginal price to scale payments, don't want to use v :) 
def net_rev_full_inputs(df_payout, custom_redux = 0, name = '', excel = True, y = 2018, 
                        infinite = False, reserve_fund = False, cost_inc = 0):    #Set Preference Customers reduction percent (number) as custom_redux 
       
    p = 10  # percent surplus that goes to reserve
    p2 = 32  # percent surplus that goes to debt opt
    d = 1
    e = 1
    
    # Yearly firm loads (aMW)
    df_load=pd.read_excel('net_rev_data.xlsx',sheet_name='load',skiprows=[0,1])#, usecols=[12]) ## column 12 = 2021
    PF_load_y = df_load.loc[13, y] - custom_redux*df_load.loc[13 ,y]
    IP_load_y = df_load.loc[3, y] - custom_redux* df_load.loc[3, y]
    ET_load_y = df_load.loc[14, y]

    # Hourly hydro generation from FCRPS stochastic simulation
    df_hydro = pd.read_csv('CAPOW_data/new_BPA_hydro_daily.csv')
    df_hydro = df_hydro['gen']
    BPA_hydro = df_hydro/24
    BPA_hydro[BPA_hydro>45000]=45000
    #Remove CAISO bad_years
    BPA_hydro = pd.DataFrame(np.reshape(BPA_hydro.values, (365,1200), order='F'))
    BPA_hydro.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
        
    BPA_hydro = pd.DataFrame(np.reshape(BPA_hydro.values, (365*1188), order='F'))
    
    length = int(len(BPA_hydro)/365 + 1)

    # Yearly resources other than hydro (aMW)
    df_resources = pd.read_excel('net_rev_data.xlsx',sheet_name='BP Net Resources',skiprows=[0,1])#, usecols=[12])
    Nuc_y = df_resources.loc[7, y]
    Wind_y = df_resources.loc[8, y]
    Purch_y = df_resources.loc[10, y]

    # Yearly costs and monthly rates (Oct-Sep)
    costs = pd.read_excel('net_rev_data.xlsx',sheet_name='costs',skiprows=[0,3,4,5])
    costs = costs.loc[0,'AVERAGE'] * pow(10,3)
    costs_y = np.repeat(costs, length - 1)
    
    PF_rates = pd.read_excel('net_rev_data.xlsx',sheet_name='PF_rates',skiprows=np.arange(13,31), usecols=[0,7])
    PF_rates.columns = ['month', str(y)]
    IP_rates = pd.read_excel('net_rev_data.xlsx',sheet_name='IP_rates',skiprows=np.arange(13,31), usecols=[0,7])
    IP_rates.columns = ['month',str(y)]
        
    #load BPAT hourly demand and wind and convert to daily
    df_synth_load = pd.read_csv('CAPOW_data/Sim_hourly_load.csv', usecols=[1])
    BPAT_load = pd.DataFrame(np.reshape(df_synth_load.values, (24*365,1200), order='F'))
    base = dt(2001, 1, 1)
    arr = np.array([base + timedelta(hours=i) for i in range(24*365)])
    BPAT_load.index=arr
    BPAT_load = BPAT_load.resample('D').mean()
    BPAT_load.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
    BPAT_load = pd.DataFrame(np.reshape(BPAT_load.values, (365*1188), order='F'))

    df_synth_wind = pd.read_csv('CAPOW_data/wind_power_sim.csv', usecols=[1])
    BPAT_wind = pd.DataFrame(np.reshape(df_synth_wind.values, (24*365,1200), order='F'))
    BPAT_wind.index = arr
    BPAT_wind = BPAT_wind.resample('D').mean()
    BPAT_wind.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
    BPAT_wind = pd.DataFrame(np.reshape(BPAT_wind.values, (365*1188), order='F'))

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
    
    ##ratio for overall generation (used for exchange resources)
    gen = BPA_hydro+Wind+Purch+Nuc
    gen_ratio = gen/gen.mean()
    
    # STOCHASTIC MIdC and California daily prices
    MidC=pd.read_csv('CAPOW_data/MidC_daily_prices_new.csv').iloc[:, 1:]
    MidC=MidC.iloc[:,0]
    MidC=pd.DataFrame(np.reshape(MidC.values, (365,1200), order='F'))
    # reshuffle 
#    MidC[[0, 121, 826, 212]]=MidC[[826, 212, 0, 121]]
    MidC.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
    MidC=pd.DataFrame(np.reshape(MidC.values, (365*1188), order='F'))
    
    CAISO=pd.read_csv('CAPOW_data/CAISO_daily_prices.csv').iloc[:, 1:]
    #reshuffle
#    CAISO[['0', '121', '826', '212']]=CAISO[['826', '212', '0', '121']]
    CAISO=pd.DataFrame(np.reshape(CAISO.values, (365*1188), order='F'))
    
    Wholesale_Mkt=pd.concat([MidC,CAISO], axis=1)
    Wholesale_Mkt.columns=['MidC','CAISO']
                       

    # Extra regional discount and Transmission Availability
    ExR=0.71
    TA=1000

    # read in other necessary data
    system_cost=pd.read_excel('rate_calc_data.xlsx', sheet_name='COSA')
    loads = pd.read_excel('rate_calc_data.xlsx', sheet_name='Load', header = 2)
    #sales=pd.read_excel('Hist_data/rate_calc_data.xlsx',sheet_name='Second_sales')
    
    exch_load_all=pd.read_excel('rate_calc_data.xlsx',sheet_name='Exchange_Load')
    rate_protect=pd.read_excel('rate_calc_data.xlsx',sheet_name='Rate_Protect') 
    unconstrained_ben=pd.read_excel('rate_calc_data.xlsx',sheet_name='Base_Ex_Cost')
    sd_alloc_all=pd.read_excel('rate_calc_data.xlsx',sheet_name='SD_cost_alloc')
    delta=pd.read_excel('rate_calc_data.xlsx',sheet_name ='IP_PF_delta')

    ##know existing resource capacity
    resources=pd.read_excel('rate_calc_data.xlsx',sheet_name='Resources')
    
    ##assume FBS resource will always be allocated to PF
    ##but give indus/surp some new 
    indus_nr = resources.loc[18,y]
    surp_nr = resources.loc[20,y]
    exch_res = resources.loc[1,y]

    ##benefits from the sale of energy (assume average) 
    unconstrained_ben = pd.read_excel('rate_calc_data.xlsx',sheet_name = 'Base_Ex_Cost')
    uncon_ben = unconstrained_ben.loc[9,y]*1000 
    rate_protect = pd.read_excel('rate_calc_data.xlsx',sheet_name = 'Rate_Protect') 
    rate_protect18 = rate_protect.loc[2,y]*1000 
    sd_alloc = sd_alloc_all.loc[0,y]*1000 ##surplus/deficit allocation ##FIX THIS???

    ##initialize any new DFs
    ## will want to recalculate every 2 years, ie every rate period
    unbif_rate = pd.DataFrame(index=range(0,1189)) ##these are average rates - not split into HLH or LLH
    bif_rate = pd.DataFrame(index=range(0,1189)) ##these are average rates - not split into HLH or LLH
    PF_costs = pd.DataFrame(index=range(0,1189))

    start = pd.read_excel('net_rev_data.xlsx', sheet_name='res_ba')

    if reserve_fund == False: 
        start_res = start.loc[0, y]*pow(10,6)
    else:
        start_res = reserve_fund * pow(10,6)
    starting_BA = start.loc[1, y]*pow(10,9)
    new_BA = 0 

    ## $320 is the in year liquidity required for power 
    Treas_fac1 = 320*pow(10,6)   # Treasury facility (1) -- avail for within year liquidity 
    Treas_fac2 = 430*pow(10,6)   # Treasury facility (2) -- according to power + transmission risk study, avail for year-to-year
    trans_BA = 9.782*pow(10,6)*0.4 #40 percent contribution to BA from transmission line
    Used_TF = 0
    Used_TF2 = pd.DataFrame(index=np.arange(1, length))
    Used_TF2.loc[1,0] = 0 #used TF over the 20 year enesemble    
    trans_losses = 3*(Wind + BPA_hydro + Nuc)/100; #consider 3% transmission losses, total BPA resources
    BPA_res = pd.DataFrame(data=(Wind + BPA_hydro + Purch + Nuc)-trans_losses) 
    #Calculate Surplus/Deficit given BPA resources BP_res
    SD = pd.DataFrame(data = BPA_res - (PF_load + IP_load + ET_load))

    months=pd.date_range('2001-1-1','2001-12-31', freq='D').strftime('%B').tolist()
    months= np.transpose(matlib.repmat(months, 1, length-1))
    PF_load['month'] = months
    IP_load['month'] = months

    PF_load_ann=pd.DataFrame(index=range(0, length), columns=['annual'])
    IP_load_ann=pd.DataFrame(index=range(0, length), columns=['annual'])
#    tot_load_ann=pd.DataFrame(index=range(0, length), columns=['annual'])
    SS_ann=pd.DataFrame(index=range(0, length), columns=['annual'])
    SD_ann=pd.DataFrame(index=range(0, length), columns=['annual'])
    P_ann=pd.DataFrame(index=range(0, length))
    

    # for rate design purposes, BPA always uses 1937 water conditions, so use that as 
    # the resources available
    # calculate own allocations, use historical Mid C prices 

    loads=pd.read_excel('rate_calc_data.xlsx',sheet_name=3)
    sd_alloc_all=pd.read_excel('rate_calc_data.xlsx',sheet_name=10)

    count=0
    hours=8760
    ##currently only calculate annual rates, so just using a ratio to approximate monthly differences 
    mon_ratio=pd.DataFrame(index=range(1,12))
    for i in range(0,12):
        mon_ratio.loc[i,0]=(PF_rates.iloc[i,1])/PF_rates.loc[i,y].mean()

    mon_ratio['month']=PF_rates['month']
    mon_ratio.columns=['ratio','month']

    #initialize standard stuff 
    BPA_rev_d = pd.DataFrame(index = PF_load.index)
    BPA_Net_rev_y = pd.DataFrame(index = np.arange(1, length))
    PF_rev = pd.DataFrame(index = PF_load.index)
    IP_rev = pd.DataFrame(index = IP_load.index, columns = ['rev'])
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
    CRAC = 0
    CRAC_y = pd.DataFrame(index = np.arange(1, length))
    CRAC_y.loc[:,0]=0
    CRAC_rev = pd.DataFrame(index = np.arange(1, length))
    CRAC_rev.loc[:,0] = 0

    #Create DataFrame list to hold results
    Result_list = ['ensemble' + str(e) for e in range(1,60,1)]
    Result_ensembles_y = {} 
    Result_ensembles_d = {} 


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
    
    for i in SD.index:
            print(i)
            if i <366:
        #daily simulation
        # Calculate revenue from Obligations
                RatePF = (PF_rates[str(y)][PF_rates['month']==months[i,0]].values)[0]
                #RatePF += CRAC
                PF_rev.loc[i,0] = PF_load.loc[i,0]*RatePF*24
                RateIP = (IP_rates[str(y)][IP_rates['month']==months[i,0]].values)[0]
                RateIP += CRAC
                IP_rev.iloc[i,0] = IP_load.loc[i,0]*RateIP*24
        
            #daily simulation
            # Calculate revenue from Obligations
#            RatePF = PF_rates[str(y)][PF_rates['month']==months[i,0]].values 
#            RatePF += CRAC
#            PF_rev.loc[i,0] = PF_load.loc[i,0]*RatePF*24
#            RateIP = IP_rates[str(y)][IP_rates['month']==months[i,0]].values 
#            RateIP += CRAC
#           IP_rev.loc[i,0] = IP_load.loc[i,0]*RateIP*24
        
        # Calculate Surplus/Deficit revenue
            if SD.loc[i,0] < 0:
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
            BPA_rev_d.loc[i,0] = PF_rev.loc[i,0] + IP_rev.iloc[i,0] + SS.loc[i,0] + P.loc[i,0]
         #   print(BPA_rev_d.loc[i,0])
         
        # yearly simulation
            if ((i+1)/365).is_integer():
                year = int((i+1)/365)
                print(str(name) + '_' + str(year))
                bol = year%2 == 0 
                PF_load_i = PF_load.iloc[(year-1)*365:year*365,0].sum()
                IP_load_i = IP_load.iloc[(year-1)*365:year*365,0].sum()
                tot_load_i = PF_load_i + IP_load_i
                BPA_Net_rev_y.loc[year,0] = (BPA_rev_d.loc[i-364:i,0]).sum() + df_payout[year-1] - costs_y[year-1]
                               
                
               # print(BPA_Net_rev_y.loc[year,0])
                ## when have negative net revenues
                if int(BPA_Net_rev_y.loc[year,0]<0):
                    losses = -BPA_Net_rev_y.loc[year,0]
                    ## try to use reserves to cover it (Res avail - absolute(losses))
                    ## so, net_res1 > 0 means reserves were sufficient
                    Net_res1 = Reserves.loc[year,0] - losses
                    ## net reserves = actual reserves minus TF, which is counted as reserves
                    ## so basically just the additional reserves (power & transmission risk study)
                    
                    ## reserves for next year set at 0 or, if there were enough reserves to cover
                    ## document the remainder of reserves 
                    Reserves.loc[year+1,0] = max(Net_res1, 0)
                    
                    ## when reserves < losses... 
                    if int(Net_res1 < 0):
                        
                        ## the thus far uncovered losses are the difference
                        losses = - Net_res1
                        
                        ## if there is any liquidity (i.e. there is enough BA
                        ## to contribute to TF)
                        if (Remaining_BA.loc[year,0] - Used_TF) > 750*pow(10,6): 
                            ## next year's TF is set to the minimum of losses 
                            ## or $320M (TF1) minus this year's existing TF
                            ## bol is because it is replenished every two years
                            TF.loc[year+1,'TF1'] = min(losses, Treas_fac1 - TF.TF1[year]*bol)
                            
                            ## note how much of the TF is used 
                            Used_TF += TF.loc[year+1,'TF1']
                            Used_TF2.loc[year, 0] = TF.loc[year+1,'TF1']
                            
                            if (Treas_fac1-TF.TF1[year]*bol - losses)<0:
                                losses= - (Treas_fac1-TF.TF1[year]*bol - losses)
                                CRAC+=calculate_CRAC(losses, tot_load_i, name)
                                #set max crac as +5$/MWh per ensemble
                                CRAC=min(CRAC, 5)
                                TF.loc[year+1,'TF2'] = min(losses , Treas_fac2-TF.TF2[year]*bol)
                                Used_TF += TF.loc[year+1,'TF2']
                                Used_TF2.loc[year, 0] = Used_TF2.loc[year, 0] + TF.loc[year+1,'TF2']
                                
                                ##if used is greater than available, than TTP is False 
                                if (Treas_fac2-TF.TF2[year]*bol - losses) <= 0:
                                    TTP.loc[year]=False
                        else:
                            print('Warning: depleted borrowing authority and deferred TP')
                            CRAC+=calculate_CRAC(losses, tot_load_i, name)
                            #set max crac as +5$/MWh per ensemble
                            CRAC=min(CRAC, 5)
                            TTP.loc[year]=losses
                             


                else:
                    ## $608,691,000 is the Reserves cap
                    ## subtract TF1 because TF1 is part of the "cash on hand"
                    Reserves.loc[year+1,0] = min(Reserves.loc[year,0] + 0.01*p*BPA_Net_rev_y.loc[year,0], 608691000-Treas_fac1) 
                    #Total_TF+= min(0.01*p*BPA_Net_rev_y.loc[year,0], pow(10,9))  #debt optimization
                    #print('Debt optimization, added: ' + str( 0.01*p*BPA_Net_rev_y.loc[year,0]))
                
                CRAC_y.loc[year+1]=CRAC
                ## a minimum of $484 M goes to reserves, with the debt optimization that 1% * p2 * net revenue goes to TP 
                ## .01 is just for the post processing units sake ('000s instead of M) 
                Remaining_BA.loc[year+1,0] = max(0, Remaining_BA.loc[year,0] - 484*pow(10,6) + trans_BA, Remaining_BA.loc[year,0] - 484*pow(10,6) + trans_BA + 0.01*p2*BPA_Net_rev_y.loc[year,0])
                ## add how much was repaid in each given year 
                ## the curious thing about repayment is that it really just feeds back into the BA, since the BA is indefinite
                repaid.loc[year+1,'repaid'] = max(0, .01*p2*BPA_Net_rev_y.loc[year,0])
                    ## let's assume the minimum repayment is the last three years' average 
                    ## -- then later we can compare the repaid column
                    ## with this value 
                
                if (infinite == True) & (Remaining_BA.loc[year+1,0] < 1*pow(10,9)): 
                    print("adding $2B to BA")
                    Remaining_BA = Remaining_BA + 2*pow(10,9)
                    ## the cumulative BA added will be the sum of all years with added BA (always add $2B)
                    new_BA_y.loc[year+1,0] = 2*pow(10,9)
                else: 
                    new_BA_y.loc[year+1,0] = 0
 
                FBS_res = BPA_res.iloc[(year-1)*365:year*365,0].sum()
                
#               PF_load_ann.iloc[(year),0] = PF_load_i/365
                SS_ann.iloc[year,0] = SS.iloc[(year-1)*365:year*365,0].sum()
                SD_ann.iloc[year,0] = SD.iloc[(year-1)*365:year*365,0].sum()
                surp_load = SD.iloc[(year-1)*365:year*365,0].sum()
                P_ann.loc[year,0] = P.iloc[(year-1)*365:year*365,0].sum()
                ET_load_ann = ET_load.iloc[(year-1)*365:year*365,0].sum()
                
                tot_PF = PF_load_i + ET_load_ann.sum()
                tot_load_ann = tot_PF + IP_load_i + SS_ann.iloc[year,0] 
                
                ##different customer groups are given "allocations" for the various costs 
                num_FBS_PF = min(tot_PF, FBS_res)
                deno_FBS_PF = (num_FBS_PF + indus_nr + surp_nr)
    
                exch_num=min(tot_PF - PF_load_i, FBS_res)     
                exch_res_IP=(exch_res + FBS_res - exch_num - num_FBS_PF)*IP_load_i/(IP_load_i+surp_load)
    
#                sales_mw = sales.loc[1,year] this is SS
                ##calculate allocations
                alloc_FBSNR = num_FBS_PF/deno_FBS_PF ##base system
                alloc_ER = ET_load_ann/exch_res ##exchange resources 
                alloc_cons = tot_PF/tot_load_ann ##conservation

                alloc_sd = tot_PF/(tot_load_ann - surp_load) ##surplus deficit
                alloc_IP_FBSNR = indus_nr/(indus_nr + surp_nr + PF_load_i)
                alloc_IP_ER = exch_res_IP/exch_res
                alloc_IP_cons = IP_load_i/(IP_load_i + tot_PF + surp_load)
                alloc_IP_nr = indus_nr/(indus_nr+surp_nr)
#                alloc_IP_sd = 1-alloc_sd
#                alloc_IP_rp = IP_load/(ET_load_ann + IP_load_i + surp_load + SD_ann.iloc[year,0])
    
                ##benefits from the sale of energy (assume average) 
#                uncon_ben = unconstrained_ben.loc[9,y]
                rate_protect2 = rate_protect.loc[2,y]
                sd_alloc = sd_alloc_all.loc[0,y] ##surplus/deficit allocation
                ##now take their yly cost and divide by total PF load
                PF_costs = (system_cost.loc[0,y]+system_cost.loc[1,y])*alloc_FBSNR+(system_cost.loc[2,y]-system_cost.loc[5,y])*alloc_ER+system_cost.loc[3,y]*alloc_cons+sd_alloc-SS_ann.iloc[year,0]*alloc_FBSNR
                        
                IP_costs = system_cost.loc[1,y]*alloc_IP_nr+(system_cost.loc[2,y]-system_cost.loc[5,y])*alloc_IP_ER+system_cost.loc[3,y] *alloc_IP_cons+system_cost.loc[4,y]*alloc_IP_cons-SS_ann.iloc[year,0]*alloc_IP_FBSNR#delta.loc[0,year] ##no allocation for FBS
                ##remove transmission costs from exchange 
                ##the first step is to get the unbifurcated PF (average)
                unbif_rate.loc[year,0] = PF_costs/(tot_PF*(hours/1000))
                
                ##adjust PF costs
                ##benefits from the sale of energy (assume average) 
                adj_PF_costs = (PF_costs*PF_load_i/(tot_PF))-rate_protect2
                ##have simplified - skipped WP-10 step and rate links
                
                bif_rate.loc[year,0] = adj_PF_costs/(PF_load_i*hours/1000)
                print(year)
    
                RatePF = bif_rate.iloc[year,0]*mon_ratio['ratio'][mon_ratio['month']==months[i,0]].values
                PF_rev.loc[year,0]=PF_load.loc[year,0]*RatePF[0]*24
                RateIP = bif_rate.iloc[year]*1.19*mon_ratio['ratio'][mon_ratio['month']==months[i,0]].values
                IP_rev.loc[year,0] = IP_load.iloc[year,0]*RateIP[0]*24
            
                if year%20 == 0:
                    ##now take their yearly cost and divide by total PF load
                    PF_costs = ((system_cost.loc[0,y]+system_cost.loc[1,y])*alloc_FBSNR+(system_cost.loc[2,y]-system_cost.loc[5,y])*alloc_ER+system_cost.loc[3,y]*alloc_cons)*1000+ losses*alloc_FBSNR-(SS_ann.iloc[year,0]+P_ann.iloc[year,0])*alloc_FBSNR+sd_alloc*alloc_sd
                    ##costs are in billions: full number
                    count=0
                else: 
                    ## now take their yearly cost and divide by total PF load
                    ## removed sd alloc
                    PF_costs = ((system_cost.loc[0,y]+system_cost.loc[1,y])*alloc_FBSNR+(system_cost.loc[2,y]-system_cost.loc[5,y])*alloc_ER + system_cost.loc[3,y]*alloc_cons)*1000*(1+cost_inc)**count + losses*alloc_FBSNR-(SS_ann.iloc[year,0]+P_ann.iloc[year,0])*alloc_FBSNR+sd_alloc*alloc_sd
                    ##costs are in billions: full number 
                    count = count+1
                
                        ##the first step is to get the unbifurcated PF (average)
                  
                    ##adjust PF costs
                adj_PF_costs = (PF_costs*PF_load_ann.iloc[year,0]/(tot_PF))-rate_protect18
                ##have simplified - skipped WP-10 step and rate links
                bif_rate.loc[year,0] = adj_PF_costs/(PF_load_i/365*hours)
                print(bif_rate.loc[year,0])
                bif_rate.loc[year,1] = count
                        
    
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
                    Reserves.loc[year+1,0]=start_res
                    CRAC=0 
                    CRAC_y.loc[year+1]=0
                    Used_TF=0    #initialize treasury facility
                    Remaining_BA.loc[year+1,0] = starting_BA  #Initialize Remaining borrowing authority 
                    new_BA_y.loc[year+1,0] = 0
                    repaid.loc[year+1, 0] = 0
                    Used_TF2.loc[year+1, 0] = 0


#Save results
    Results_d=pd.DataFrame(np.stack([BPA_rev_d[0],PF_rev[0],IP_rev[0],P[0],SS[0],BPA_hydro[0],PF_load[0],IP_load[0],SD[0],BPA_res[0],Wholesale_Mkt['MidC'],Wholesale_Mkt['CAISO']],axis=1),
                        columns=['Rev_gross','PF_rev','IP_rev','P','SS','BPA_hydro','PF_load','IP_load','Surplus/Deficit','BPA_resources','MidC','CAISO' ])

    with pd.ExcelWriter('Results//BPA_net_rev_stoc_d' + name + '.xlsx' ) as writer:
        Results_d.to_excel(writer, sheet_name='Results_d')
        for e in range (1,60):
            Result_ensembles_d['ensemble' + str(e)].to_excel(writer, sheet_name='ensemble' + str(e))

    with pd.ExcelWriter('Results//BPA_net_rev_stoc_y' + name + '.xlsx' ) as writer:
        for e in range (1,60):
            Result_ensembles_y['ensemble' + str(e)].to_excel(writer, sheet_name='ensemble' + str(e))
        #costs_y.to_excel(writer,sheet_name='Costs_y')

    BPA_rev_d.to_csv('Results//daily_net_rev_'+str(name)+'.csv')
    BPA_Net_rev_y.to_csv('Results//ann_net_rev_'+str(name)+'.csv')
    print(name)
    print()
    print(BPA_Net_rev_y.mean())
    
    return BPA_Net_rev_y, repaid


df_payout = [0]*1200
#avg, avg_repaid = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'avg', y = 'AVERAGE', infinite = False, excel = False)
#og, og_repaid = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'og2', excel = False, y = 2018, infinite = False)
#update = net_rev_full_inputs(df_payout, custom_redux = 0, name = '2021update', excel = True, y = 2022)
#infinite, infinite_repaid = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'infinite', excel = True, y = 'AVERAGE', infinite = True)
#avg_ba = net_rev_full_inputs(df_payout, name = 'avg_ba', y = 'AVERAGE', infinite = True)
#high_res, high_res_repaid = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'high_res_500', y = 'AVERAGE', infinite = False, excel = True, reserve_fund = 500)
#avg = pd.read_csv("Results//ann_net_rev_avg.csv")
#avg2 = pd.read_csv("Results//ann_net_rev_avg_ba.csv")
dyn, dyn_repaid = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'avg', y = 'AVERAGE', infinite = False, excel = False)

#sns.kdeplot(og.iloc[:,0], label = 'og', fill = True, alpha = .5)
#sns.kdeplot(infinite.iloc[:,0], label = 'infinite', fill = True, alpha = .5)
#sns.kdeplot(avg.iloc[:,0], label = 'avg', fill = True, alpha = .5)
#plt.legend()

