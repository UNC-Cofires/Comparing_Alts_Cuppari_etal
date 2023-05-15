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
## p2 = % pos net revenues used to repay BA
## p = % pos net revenues used to replenish reserves
def net_rev_full_inputs(df_payout, custom_redux = 0, name = '', excel = True, y = 2018, infinite = False, 
                        reserve_fund = False, p = 10, p2 = 32, res_cap = 608691000):
    #Set Preference Customers reduction percent (number)
            
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
    
    # Yearly borrowing authority outlays
    ba_capex = pd.read_excel('net_rev_data.xlsx',sheet_name='costs', header = 1).loc[4, 'AVERAGE']
    
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

    ##Calculate revenue
    start = pd.read_excel('net_rev_data.xlsx', sheet_name='res_ba')

    if reserve_fund == False: 
        start_res = start.loc[0, y]*pow(10,6)
    else:
        start_res = reserve_fund * pow(10,6)
    starting_BA = start.loc[2, y]*pow(10,9)
    new_BA = 0 
    ## NEED TO ADD CONDITION THAT IF INFINITE, EXPAND BA AS NEEDED 
    ## AND ALSO TRACK EXPANSIONS 
       
    Treas_fac1 = 320*pow(10,6)   # Treasury facility (1)
    Treas_fac2 = 430*pow(10,6)   # Treasury facility (2)
    # before contributed to BA from transmission -- now isolating contribution
    # from power 
    trans_BA = 0 # 9.782*pow(10,6)*0.4 #40 percent contribution to BA from transmission line
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

    #initialize

    BPA_rev_d = pd.DataFrame(index = PF_load.index)
    BPA_Net_rev_y = pd.DataFrame(index = np.arange(1, length))
    BPA_Net_rev_y2 = pd.DataFrame(index = np.arange(1, length))
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

    res_p = pd.DataFrame(index = np.arange(1, length))
    res_p.loc[:,0] = 0

    CRAC = 0
    CRAC_y = pd.DataFrame(index = np.arange(1, length))
    CRAC_y.loc[:,0]=0
    CRAC_rev = pd.DataFrame(index = np.arange(1, length))
    CRAC_rev.loc[:,0] = 0

    #Create DataFrame list to hold results
    Result_list = ['ensemble' + str(e) for e in range(1,60,1)]
    Result_ensembles_y = {} 
    Result_ensembles_d = {} 

#    p = 30  # percent surplus that goes to reserve
#    p2 = 32  # percent surplus that goes to debt opt
    d = 1
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
            RatePF = PF_rates[str(y)][PF_rates['month']==months[i,0]].values 
        #    RatePF += CRAC
            PF_rev.loc[i,0] = PF_load.loc[i,0]*RatePF*24
            RateIP = IP_rates[str(y)][IP_rates['month']==months[i,0]].values 
        #    RateIP += CRAC
            IP_rev.loc[i,0] = IP_load.loc[i,0]*RateIP*24
        
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
            BPA_rev_d.loc[i,0] = PF_rev.loc[i,0] + IP_rev.loc[i,0] + SS.loc[i,0] + P.loc[i,0]
         #   print(BPA_rev_d.loc[i,0])
        #yearly simulation
        
            if ((i+1)/365).is_integer():
                year = int((i+1)/365)
                print(str(name) + '_' + str(year))
                bol = year%2 == 0 
                PF_load_i = PF_load.iloc[(year-1)*365:year*365,0].sum()
                IP_load_i = IP_load.iloc[(year-1)*365:year*365,0].sum()
                tot_load_i = PF_load_i + IP_load_i
                BPA_Net_rev_y.loc[year,0] = (BPA_rev_d.loc[i-364:i,0]).sum() + df_payout[year-1] - costs_y[year-1]
                #print(BPA_Net_rev_y.loc[year,0])
                #print(df_payout[year-1])
                ## when have negative net revenues
                if int(BPA_Net_rev_y.loc[year,0]<0):
                    losses = -BPA_Net_rev_y.loc[year,0]
                    ## try to use reserves to cover it (Res avail - absolute(losses))
                    Net_res1 = Reserves.loc[year,0] - losses
                    ## reserves for next year set at 0 or, if there were enough reserves to cover
                    ## document the remainder of reserves 
                    Reserves.loc[year+1,0] = max(Net_res1, 0)
                    ## when reserves < losses... 
                    if int(Net_res1 < 0):
                        ## compensate for what the reserves cover 
                        BPA_Net_rev_y2.loc[year,0] = BPA_Net_rev_y.loc[year,0] + Reserves.loc[year,0]
                        ## the thus far uncovered losses are the difference
                        losses=-Net_res1
                        #if there is BA AND TF
                        if (Remaining_BA.loc[year,0] - Used_TF) > 750*pow(10,6): 
                            ## losses here are a positive number (abs value) 
                            ## next year's TF is set to the minimum of losses 
                            ## or $320M (TF1) minus this year's existing TF
                            ## bol is because it is replenished every two years
                            TF.loc[year+1,'TF1'] = min(losses, Treas_fac1 - TF.TF1[year]*bol)
                            
                            ## note how much of the TF is used 
                            Used_TF += TF.loc[year+1,'TF1']
                            Used_TF2.loc[year, 0] = TF.loc[year+1,'TF1']
                            
                            ## offset losses by Used TF
                            BPA_Net_rev_y2.loc[year,0] = BPA_Net_rev_y.loc[year,0] + TF.loc[year+1,'TF1']
                            
                            if (Treas_fac1-TF.TF1[year]*bol - losses)<0:
                                losses= - (Treas_fac1-TF.TF1[year]*bol - losses)
                                CRAC += calculate_CRAC(losses, tot_load_i, name)
                                #set max crac as +5$/MWh per ensemble
                                CRAC = min(CRAC, 5)
                                TF.loc[year+1,'TF2'] = min(losses , Treas_fac2-TF.TF2[year]*bol)
                                Used_TF += TF.loc[year+1,'TF2']
                                Used_TF2.loc[year, 0] = Used_TF2.loc[year, 0] + TF.loc[year+1,'TF2']
                                
                                ##if used is greater than available, than TTP is False 
                                if (Treas_fac2-TF.TF2[year]*bol - losses) <= 0:
                                    TTP.loc[year] = False
                        else:
                            print('Warning: depleted borrowing authority and deferred TP')
                            CRAC+=calculate_CRAC(losses, tot_load_i, name)
                            #set max crac as +5$/MWh per ensemble
                            CRAC=min(CRAC, 5)
                            TTP.loc[year]=losses
                    else: 
                        BPA_Net_rev_y2.loc[year,0] = BPA_Net_rev_y.loc[year,0] + losses           
                    print("res add: $0")
                else:## when net revs > 0 
                    ## $608,691,000 is the Reserves cap
                    ## subtract TF1 because TF1 is part of the "cash on hand" and considered part of reserves
                    Reserves.loc[year+1,0] = min(Reserves.loc[year,0] + 0.01*p*BPA_Net_rev_y.loc[year,0], res_cap-Treas_fac1) 
                    if Reserves.loc[year+1, 0] < 0:
                        Reserves.loc[year+1, 0] = 0
                    
                    ## remove from net revenues whatever was contributed to reserves
                    res_add = (Reserves.loc[year+1,0] - Reserves.loc[year,0])
                    print("res added: $" + str(res_add))
                    BPA_Net_rev_y2.loc[year,0] = BPA_Net_rev_y.loc[year,0] - res_add
                print("pre-mitigation NR: $" + str(BPA_Net_rev_y.loc[year,0]))
                print("mitigated NR: $" + str(BPA_Net_rev_y2.loc[year,0]))
                    #print(Reserves.loc[year+1,0])
                    #Total_TF+= min(0.01*p*BPA_Net_rev_y.loc[year,0], pow(10,9))  #debt optimization
                    #print('Debt optimization, added: ' + str( 0.01*p*BPA_Net_rev_y.loc[year,0]))
                CRAC_y.loc[year+1] = CRAC
                ## a minimum of $484 M goes to reserves, with the debt optimization that 1% * p2 * net revenue goes to TP 
                ## .01 is just for the post processing units sake ('000s instead of M) 
                ## ASSUME WON'T USE RESERVES TO REPAY TREASURY 
                ## by setting max, essentially only doing this during pos NR years
                Remaining_BA.loc[year+1,0] = max(0, Remaining_BA.loc[year,0] + trans_BA - ba_capex, Remaining_BA.loc[year,0] + trans_BA - ba_capex + 0.01*p2*BPA_Net_rev_y.loc[year,0])
                ## add how much was repaid in each given year 
                ## the curious thing about repayment is that it really just feeds back into the BA, since the BA is indefinite
                repaid.loc[year+1,'repaid'] = max(0, .01*p2*BPA_Net_rev_y.loc[year,0])
                    ## let's assume the minimum repayment is the last three years' average 
                    ## -- then later we can compare the repaid column
                    ## with this value 
                ## remove from NR whatever was repaid to BA
                BPA_Net_rev_y2.loc[year,0] = BPA_Net_rev_y2.loc[year,0] - repaid.loc[year+1, 'repaid']
                
                if (infinite == True) & (Remaining_BA.loc[year+1,0] < 1*pow(10,9)): 
                    print("adding $2B to BA")
                    Remaining_BA = Remaining_BA + 2*pow(10,9)
                    ## the cumulative BA added will be the sum of all years with added BA (always add $2B)
                    new_BA_y.loc[year+1,0] = 2*pow(10,9)
                else: 
                    new_BA_y.loc[year+1,0] = 0
                    
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
    
    if excel == True: 
        with pd.ExcelWriter('Results//BPA_net_rev_stoc_d' + name + '.xlsx' ) as writer:
            Results_d.to_excel(writer, sheet_name='Results_d')
            for e in range (1,60):
                Result_ensembles_d['ensemble' + str(e)].to_excel(writer, sheet_name='ensemble' + str(e))
    
        with pd.ExcelWriter('Results//BPA_net_rev_stoc_y' + name + '.xlsx' ) as writer:
            for e in range (1,60):
                Result_ensembles_y['ensemble' + str(e)].to_excel(writer, sheet_name='ensemble' + str(e))
            #costs_y.to_excel(writer,sheet_name='Costs_y')

    BPA_rev_d.to_csv('Results//daily_net_rev_'+str(name)+'.csv')
    # BPA_Net_rev_y.to_csv('Results//ann_net_rev_'+str(name)+'.csv')
    # BPA_Net_rev_y2.to_csv('Results//ann_net_rev2_'+str(name)+'.csv')
    print(name)
    print()
    print('mean NR1: ' + str(BPA_Net_rev_y.mean()))
    print('mean NR2: ' + str(BPA_Net_rev_y2.mean()))
    print('95% VaR1: ' + str(np.percentile(BPA_Net_rev_y, 5)))
    print('95% VaR2: ' + str(np.percentile(BPA_Net_rev_y2, 5)))
    BPA_NR = pd.concat([BPA_Net_rev_y, BPA_Net_rev_y2], axis = 1)
    BPA_NR.to_csv('Results//ann_net_rev_'+str(name)+'.csv')
    return BPA_NR, repaid






df_payout = [0]*1200
#avg21, avg_repaid21 = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'avg_18_22', y = 'AVERAGE', infinite = False)
#og, og_repaid = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'og2', excel = False, y = 2018, infinite = False)
#update = net_rev_full_inputs(df_payout, custom_redux = 0, name = '2021update', excel = True, y = 2022)
#infinite, infinite_repaid = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'infinite', excel = True, y = 'AVERAGE', infinite = True)
#avg_ba = net_rev_full_inputs(df_payout, name = 'avg_ba', y = 'AVERAGE', infinite = True)
#low_res, low_res_repaid = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'res90', y = 'AVERAGE', infinite = False, excel = True, res_cap = (608.6*pow(10,6))*90/120)
#high_res, high_res_repaid = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'res_og', y = 'AVERAGE', infinite = False, excel = True, p2 = 32, p = 15)
#avg = pd.read_csv("Results//ann_net_rev_avg.csv")
#avg2 = pd.read_csv("Results//ann_net_rev_avg_ba.csv")
#avg_no_crac, avg2_no_crac = net_rev_full_inputs(df_payout, custom_redux = 0, name = 'avg_no_crac', y = 'AVERAGE', infinite = False, excel = True, p2 = 32, p = 15)












