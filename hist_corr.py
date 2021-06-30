# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:47:30 2020

@author: rcuppari
"""
##MAKE SURE TDA IN IS TDA ARF AND NOT TDA M OR L###
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
#################################################################
##correlations with low demand & generation & net interchange
hist_load=pd.read_csv("Hist_data/hist_ann_load.csv")
hist_load=hist_load.iloc[0:7,:]

hist_gen=pd.read_csv("Hist_data/hist_ann_hydro.csv")
hist_gen=hist_gen.iloc[0:7,:]

hist_inter=pd.read_csv("Hist_data/hist_ann_interchange.csv")
hist_inter=hist_inter.iloc[0:7,:]

#################################################################
##correlations with low revenue? 
##use historical revenue 2012-2019 (since Simona did collect it) 
hist_rev=pd.read_excel("net_rev_data.xlsx",sheet_name=6,header=1)
hist_rev=hist_rev.iloc[:,0:7]
hist_net_rev=hist_rev.iloc[7,:]
hist_rev=hist_rev.iloc[2,:]


hist_load=hist_load.set_index(hist_rev.index)
hist_gen=hist_gen.set_index(hist_rev.index)
hist_inter=hist_inter.set_index(hist_rev.index)

np.corrcoef(hist_rev,hist_load.iloc[:,1])
np.corrcoef(hist_rev,hist_gen.iloc[:,1])
plt.scatter(hist_rev,hist_gen.iloc[:,1])

np.corrcoef(hist_load.iloc[:,1],hist_gen.iloc[:,1])
np.corrcoef(hist_inter.iloc[:,1],hist_gen.iloc[:,1])
np.corrcoef(hist_rev,hist_inter.iloc[:,1])

hist=pd.concat([hist_rev,hist_load.iloc[:,1],hist_gen.iloc[:,1]],axis=1)
hist.columns=['rev','load','hydro']

X=pd.concat([hist_inter.iloc[:,1],hist_gen.iloc[:,1]],axis=1)
reg=LinearRegression().fit(X,hist_rev)
reg.score(X,hist_rev)
##ok, obvious - interchange and hydro generation can predict revenues fairly well
##interestingly - load doesn't really play a big factor here (pretty invariable I think)

##now let's get streamflow at the dams
##in particular, Simona's work suggests that Dalles, Hungry Horse, and Dworschak 
## are the most influential. Simona had already compiled a few 
inflows=pd.read_excel("../../../Simona_Work/Columbia_BPA_financial_risk/DATA/Correlation/inputs/inflows_BPA_2003_18.xlsx")
#inflows.index=pd.to_datetime(inflows.index)
inflows['year']=inflows['Date Time'].astype(str).str[:4]
inflows['TDA_in']=inflows['TDA_in']*1000
inflows['HGH_in']=inflows['HGH_in']*1000

inflows_ann=inflows.groupby('year').agg({'ALF_in':'mean','BON_in':'mean','CHJ_in':'mean',
                                         'DWR_in':'mean','GCL_in':'mean','HGH_in':'mean',
                                         'IHR_in':'mean','JDA_in':'mean','LGS_in':'mean',
                                         'LMN_in':'mean','LWG_in':'mean','MCN_in':'mean',
                                         'TDA_in':'mean','Total_in':'mean'})
inflows_ann['year']=inflows_ann.index

sns.kdeplot(inflows_ann['TDA_in'],label="Flows at the Dalles",shade=True)
plt.xticks([100000,150000,200000,250000],[100000,150000,200000,250000])
plt.xlabel("Modified flows ft3/s")
plt.ylabel("Probability")

inflows03=inflows_ann[inflows_ann['year'].astype(int)>=2003]

inflows_ann=inflows_ann[inflows_ann['year'].astype(int)>2011]
inflows_ann=pd.DataFrame(inflows_ann)

hist_gen['year']=hist_rev.index
tda=inflows_ann['TDA_in'].reset_index()
tda.index=hist_gen.index
hist_gen2=pd.concat([hist_gen,tda],axis=1)

hist_conf=pd.DataFrame([[2012,2013,2014,2015,2016,2017,2018],[9909,8559,8918,7983,8485,9350,9014]])
hist_conf=hist_conf.T
hist_conf.index=hist_gen.index
hist_gen2['true']=hist_conf.iloc[:,1]
      

plt.scatter(hist_gen2['TDA_in'],hist_net_rev)
plt.title("Historical TDA vs. Generation")
plt.xlabel("TDA Inflows")
plt.ylabel("Rev")

plt.scatter(hist_gen2['TDA_in'],hist_gen2['true'])
plt.title("Historical TDA vs. Generation")
plt.xlabel("TDA Inflows")
plt.ylabel("Generation (MW)")

##presumably would rather this be 1937 min, not min from 2012 to 2018
##https://www.nwcouncil.org/reports/columbia-river-history/hydropower
inflows_ann['TDA>min']=inflows_ann['TDA_in']-39160

##I think weighted temperatures might also be successful, using cities > 150k
##deno = pops of Seattle+Spokane+Tacoma+Vancouver+Portland+Salem+Eugene+Boise
##source: US Census Bureau pop. estimate avg 2010-2019
misc_data=pd.read_excel("Hist_data/misc_data.xlsx",sheet_name=0,header=1)
misc_data.index=misc_data.iloc[:,0]
deno=misc_data.loc['Total','Average']
boise=misc_data.loc['Boise','Average']/deno
##salem and eugene are so close together that I'm just using the Salem airport temperature data
##same with seattle-tacoma and portland-vancouver
eug_sal=(misc_data.loc['Eugene','Average']+misc_data.loc['Salem','Average'])/deno
portland_van=(misc_data.loc['Portland','Average']+misc_data.loc['Vancouver','Average'])/deno
seattle_tac=(misc_data.loc['Seattle','Average']+misc_data.loc['Tacoma','Average'])/deno
spokane=misc_data.loc['Spokane','Average']/deno
weights=pd.DataFrame([boise,portland_van,eug_sal,seattle_tac,spokane]).T

temps=pd.read_excel("Hist_data/misc_data.xlsx",sheet_name=1)
temps.index=temps['DATE']
temps.index=pd.to_datetime(temps.index)

#
temps['Month']=temps.index.month
temps['Year']=temps.index.year

temps=temps.replace({'BOISE AIR TERMINAL, ID US':'boise',
                     'SALEM AIRPORT MCNARY FIELD, OR US':'salem',
                     'SEATTLE TACOMA AIRPORT, WA US':'seattle',
                     'SPOKANE INTERNATIONAL AIRPORT, WA US':'spokane',
                     'PORTLAND TROUTDALE AIRPORT, OR US':'portland'})

names=['boise','salem','spokane','seattle','portland']
boise=temps[temps['NAME']=='boise']
boise.columns=['STATION','NAME','DATE','BOI_AVG1','BOI_MAX','BOI_MIN','BOI_AVG','Month','Year']
salem=temps[temps['NAME']=='salem']
salem.columns=['STATION','NAME','DATE','SAL_AVG1','SAL_MAX','SAL_MIN','SAL_AVG','Month','Year']
seattle=temps[temps['NAME']=='seattle']
seattle.columns=['STATION','NAME','DATE','SEA_AVG1','SEA_MAX','SEA_MIN','SEA_AVG','Month','Year']
portland=temps[temps['NAME']=='portland']
portland.columns=['STATION','NAME','DATE','POR_AVG1','POR_MAX','POR_MIN','POR_AVG','Month','Year']

temps2=pd.concat([salem.loc[:,['Month','Year','SAL_AVG','SAL_MAX','SAL_MIN']],boise.loc[:,['BOI_AVG','BOI_MAX','BOI_MIN']],seattle.loc[:,['SEA_AVG','SEA_MAX','SEA_MIN']],portland.loc[:,['POR_AVG','POR_MAX','POR_MIN']]],axis=1)

mon_temps=temps2.groupby(['Year','Month']).agg({'sum'})
mon_temps.reset_index(inplace=True)
mon_temps.columns=['year','month','sal_avg','sal_max','sal_min','boi_avg','boi_max','boi_min','sea_avg','sea_max','sea_min','por_avg','por_max','por_min']
#months=['1','2','3','4','5','6','7','8','9','10','11','12']
#temps_mon=temps.groupby('NAME')[['Avg','TMAX','TMIN','Month','Year']].apply(lambda x: x.set_index('Year','Month').to_dict()).to_dict()

from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
##want to see which months' data are best predictors for historical revenue 
#for m in months: 
#    for n in names: 
#        y=hist_rev
#        X=pd.DataFrame(temps_mon[n]['Avg'])

#        X=temps_mon[(temps_mon['NAME']==n) & (temps_mon['Month']==m)]
#        X=X['Avg']        
#                [n]['Avg'].Values.ToArray()
#       X_train,X_test,y_train,y_test=tts(X,hist_rev,test_size=.2)
        
#        est=sm.OLS(y_train,X_train)
#        est2=est.fit()
#        pred=est.predict(x_test)
#        print(str(m)+str(n)+':'+ est2.rsquared)
#        print(str(m)+str(n)+'r with test set'+str(r2(pred,y_test)))

##or even flow
for i in range(1,13):
    X=mon_temps[mon_temps['month']==i]
    X=sm.add_constant(X)
    X.index=X['year']
    X_train,X_test,y_train,y_test=tts(X,hist_rev,test_size=.2)
    est=sm.OLS(y_train,X_train)
    est2=est.fit()
    #print(est2.summary())

    pred=est2.predict(X_test)    
    train_pred=est2.predict(X_train)
    print("month " + str(i) + ", r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(train_pred,y_train)))

X.index=hist_rev.index

ann_temps=temps2.groupby(['Year']).agg({'mean'})
ann_temps.reset_index(inplace=True)
ann_temps.columns=['year','month','sal_avg','sal_max','sal_min','boi_avg','boi_max','boi_min','sea_avg','sea_max','sea_min','por_avg','por_max','por_min']

X=ann_temps.loc[:,['boi_max','sea_max','por_min']]
y=hist_rev.reset_index()
X_train,X_test,y_train,y_test=tts(X.reset_index(),y.iloc[:,1],test_size=.2)

#reg=LinearRegression().fit(X_train,y_train)
#reg.score(X,hist_rev)
#reg.coef_
#pred=reg.predict(X_test)

est=sm.OLS(y_train,X_train)
est2=est.fit()
est2.summary()
pred=est2.predict(X_test)
plt.scatter(pred,y_test)
plt.scatter(est2.predict(X_train),y_train)


boise_T=temps[temps['NAME']=='boise']
#boise_T=boise_T.resample('M').sum()
#boise_T['Month']=boise_T.index.month
#boise_T['Year']=boise_T.index.year

salem_T=temps[temps['NAME']=='salem']
#salem_T=salem_T.resample('M').sum()
#salem_T['Month']=salem_T.index.month
#salem_T['Year']=salem_T.index.year

seattle_T=temps[temps['NAME']=='seattle']
#seattle_T=seattle_T.resample('M').sum()
#seattle_T['Month']=seattle_T.index.month
#seattle_T['Year']=seattle_T.index.year

spokane_T=temps[temps['NAME']=='spokane']
#spokane_T=spokane_T.resample('M').sum()
#spokane_T['Month']=spokane_T.index.month
#spokane_T['Year']=spokane_T.index.year

portland_T=temps[temps['NAME']=='portland']
#portland_T=portland_T.resample('M').sum()
#portland_T['Month']=portland_T.index.month
#portland_T['Year']=portland_T.index.year

##BASED ON FINDINGS NEED BOISE_T, SEP ORO, BFE, TDA 
boi_T=ann_temps['boi_avg']/365
sep_oro=

t_inputs=pd.concat([boise_T.Avg,portland_T.Avg,salem_T.Avg,spokane_T.Avg,seattle_T.Avg],axis=1)
t_inputs.columns=['boise_T','portland_T','salem_T','spokane_T','seattle_T']
t_inputs=(t_inputs-32)*(5/9)
##convert to C for consistency's sake 
t_inputs['boi_CDD']=0
t_inputs['port_CDD']=0
t_inputs['sal_CDD']=0
t_inputs['spok_CDD']=0
t_inputs['seatt_CDD']=0

t_inputs['boi_HDD']=0
t_inputs['port_HDD']=0
t_inputs['sal_HDD']=0
t_inputs['spok_HDD']=0
t_inputs['seatt_HDD']=0

for i in range(0,len(t_inputs)):
    for c in range(0,5):
        if t_inputs.iloc[i,c]>18:
            t_inputs.iloc[i,c+5]=t_inputs.iloc[i,c]-18
            t_inputs.iloc[i,c+10]=0
        elif t_inputs.iloc[i,c]<18:
            t_inputs.iloc[i,c+5]=0
            t_inputs.iloc[i,c+10]=18-t_inputs.iloc[i,c]
        else:
            t_inputs.iloc[i,c+5]=0
            t_inputs.iloc[i,c+10]=0            

#weighted=t_inputs

#for j in range(0,len(weighted.columns)):
#    weighted.iloc[:,j]=t_inputs.iloc[:,j]*weights.iloc[0,j]

#X=weighted
#X2=np.array(X.iloc[:,1])
t_inputs['year']=t_inputs.index.year
t_inputs_ann=t_inputs.groupby('year').agg({'salem_T':'sum','seattle_T':'sum',
                             'boise_T':'sum','portland_T':'sum',
                             'spokane_T':'sum','sal_CDD':'sum','sal_HDD':'sum',
                             'boi_CDD':'sum','boi_HDD':'sum',
                             'seatt_CDD':'sum','seatt_HDD':'sum',
                             'port_CDD':'sum','port_HDD':'sum',
                             'spok_CDD':'sum','spok_HDD':'sum'})

##alternatively can consider SWE
swe2=pd.DataFrame()
for i in range(1,len(range(3,20))):
    swe=pd.read_excel("../../../Simona_Work/Columbia_BPA_financial_risk/DATA/Correlation/inputs/SNOTEL_Western_US_April_03_18.xlsx",sheet_name=i)
    swe['year']=i
    swe2=swe2.append(swe)

##well that worked :)
swe2=swe2[(swe2['State/River Basin']=='Columbia River Basin')|(swe2['State/River Basin']=='Oregon')|(swe2['State/River Basin']=='Washington')]
swe2=swe2.loc[:,['State/River Basin','Unnamed: 4','Unnamed: 5','year']]
swe2.columns=['loc','pct_med','pct_med_peak','year']

swe12=swe2[swe2['year']>=10]

ann_swe=pd.concat([swe2[swe2['loc']=='Oregon'].reset_index(),swe2[swe2['loc']=='Washington'].reset_index(),swe2[swe2['loc']=='Columbia River Basin'].reset_index()],axis=1)
ann_swe=ann_swe.iloc[:,[2,3,7,8,12,13]]
ann_swe.columns=['OR_pct_med','OR_pct_peak','WA_pct_med','WA_pct_peak','CRB_pct_med','CRB_pct_peak']

comb=pd.concat([inflows03.iloc[:,:13].reset_index(),ann_swe],axis=1)
cors=(comb).corr()
#hist_rev2=hist_rev.iloc[0:6]
plt.rcParams.update({'font.size': 20})

##daily flows, need to make annual 
TDA2=pd.read_csv("../../../Simona_Work/Columbia_BPA_financial_risk/DATA/Correlation/inputs/TDA_unreg_est_2003_18.csv")

#TDA2=TDA2.iloc[10:,:]

#inflows_ann=inflows_ann.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15]].astype(float)

X=pd.concat([swe12[swe12['loc']=='Oregon'].reset_index(),swe12[swe12['loc']=='Washington'].reset_index(),swe12[swe12['loc']=='Columbia River Basin'].reset_index()],axis=1)
X=X.iloc[:,[2,3,7,8,12,13]]
#X.index=hist_rev.index
X.columns=['OR_pct_med','OR_pct_peak','WA_pct_med','WA_pct_peak','CRB_pct_med','CRB_pct_peak']

##add in the temperatures and cdd/hdd
#inflows_ann=inflows_ann.drop('year',axis=1)
X=pd.concat([X,t_inputs_ann.reset_index(),inflows_ann.TDA_in.reset_index()],axis=1)
X=X.loc[:,['TDA_in','port_CDD','boi_HDD']]#'TDA>min']]
#X=np.array(X['WA_pct_med'])#.reshape(1,-1)
hist2=np.array(hist_rev)#.reshape(1,-1)
hist2=hist2*.001

X_train,X_test,y_train,y_test=tts(X,hist2,test_size=.3,random_state=3) ##or 5


#X_train=np.array(X_train).reshape(1,-1)
#y_train=np.array(y_train).reshape(1,-1)
#X=np.array(X).reshape(1,-1)
#hist2=np.array(hist2).reshape(1,-1)
#y_train.index=X_train.index
#y_test.index=X_test.index

X_train=X_train.astype(float)
est=sm.OLS(y_train.astype(float),X_train.astype(float))
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)

from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

print('test r2: '+str(r2(pred,y_test)))
print('training r2 '+str(r2(est2.predict(X_train),y_train)))


pred_X=np.array(est2.predict(X)).astype(float)
plt.scatter(pred,y_test,label="test")
plt.scatter(est2.predict(X_train),y_train,label="train")
z = np.polyfit(pred_X, np.array(hist2).astype(float), 1)
p = np.poly1d(z)
plt.plot(pred_X,p(pred_X),"r--")
plt.legend()
plt.title("Using TDA, Seattle T, and Boise HDD on Historical Revenues")
plt.xlabel("Predicted Revenue ($M)")
plt.ylabel("Historical Revenue ($M)")


################################################
##net revenue

##add in the temperatures and cdd/hdd
X=pd.concat([t_inputs_ann.reset_index(),inflows_ann.reset_index()],axis=1)
X=X.loc[:,['TDA_in','seattle_T','boi_HDD','boise_T']]#'TDA>min']]
#X=np.array(X['WA_pct_med'])#.reshape(1,-1)
hist2=np.array(hist_net_rev)#.reshape(1,-1)
hist2=hist2

X_train,X_test,y_train,y_test=tts(X,hist2,test_size=.2,random_state=5) ##or 5


#X_train=np.array(X_train).reshape(1,-1)
#y_train=np.array(y_train).reshape(1,-1)
#X=np.array(X).reshape(1,-1)
#hist2=np.array(hist2).reshape(1,-1)
#y_train.index=X_train.index
#y_test.index=X_test.index

X_train=X_train.astype(float)
est=sm.OLS(y_train.astype(float),X_train.astype(float))
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)

from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

print('test r2: '+str(r2(pred,y_test)))
print('training r2 '+str(r2(est2.predict(X_train),y_train)))


pred_X=np.array(est2.predict(X)).astype(float)
plt.scatter(pred,y_test,label="test")
plt.scatter(est2.predict(X_train),y_train,label="train")
z = np.polyfit(pred_X, np.array(hist2).astype(float), 1)
p = np.poly1d(z)
plt.plot(pred_X,p(pred_X),"r--")
plt.legend()
plt.title("Using TDA, Seattle T, and Boise HDD on Historical Revenues")
plt.xlabel("Predicted Revenue ($M)")
plt.ylabel("Historical Revenue ($M)")


#################################################
##generation alone?
#################################################

X=pd.concat([swe12[swe12['loc']=='Oregon'].reset_index(),swe12[swe12['loc']=='Washington'].reset_index(),swe12[swe12['loc']=='Columbia River Basin'].reset_index()],axis=1)
X=X.iloc[:,[2,3,7,8,12,13]]
#X.index=hist_rev.index
X.columns=['OR_pct_med','OR_pct_peak','WA_pct_med','WA_pct_peak','CRB_pct_med','CRB_pct_peak']

##add in the temperatures and cdd/hdd
X=pd.concat([X,t_inputs_ann.reset_index(),inflows_ann.reset_index()],axis=1)
hist_gen.columns=['date','gen']
s=pd.concat([X,hist_gen],axis=1)
s.corr()['BON_in']
#X=np.array(X['WA_pct_med'])#.reshape(1,-1)
hist2=np.array(hist_gen['gen'])#en.reshape(1,-1)
X=X.loc[:,['BON_in']]#'TDA>min']]
#hist2=hist2*.0001

X_train,X_test,y_train,y_test=tts(X,hist2,test_size=.2,random_state=1)

#X_train=np.array(X_train).reshape(1,-1)
#y_train=np.array(y_train).reshape(1,-1)
#X=np.array(X).reshape(1,-1)
#hist2=np.array(hist2).reshape(1,-1)

X_train=X_train.astype(float)
est=sm.OLS(y_train.astype(float),X_train.astype(float))
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)

from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

print('test r2: '+str(r2(pred,y_test)))
print('training r2 '+str(r2(est2.predict(X_train),y_train)))

pred_X=np.array(est2.predict(X)).astype(float)
plt.scatter(pred,y_test,label="test")
plt.scatter(est2.predict(X_train),y_train,label="train")
z = np.polyfit(pred_X, np.array(hist2).astype(float), 1)
p = np.poly1d(z)
plt.plot(pred_X,p(pred_X),"r--")
plt.legend()
plt.title("Using Historical BON Inflows to Predict Historical Gen")
plt.xlabel("Predicted Generation")
plt.ylabel("Historical Generation")


snotel=pd.read_csv("Hist_data/snotel2.csv")
snotel['date2']=pd.to_datetime(snotel['Date'],infer_datetime_format=True)
snotel['year']=snotel.date2.dt.year

ann_sno=snotel.groupby('year').agg('mean')
ann_sno.columns=['hemlock','myrtle','shanghai']

BFE=pd.read_csv("Hist_data/BFE6L_daily.csv")
ORO=pd.read_csv("Hist_data/ORO6H_daily.csv")
TDA=pd.read_csv("Hist_data/TDA6ARF_daily.csv")

BFE['date']=pd.to_datetime(BFE.iloc[:,0])
ORO['date']=pd.to_datetime(ORO.iloc[:,0])
TDA['date']=pd.to_datetime(TDA.iloc[:,0])

BFE['year']=BFE.date.dt.year
ORO['year']=ORO.date.dt.year
TDA['year']=TDA.date.dt.year

ann_BFE=BFE.groupby('year').agg('mean')
ann_ORO=ORO.groupby('year').agg('mean')
ann_TDA=TDA.groupby('year').agg('mean')

ann_comb=pd.merge(ann_BFE,ann_ORO,on='year')
ann_comb=pd.merge(ann_comb,ann_sno,on='year')
ann_comb=pd.merge(ann_comb,ann_TDA,on='year')

ann_comb.columns=['bonners_ferry','orofino','hemlock','myrtle','shanghai','tda']
c=ann_comb.corr()

plt.scatter(ann_comb.tda,ann_comb.orofino)
plt.scatter(ann_comb.tda,ann_comb.bonners_ferry)
plt.scatter(ann_comb.orofino,ann_comb.bonners_ferry)

plt.scatter(ann_comb.tda,ann_comb.hemlock)
plt.scatter(ann_comb.tda,ann_comb.myrtle)
plt.scatter(ann_comb.tda,ann_comb.shanghai)

plt.scatter(ann_comb.bonners_ferry,ann_comb.hemlock,color='lightgreen') ##noticeable, although hemlock = orofino 
plt.scatter(ann_comb.bonners_ferry,ann_comb.myrtle,color='violet') #this should be with bonners
plt.scatter(ann_comb.bonners_ferry,ann_comb.shanghai,color='black')

plt.scatter(ann_comb.orofino,ann_comb.hemlock,color='lightgreen')
plt.scatter(ann_comb.orofino,ann_comb.myrtle,color='violet')
plt.scatter(ann_comb.orofino,ann_comb.shanghai,color='black')

gray=pd.read_excel("E:/Research/PhD/BPA/Hist_data/hist_snow.xlsx",sheet_name="gray_creek")
gray['year']=gray[' Date of Survey'].dt.year
gray=gray.groupby('year').agg({' Water Equiv. mm':'mean'})
gray=gray.reset_index()
gray.columns=['year','gray']

ann_comb=ann_comb.reset_index()
ann_comb.columns=['bonners_ferry','orofino','hemlock','myrtle','shanghai','tda']

koch=pd.read_excel("E:/Research/PhD/BPA/Hist_data/hist_snow.xlsx",sheet_name="koch_creek")
koch['year']=koch[' Date of Survey'].dt.year
koch=koch.groupby('year').agg({' Water Equiv. mm':'mean'})
koch=koch.reset_index()
koch.columns=['year','koch']

nelson=pd.read_excel("E:/Research/PhD/BPA/Hist_data/hist_snow.xlsx",sheet_name="nelson")
nelson['year']=nelson[' Date of Survey'].dt.year
nelson=nelson.groupby('year').agg({' Water Equiv. mm':'mean'})
nelson=nelson.reset_index()
nelson.columns=['year','nelson']

all=gray.merge(ann_comb)
all=all.merge(koch)
all=all.merge(nelson)
corr=all.corr()



BPA_streamflow=pd.read_excel('C:/Users/rcuppari/Desktop/CAPOW_PY36-master/Stochastic_engine/Synthetic_streamflows/BPA_hist_streamflow.xlsx',sheet_name='Inflows',header=0)
ann_st=BPA_streamflow.groupby('year').agg('mean')
mon_st=BPA_streamflow.groupby(['year','month']).agg('mean')

names=pd.read_excel("E:/Research/PhD/CAPOW_SD_New/BPA_name.xlsx")

ann_st=ann_st.loc[:,'1M':]
ann_st.columns=names.iloc[0,:]

mon_st=mon_st.loc[:,'1M':]
mon_st.columns=names.iloc[0,:]

st_corr=ann_st.corr()['Dalls ARF']

low_TDA=ann_st[ann_st['Dalls ARF']<np.quantile(ann_st['Dalls ARF'],0.05)]
low_TDA_mon=mon_st[mon_st['Dalls ARF']<np.quantile(mon_st['Dalls ARF'],0.05)]

mon_corr=mon_st.corr()['Dalls ARF']
low_mon_corr=low_TDA_mon.corr()['Dalls ARF']


hist_gen2=pd.read_csv("Hist_data/hist_ann_hydro.csv")
hist_gen2['date']=pd.to_datetime(hist_gen2.iloc[:,0])
hist_gen2['year']=hist_gen2['date'].dt.year
hist_gen2.index=hist_gen2.year

gen_st=hist_gen2.join(ann_st,how='outer')


key_hist=pd.read_csv('C:/Users/rcuppari/Desktop/CAPOW_PY36-master/Updated streamflow data/ann_hist.csv')
hist_rev2=hist_rev
hist_rev2['year']=[2012,2013,2014,2015,2016,2017,2018]
comb=key_hist.join(hist_rev,on='year')
comb=comb.dropna()
comb=comb.join(t_inputs_ann[['boise_T','boi_HDD','boi_CDD']])

X=sm.add_constant(comb[['boi_HDD','boise_T']]) ##'TDA6ARF_daily','BFE6L_daily'
X_train,X_test,y_train,y_test=tts(X,comb.iloc[:,5],test_size=.2,random_state=1)

est=sm.OLS(y_train.astype(float),X_train.astype(float))
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)

print('test r2: '+str(r2(pred,y_test)))
print('training r2 '+str(r2(est2.predict(X_train),y_train)))

pred_X=np.array(est2.predict(X)).astype(float)
plt.scatter(pred,y_test,label="test")
plt.scatter(est2.predict(X_train),y_train,label="train")
plt.legend()
plt.title("Using Historical BON Inflows to Predict Historical Gen")
plt.xlabel("Predicted Generation")
plt.ylabel("Historical Generation")

comb.iloc[:,4]=(comb.iloc[:,4]).astype(float)