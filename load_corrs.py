# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:40:33 2020

@author: rcuppari
"""

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta  
import numpy as np
import numpy.matlib as matlib
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
import scipy.stats
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

##import stochastic generator results -- what correlates most with BPA?
stoch_weath2=pd.read_csv("E:/Research/PhD/BPA/daily_DD_clean.csv")
ann_weath=stoch_weath2.groupby('year').agg({'SALEM_T':'sum','SALEM_W':'sum',
                             'EUGENE_T':'sum','EUGENE_W':'sum',
                             'SEATTLE_T':'sum','SEATTLE_W':'sum',
                             'BOISE_T':'sum','BOISE_W':'sum',
                             'PORTLAND_T':'sum','PORTLAND_W':'sum',
                             'SPOKANE_T':'sum','SPOKANE_W':'sum',
                             'eug_CDD':'sum','eug_HDD':'sum',
                             'sal_CDD':'sum','sal_HDD':'sum',
                             'boi_CDD':'sum','boi_HDD':'sum',
                             'seat_CDD':'sum','seat_HDD':'sum',
                             'port_CDD':'sum','port_HDD':'sum',
                             'spok_CDD':'sum','spok_HDD':'sum'})
ann_weath['year']=range(0,1188)


#import BPA demand (simply weighted as a fraction of total demand, from Simona)
df_synth_load=pd.read_csv('E:/Research/PhD/BPA/Sim_hourly_load.csv', usecols=[1])
BPAT_load=pd.DataFrame(np.reshape(df_synth_load.values, (24*365,1200), order='F'))
base = dt(2001, 1, 1)
arr = np.array([base + timedelta(hours=i) for i in range(24*365)])
BPAT_load.index=arr
BPAT_load=BPAT_load.resample('D').mean()
BPAT_load.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
BPAT_load=pd.DataFrame(np.reshape(BPAT_load.values, (365*1188), order='F'))

years=pd.DataFrame(np.arange(0,1188))
df=pd.DataFrame({'year':years.values.repeat(365)})

BPAT_load['year']=df.year
BPAT_load.columns=['load','year']

ann_load=BPAT_load.groupby('year').agg({'load':['sum','mean','std']})
ann_load.columns=['sum_load','avg_load','std_load']

##let's start with an easy regression
comb=pd.concat([ann_weath,ann_load],axis=1)

X=ann_weath.loc[:,['boi_CDD','port_HDD']]#.loc[:,['boi_CDD','port_HDD']]
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,ann_load['avg_load'],test_size=.2,random_state=2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
print(r2(pred,y_test))
print(r2(est2.predict(X_train),y_train))

plt.scatter(pred,y_test)
plt.scatter(est2.predict(X_train),y_train,label="training set")
z = np.polyfit(pred, y_test, 1)
p = np.poly1d(z)
plt.plot(pred,p(pred),"r--")
plt.ylabel("Calculated Average Load")
plt.xlabel("Predicted Average Load")
plt.title("Regression with Boise CDD and Portland HDD")

resids=pd.DataFrame(est2.resid)
print('mean residual '+str(est2.resid.mean()))
print('stdev residuals '+str(est2.resid.std())) 






##subtract min?
ann_load['min']=(ann_load['avg_load']-ann_load['avg_load'].min())#())/ann_load['avg_load'].std()

ann_weath['boi_CDD_dev']=(ann_weath['boi_CDD']-ann_weath['boi_CDD'].mean())/ann_weath['boi_CDD'].std()
ann_weath['port_HDD_dev']=(ann_weath['port_HDD']-ann_weath['port_HDD'].mean())/ann_weath['port_HDD'].std()
X=ann_weath.loc[:,['boi_CDD_dev','port_HDD_dev']]#.loc[:,['boi_CDD','port_HDD']]
#X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,ann_load['min'],test_size=.2,random_state=2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
print(r2(pred,y_test))
print(r2(est2.predict(X_train),y_train))

plt.scatter(pred,y_test)
plt.scatter(est2.predict(X_train),y_train,label="training set")
z = np.polyfit(pred, y_test, 1)
p = np.poly1d(z)
plt.plot(pred,p(pred),"r--")
plt.ylabel("Calculated Average Load")
plt.xlabel("Predicted Average Load")
plt.title("Regression with Boise CDD and Portland HDD")

resids=pd.DataFrame(est2.resid)
print('mean residual '+str(est2.resid.mean()))
print('stdev residuals '+str(est2.resid.std())) 

ret_pred=(pred+ann_load.avg_load.min())#*ann_load['avg_load'].std())+ann_load.avg_load.mean()
ret_pred.sort_index(inplace=True)
ind=pred.index

test=ann_load[ann_load.index.isin(ind)]

print(r2(ret_pred,test['avg_load']))
plt.scatter(ret_pred,test['avg_load'])



















