# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:21:47 2020

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

##want to find correlations with low BPA demand/low BPA hydropower specifically 
##start with historical matches


###############################################################################
##given at the synthetic level also have Simona's model, can also start to see 
##what correlates with usage of Treasury funds (which I guess is just below a 
##certain revenue threshold, so should be the same as for revenue)
##we're going to try everything at the annual timestep for starters
###############################################################################
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

jan=[1]*31
feb=[2]*28
mar=[3]*31
apr=[4]*30
may=[5]*31
jun=[6]*30
jul=[7]*31
aug=[8]*31
sep=[9]*30
oc=[10]*31
nov=[11]*30
dec=[12]*31
mons=pd.DataFrame(jan+feb+mar+apr+may+jun+jul+aug+sep+oc+nov+dec)
month=pd.concat([mons]*len(ann_weath))
month.columns=['month']


years=pd.DataFrame(np.arange(0,1188))
df=pd.DataFrame({'year':years.values.repeat(365)})

##import BPA hydropower 
# Hourly hydro generation from FCRPS stochastic simulation
df_hydro=pd.read_csv('E:/Research/PhD/BPA/new_BPA_hydro_daily.csv')
#                     BPA_owned_dams.csv', header=None)
#BPA_hydro=pd.DataFrame(data=df_hydro.sum(axis=1)/24, columns=['hydro'])
#Remove CAISO bad_years
BPA_hydro=pd.DataFrame(np.reshape(df_hydro.gen.values, (365,1200), order='F'))
BPA_hydro.drop([82, 150, 374, 377, 540, 616, 928, 940, 974, 980, 1129, 1191],axis=1, inplace=True)
BPA_hydro=pd.DataFrame(np.reshape(BPA_hydro.values, (365*1188), order='F'))
BPA_hydro['year']=df['year']
BPA_hydro.columns=['hydro','year']

mon_hydro=pd.concat([BPA_hydro.reset_index(),month.reset_index()],axis=1)
mon_hydro=mon_hydro.loc[:,['hydro','year','month']]
mon_hydro=mon_hydro.groupby(['year','month']).agg({'mean'})
mon_hydro.reset_index(inplace=True)
mon_hydro.columns=['year','month','hydro']

ann_hydro=BPA_hydro.groupby('year').agg('mean')


#ann_weath.columns=['year','salem_T','salem_W','eugene_T','eugene_W','seattle_T',
#                   'seattle_W','boise_T','boise_W','portland_T','portland_W',
#                   'spokane_T','spokane_W','eug_CDD','eug_HDD','sal_CDD',
#                   'sal_HDD','boi_CDD','boi_HDD','seat_CDD','seat_HDD',
#                   'port_CDD','port_HDD','spoke_CDD','spok_HDD']

#ann_weath=ann_weath.reset_index()

X_train,X_test,y_train,y_test=tts(ann_weath[['SALEM_T','SALEM_W','boi_CDD','boi_HDD']],ann_hydro['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
est2.summary()
pred=est2.predict(X_test)

plt.scatter(pred,y_test)
plt.scatter(est2.predict(X_train),y_train)

#want to add months to get a monthly breakdown

stoch_weath2=pd.concat([stoch_weath2.reset_index(),month.reset_index()],axis=1)

mon_weath=stoch_weath2.groupby(['year','month']).agg({'SALEM_T':'sum','SALEM_W':'sum',
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


mon_weath.reset_index(inplace=True)
mon_weath.isnull().sum()

########################################
##streamflows?
TDA=pd.read_csv("../../Yufei_data/Synthetic_streamflows/synthetic_streamflows_TDA.csv")
TDA.columns=['day','flow']

years=pd.DataFrame(np.arange(0,1218))
df=pd.DataFrame({'year':years.values.repeat(365)})
TDA['year']=df['year']

##run 6x200 so need to drop the first year and last two every 200 years
drop=[0,201,202,203,404,405,406,607,608,609,810,811,812,1013,1014,1015,1216,1217]
TDA=TDA[~TDA.year.isin(drop)]

years=pd.DataFrame(np.arange(0,1200))
df=pd.DataFrame({'year':years.values.repeat(365)})

TDA2=TDA.loc[:,['day','flow']]
TDA2=TDA2.reset_index()
TDA2['year']=df['year']
drop2=[82,150,374,377,540,616,928,940,974,980,1129,1191]
TDA2=TDA2[~TDA2.year.isin(drop2)]

years=pd.DataFrame(np.arange(0,1188))
df=pd.DataFrame({'year':years.values.repeat(365)})

TDA2=TDA2.loc[:,['flow']]
TDA2=TDA2.reset_index()
TDA2['year']=df['year']
TDA2=pd.concat([TDA2.reset_index(),month.reset_index()],axis=1)
mon_TDA=TDA2.groupby(['year','month']).agg('mean')
mon_TDA.reset_index(inplace=True)
mon_TDA=mon_TDA.loc[:,['year','month','flow']]

ann_TDA=TDA2.groupby('year').agg('mean')


##LOOKING AT THE LOWER END OF THINGS
plt.scatter(ann_TDA.flow,ann_hydro['hydro'])
comb=pd.concat([ann_TDA,ann_hydro],axis=1)
comb=comb[comb['flow']<100000]
plt.scatter(comb['flow'],comb['hydro'])
plt.xlabel("TDA Flow",fontsize=16)
plt.ylabel("Hydropower Generation",fontsize=16)
plt.title("TDA Flows v BPA Generation",fontsize=16)
plt.axvline(x=100000,color="black")

remain=comb.index
mon_TDA2=mon_TDA[mon_TDA.year.isin(remain)]

over=pd.concat([ann_TDA,ann_hydro],axis=1)
over=over[over['flow']>100000]
mon_TDA3=mon_TDA[mon_TDA.year.isin(over.index)]

cols=pd.DataFrame(index=comb.index)
cols_full=pd.DataFrame(index=over.index)

for i in range(1,13):
    mon=pd.DataFrame(mon_TDA2[mon_TDA2['month']==i].flow)
    mon.index=cols.index.copy()
    cols=pd.concat([cols,mon.flow],axis=1)    

    mon_f=pd.DataFrame(mon_TDA3[mon_TDA3['month']==i].flow)
    mon_f.index=cols_full.index.copy()
    cols_full=pd.concat([cols_full,mon_f.flow],axis=1)    

#    X_train,X_test,y_train,y_test=tts(mon.flow.reset_index(),comb.hydro.reset_index(),test_size=.2)

 #   est=sm.OLS(y_train.hydro,X_train.flow)
#    est2=est.fit()
#    pred=est2.predict(X_test.flow)
#    print("month: "+str(i)+" r2: "+str(r2(pred,y_test.hydro)))
#    pred_all=est2.predict(mon.flow)
#    plt.scatter(pred_all,comb.hydro,label=str(i))
#    plt.legend()
cols.columns=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
cols_full.columns=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

X=sm.add_constant(cols)
X_train,X_test,y_train,y_test=tts(X,comb['hydro'],test_size=.2,random_state=1)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
plt.scatter(est2.predict(X),comb.hydro)
r2(est2.predict(X),comb.hydro)

X=sm.add_constant(cols_full)
X_train,X_test,y_train,y_test=tts(X,over['hydro'],test_size=.2,random_state=1)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
plt.scatter(est2.predict(X),over.hydro)
r2(est2.predict(X),over.hydro)

cols_diff=cols.copy()
cols_diff=cols-cols.mean(axis=0)
cols_diff['lowest']=cols_diff.idxmin(axis=1)
print("most freq diff from mean: "+str(cols_diff.lowest.value_counts()))

cols_diff2=cols_full.copy()
cols_diff2=cols_full-cols_full.mean(axis=0)
cols_diff2['lowest']=cols_diff2.idxmin(axis=1)
print("most freq diff from mean: "+str(cols_diff2.lowest.value_counts()))

x=pd.concat([ann_TDA['flow'],ann_hydro['hydro']],axis=1)
x['year']=df['year']
x.corr()

X=ann_TDA['flow']
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,ann_hydro['hydro'],test_size=.2)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)


##hydro?
fcrps=pd.read_csv("../../Yufei_data/Synthetic_streamflows/synthetic_streamflows_FCRPS.csv")
fcrps=fcrps.iloc[:,1:]
names=pd.read_excel("../../Yufei_data/Synthetic_streamflows/BPA_name.xlsx")
fcrps.columns=np.array(names.iloc[0,:])

years=pd.DataFrame(np.arange(0,1218))
df=pd.DataFrame({'year':years.values.repeat(365)})
fcrps['year']=df['year']

##run 6x200 so need to drop the first year and last two every 200 years
drop=[0,201,202,203,404,405,406,607,608,609,810,811,812,1013,1014,1015,1216,1217]
fcrps=fcrps[~fcrps.year.isin(drop)]

years=pd.DataFrame(np.arange(0,1200))
df=pd.DataFrame({'year':years.values.repeat(365)})

fcrps=fcrps.drop(['year'],axis=1)

fcrps2=fcrps.reset_index()
fcrps2['year']=df['year']

drop2=[82,150,374,377,540,616,928,940,974,980,1129,1191]
fcrps2=fcrps2[~fcrps2.year.isin(drop2)]

years=pd.DataFrame(np.arange(0,1188))
df=pd.DataFrame({'year':years.values.repeat(365)})

fcrps2=fcrps2.drop(['year','index'],axis=1)
#fcrps2=fcrps2.iloc[:,2:]
fcrps2.reset_index(inplace=True)

fcrps2['year']=df['year']
fcrps2.reset_index(inplace=True)


ann_fcrps=fcrps2.groupby('year').agg('mean')

ann_fcrps.reset_index(inplace=True)
#ann_fcrps=ann_fcrps.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]]

x=pd.concat([ann_hydro['hydro'],ann_fcrps,ann_TDA],axis=1)
corrs=x.corr()['hydro']
corrs_TDA=x.corr()['Dalls ARF']

plt.scatter(x['Bonners Ferry L'],x['hydro'])
plt.scatter(ann_fcrps['Dworshak M'],ann_hydro['hydro'])
plt.scatter(ann_hydro['hydro'],ann_fcrps['Dalls ARF'])

##below certain hydro threshold how can we tell? 
two=pd.concat([ann_hydro['hydro'],ann_fcrps.iloc[:,3:],ann_weath],axis=1)
two=two[(two['Dalls ARF']<100000)] # & (two['Dalls ARF']>200000)
#plt.scatter(two['Dalls ARF'],two['hydro'],alpha=.5)

two_ann=two.groupby('year').agg('mean')
#plt.scatter(two_ann['BOISE_T'],two_ann['hydro'])

X=two_ann[['Bonners Ferry L','Orofino H']]
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,two_ann['hydro'],test_size=.2,random_state=6)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())

pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,label='Testing with Bonners Ferry and Orofino',color="purple")
plt.scatter(pred_tr,y_train,label='Training with Bonners Ferry and Orofino',color="darkorange")

X=two_ann[['Dalls ARF']]
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,two_ann['hydro'],test_size=.2,random_state=6)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())

pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(est2.predict(X),two_ann.hydro,label='With TDA',color="blue",alpha=.5)
plt.xlabel("Predicted Generation",fontsize=16)
plt.ylabel("Simulated Generation",fontsize=16)
plt.legend()

flo=two[['Dalls ARF','Orofino H','Bonners Ferry L','hydro']]
np.corrcoef(flo['Dalls ARF'],flo['Bonners Ferry L'])
np.corrcoef(flo['Dalls ARF'],flo['Orofino H'])
np.corrcoef(flo['Bonners Ferry L'],flo['Orofino H'])

np.corrcoef(flo['Dalls ARF'],flo['hydro'])
np.corrcoef(flo['hydro'],flo['Orofino H'])
np.corrcoef(flo['Bonners Ferry L'],flo['hydro'])
#################################################
nof=pd.concat([ann_hydro['hydro'],ann_fcrps],axis=1)
cor2=nof.corr()['hydro']


plt.scatter(ann_fcrps['Dalls ARF'],ann_hydro['hydro'])
plt.title("Annual Generation versus TDA",fontsize=18)
#plt.yticks([1500000,2000000,2500000,3000000,3500000,4000000,4500000],
#           ["1500","2000","2500","3000","3500","4000","4500"])
plt.xticks([0,50000,100000,150000,200000,250000,300000],
           [0,50,100,150,200,250,300])
plt.ylabel("Total Generation (million MWh)",fontsize=18)
plt.xlabel("Average Daily Flow (thousand m3/s)",fontsize=18)

    
##water year from october-september?
mon_fcrps=pd.concat([fcrps2,month.reset_index()],axis=1)
mon_fcrps=mon_fcrps.groupby(['year','month']).agg('mean')
mon_fcrps.reset_index(inplace=True)

wat_year=mon_fcrps.iloc[9:14253,:]
wat_year=wat_year.drop(columns=['level_0','index'])

wat_year.reset_index(inplace=True)

years=pd.DataFrame(np.arange(0,1187))
df=pd.DataFrame({'year':years.values.repeat(12)})

wat_year.loc[:,'year']=df.loc[:,'year'].copy()

ann_wat_year=wat_year.groupby('year').agg('mean')
hydro_wat_year=ann_hydro.iloc[:-1,:]

from sklearn import preprocessing as pp
scaler=pp.StandardScaler(copy=True,with_mean=False,with_std=True)
comb=pd.concat([ann_fcrps,ann_hydro['hydro']],axis=1)
x_scaled=scaler.fit_transform(comb.values)
norm=pd.DataFrame(x_scaled)
norm.columns=comb.columns

plt.scatter(norm['Dalls ARF'],norm['hydro'])

X=sm.add_constant(norm['Dalls ARF'])
X_train,X_test,y_train,y_test=tts(X,norm['hydro'],test_size=.2,random_state=2)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())

pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))


X=sm.add_constant(comb)
X_train,X_test,y_train,y_test=tts(X['Dalls ARF'],comb['hydro'],test_size=.2,random_state=2)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())

pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))



##subtract min?
ann_hydro['min']=(ann_hydro['hydro']-ann_hydro['hydro'].min())#())/ann_load['avg_load'].std()


X=ann_fcrps['Dalls ARF']-ann_fcrps['Dalls ARF'].min()
#X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,ann_hydro['min'],test_size=.2,random_state=2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
print(r2(pred,y_test))
print(r2(est2.predict(X_train),y_train))

resids=pd.DataFrame(est2.resid)
print('mean residual '+str(est2.resid.mean()))
print('stdev residuals '+str(est2.resid.std())) 

ret_pred=(pred+ann_hydro.hydro.min())#*ann_load['avg_load'].std())+ann_load.avg_load.mean()
ret_pred.sort_index(inplace=True)
ind=pred.index

test=ann_hydro[ann_hydro.index.isin(ind)]

print(r2(ret_pred,test['hydro']))
plt.scatter(ret_pred,test['hydro'])


##PICK UP HERE!
# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.0001
lasso_nalpha=20
lasso_iter=5000

# Min and max degree of polynomials features to consider
degree_min = 2
degree_max = 4

# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LassoCV
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline

X=pd.concat([ann_TDA['flow'],ann_weath[['BOISE_T','PORTLAND_T']]],axis=1)
# Test/train split
X_train, X_test, y_train, y_test = tts(X, ann_hydro['hydro'],test_size=.3)

colors = ['teal', 'yellowgreen', 'gold', 'orange']
lw = 2

# Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
for count, degree in enumerate([2,3,4]):
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), 
                          LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,normalize=True,cv=5))
    model.fit(X_train,y_train)
    test_pred = np.array(model.predict(X_test))
    RMSE=np.sqrt(np.sum(np.square(test_pred-y_test)))
    test_score = model.score(X_test,y_test)
    print('RMSE'+str(RMSE)+' for degree ' + str(degree))
    print('score'+str(test_score)+' for degree ' + str(degree))
    y_plot = model.predict(X)
    plt.plot(ann_hydro.hydro, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree,alpha=.5)

plt.legend(loc='lower left')

plt.show()


poly=PolynomialFeatures(3,include_bias=False)
X_poly=poly.fit_transform(X)
X_poly_feature_name = poly.get_feature_names(['Feature'+str(l) for l in range(1,6)])

df_poly = pd.DataFrame(X_poly, columns=X_poly_feature_name)
df_poly['y']=ann_hydro['hydro']

X_train=df_poly.drop('y',axis=1)
y_train=df_poly['y']

model1 = LassoCV(cv=10,verbose=0,normalize=True,eps=0.001,n_alphas=100, tol=0.0001,max_iter=5000)
model1.fit(X_train,y_train)
y_pred1 = np.array(model1.predict(X_train))
RMSE_1=np.sqrt(np.sum(np.square(y_pred1-y_train)))
r2_1=model1.score(X_test,y_test)
print("Root-mean-square error of Metamodel:",RMSE_1)
print("R2",r2_1)
coeff1 = pd.DataFrame(model1.coef_,index=df_poly.drop('y',axis=1).columns, columns=['Coefficients Metamodel'])
coeff1[coeff1['Coefficients Metamodel']!=0]

model1.score(X_train,y_train)
model1.alpha_


plt.figure(figsize=(12,8))
plt.xlabel("Predicted value with Metamodel",fontsize=20)
plt.ylabel("Actual y-values",fontsize=20)
plt.grid(1)
plt.scatter(y_pred1,y_train,edgecolors=(0,0,0),lw=2,s=80)
plt.plot(y_pred1,y_pred1, 'k--', lw=2)

# Display results
m_log_alphas = -np.log10(model1.alphas_)

plt.figure()
ymin, ymax = 2300, 3800
plt.plot(m_log_alphas, model1.mse_path_, ':')
plt.plot(m_log_alphas, model1.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model1.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.axis('tight')


from scipy.optimize import curve_fit
ydata=ann_hydro.hydro
xdata=ann_TDA['flow']


def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess

popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox')
x=ann_TDA.flow
y=sigmoid(x,*popt)

plt.plot(xdata, ydata, 'o', label='data')
plt.plot(x,y, label='fit')
plt.legend(loc='best')


sns.kdeplot(ann_TDA.flow)
pdf=stats.norm.pdf(ann_TDA.flow)

def sigmoid (x, A, h, slope, C):
    return 1 / (1 + np.exp ((x - h) / slope)) *  A + C

# Fits the function sigmoid with the x and y data
#   Note, we are using the cumulative sum of your beta distribution!
p, _ = curve_fit(sigmoid, xdata,pdf.cumsum())

# Plots the data
plt.scatter(xdata, ydata, label='original')
plt.scatter(xdata, sigmoid(xdata, *p), label='sigmoid fit')
plt.legend()


cor=two.corr()['hydro']
X_train,X_test,y_train,y_test=tts(two.loc[:,['Dalls ARF','boi_HDD','Revelstoke L']],two['hydro'],test_size=.2)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)

thr=two.groupby('year').agg('mean')
cor2=thr.corr()['hydro']
X_train,X_test,y_train,y_test=tts(thr.loc[:,['Dalls ARF',]],thr['hydro'],test_size=.2)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)



##piecewise for hydro 

def piecewise_linear(x, x0, y0, k1, k2,k3):
    return np.piecewise(x, [x < x0, x0<x<160000,x>160000], 
                        [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0, lambda x:k3*x + y0-k3*x0])

x = np.array(ann_TDA.flow).astype(np.float)
y = np.array(ann_hydro.hydro).astype(np.float)


from scipy import interpolate
tck=interpolate.splrep(x,y,k=2,s=0)

from scipy.optimize import curve_fit
popt_piecewise, pcov = curve_fit(piecewise_linear, x, y, p0=[-1000,0,1000,2000,4000])
new_x = np.linspace(x.min(), x.max(), 61)

fig, ax = plt.subplots()
ax.plot(x, y, 'o', ls='')
ax.plot(new_x, piecewise_linear(new_x, *popt_piecewise))

pred=piecewise_linear(x, *popt_piecewise)
r2(pred,y)

from scipy import optimize
def segments_fit(X, Y, count):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)


px, py = segments_fit(x, y, 3)

plt.plot(x, y, ".")
plt.plot(px, py, "-or");
plt.xlabel("TDA flow",fontsize=16)
plt.ylabel("BPA Hydropower Generation",fontsize=16)
plt.title("TDA versus Hydropower",fontsize=16)


##detrend? 
ann_fcrps_det=ann_fcrps.loc[:,['Dalls ARF','Dworshak M','Orofino H','Mica M','Bonneville L']]
review=ann_fcrps.loc[:,['Dalls ARF','Dworshak M','Orofino H','Mica M','Bonneville L']]
for i in range(0,len(review)): 
    for c in range(0,len(review.columns)):
        m=review.iloc[:,c].mean()
        std=review.iloc[:,c].std()
        ann_fcrps_det.iloc[i,c]=(review.iloc[i,c]-m)#/std

ann_hydro_det=ann_hydro['avg_hydro']

for i in range(0,len(ann_hydro)):
    ann_hydro_det[i]=(ann_hydro.iloc[i,1]-ann_hydro['avg_hydro'].mean())#/ann_hydro['avg_hydro'].std()

x=pd.concat([ann_fcrps_det,ann_hydro_det],axis=1)
x.corr()

X_train,X_test,y_train,y_test=tts(ann_fcrps_det.iloc[:,[2,3]],ann_hydro_det,test_size=.2,random_state=10)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)

pred_all=est2.predict(ann_fcrps_det.iloc[:,[2,3]])
##retrend the hydro
ret=(pred_all*ann_hydro['avg_hydro'].std())+ann_hydro.avg_hydro.mean()
plt.scatter(ret,ann_hydro['avg_hydro'])

#X=sm.add_constant(mon_fcrps) #mon_fcrps.loc[:,['Dalls ARF','Dworshak M']]
#X.loc[:,['month','Dalls ARF','Dworshak M']]
X_train,X_test,y_train,y_test=tts(mon_fcrps.loc[:,['Dalls ARF','Dworshak M']],mon_hydro['hydro'],test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)

X_train,X_test,y_train,y_test=tts(ann_fcrps.loc[:,['Dalls ARF','Dworshak M']],ann_hydro['hydro'],test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)

##or even flow
for i in range(1,13):
    X=mon_fcrps[mon_fcrps['month']==i]
    X=sm.add_constant(X)
    X.index=X['year']
    X_train,X_test,y_train,y_test=tts(X,ann_hydro['sum_hydro'],test_size=.2)
    est=sm.OLS(y_train,X_train)
    est2=est.fit()
    #print(est2.summary())

    pred=est2.predict(X_test)    
    train_pred=est2.predict(X_train)
    print("month " + str(i) + ", r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(train_pred,y_train)))

X=ann_fcrps.loc[:,['Arrow L','Ice Harbor L','Chief Joseph L','Dalls ARF','Waneta L','Dworshak M']]
#X=sm.add_constant(X)
#X.index=X['year']
X_train,X_test,y_train,y_test=tts(X,ann_hydro['sum_hydro'],test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)    
train_pred=est2.predict(X_train)
print("month " + str(i) + ", r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(train_pred,y_train)))
plt.scatter(pred,y_test)
plt.scatter(train_pred,y_train)

water_year=pd.DataFrame(index=df['year'])

##so first year and last year will be incomplete, so drop them (jan-sep year 0 and oct-dec year 1187)
mon_TDA['min']=0
for i in range(0,len(mon_TDA)):
    mon_TDA.iloc[i,3]=mon_TDA.iloc[i,2]-mon_TDA[mon_TDA['month']==mon_TDA.iloc[i,1]].flow.min()
x=pd.concat([mon_TDA,mon_hydro,mon_weath],axis=1)

x.corr()['hydro']
wat_yr_data=pd.concat([mon_TDA.loc[:,['flow','min']],mon_fcrps,mon_weath.iloc[:,2:],mon_hydro['hydro']],axis=1)
wat_yr_data=wat_yr_data.iloc[9:14253]
##then sum every 12 rows (12 sequential months, oct-sep for each year)
wat_yr=wat_yr_data.groupby(np.arange(len(wat_yr_data))//12).sum()
##reassign year, but note that the first year is omitted 
wat_yr['year']=df['year']+1
wat_yr.corr()['hydro']

##join with other relevant data, remembering to omit year 0 
##let's see what happens
X=wat_yr.loc[:,['Arrow L','BOISE_T','Anatone L','SEATTLE_T','Brilliant L']]#.iloc[:,0:81]# #'Dalls ARF','Duncan M','Revelstoke L'
#X.index=X.year
X=sm.add_constant(X)
y=wat_yr['hydro']
X_train,X_test,y_train,y_test=tts(X,y,test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)    
train_pred=est2.predict(X_train)
print("month " + str(i) + ", r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(train_pred,y_train)))
plt.scatter(pred,y_test)
plt.scatter(train_pred,y_train)




##by season? 
jm=[1]*59 ##jan-feb = winter
dec=[1]*31 ##dec=winter
spring=[2]*92 ##mar,apr,may = spring
summer=[3]*92 ##jun,jul,aug
fall=[4]*91 ##sep,oct,nov

seas=pd.DataFrame(jm+spring+summer+fall+dec)
seas=pd.concat([seas]*len(ann_weath))
seas.columns=['season']
stoch_weath3=pd.concat([stoch_weath2.reset_index(),seas.reset_index()],axis=1)

seas_weath=stoch_weath3.groupby(['year','season']).agg({'mean'})
seas_data=pd.concat([stoch_weath2,fcrps2,seas['season'],BPA_hydro['hydro']],axis=1)
seas_data=seas_data.iloc[:,3:]
seas_data=seas_data.groupby(['year','season']).agg({'mean'})

#wint=seas_weath[seas_weath.season==1]
#spri=seas_weath[seas_weath.season==2]
#summ=seas_weath[seas_weath.season==3]
#fall=seas_weath[seas_weath.season==4]

wint=seas_data[seas_data.season==1]
spri=seas_data[seas_data.season==2]
summ=seas_data[seas_data.season==3]
fall=seas_data[seas_data.season==4]

#X=wint.iloc[:,:23].reset_index()
#X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(seas_data.iloc[:,:89],ann_hydro['avg_hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)

plt.scatter(pred,y_test)

##still womp

##now check these things with hydro or load
##there's no way HDD/CDD don't link to load at least

X=ann_weath.loc[:,['boi_HDD','boi_CDD']]
y=ann_load
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X.reset_index(),y.reset_index(),test_size=.2,random_state=5)
X_train=X_train.iloc[:,1:]
y_train=y_train.iloc[:,1]
y_test=y_test.iloc[:,1]
X_test=X_test.iloc[:,1:]
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test) 
r2(pred,y_test)

fig, ax = plt.subplots()
ax.scatter(pred,y_test,label='Test Set')
ax.scatter(est2.predict(X_train),y_train,alpha=.3,label='Training Set')
ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)
ax.set_title("Using Boise HDD/CDD to Predict Annual BPA Load")
ax.annotate("r2=0.65",(2200000,2280000))
ax.set_xlabel("Predicted Annual Load (Total MWh)")
ax.set_ylabel("CAPOW Annual Load (Total MWh)")
plt.legend()


#comb1=pd.concat([mon_fcrps.iloc[:,0:20],mon_hydro,mon_rev],axis=1)
#comb2=pd.concat([mon_fcrps.iloc[:,20:40],mon_hydro,mon_rev],axis=1)
#comb3=pd.concat([mon_fcrps.iloc[:,40:],mon_hydro,mon_rev],axis=1)

#sns.heatmap(comb1.corr(),annot=True)
#sns.heatmap(comb2.corr(),annot=True)
#sns.heatmap(comb3.corr(),annot=True)

X=mon_fcrps.loc[:,['Chief Joseph L','Dalls ARF']]#.drop(columns=['year','level_0','index'])
y=mon_rev
#X=mon_fcrps.iloc[:,[4,55,52,51,46]]
#X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,y.reset_index(),test_size=.2,random_state=2)
#X_train=X_train.iloc[:,1:]
y_train=y_train.iloc[:,1]
y_test=y_test.iloc[:,1]
#X_test=X_test.iloc[:,1:]
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test) 

r2(pred,y_test)
r2(est2.predict(X_train),y_train)

plt.scatter(pred,y_test)


TDA=pd.read_csv("../../Yufei_data/Synthetic_streamflows/synthetic_streamflows_TDA.csv")
TDA.columns=['day','flow']

years=pd.DataFrame(np.arange(0,1218))
df=pd.DataFrame({'year':years.values.repeat(365)})
TDA['year']=df['year']

##run 6x200 so need to drop the first year and last two every 200 years
drop=[0,201,202,203,404,405,406,607,608,609,810,811,812,1013,1014,1015,1216,1217]
TDA=TDA[~TDA.year.isin(drop)]

years=pd.DataFrame(np.arange(0,1200))
df=pd.DataFrame({'year':years.values.repeat(365)})

TDA2=TDA.loc[:,['day','flow']]
TDA2=TDA2.reset_index()
TDA2['year']=df['year']
drop2=[82,150,374,377,540,616,928,940,974,980,1129,1191]
TDA2=TDA2[~TDA2.year.isin(drop2)]

years=pd.DataFrame(np.arange(0,1188))
df=pd.DataFrame({'year':years.values.repeat(365)})

TDA2=TDA2.loc[:,['flow']]
TDA2=TDA2.reset_index()
TDA2['year']=df['year']
TDA2=pd.concat([TDA2.reset_index(),month.reset_index()],axis=1)
mon_TDA=TDA2.groupby(['year','month']).agg('sum')
mon_TDA.reset_index(inplace=True)
mon_TDA=mon_TDA.loc[:,['year','month','flow']]


##let's see if any months work out for weather
for i in range(1,13):
    X=mon_weath[mon_weath['month']==i]
    X=X.loc[:,['year','boi_CDD','boi_HDD','eug_HDD','port_HDD','spok_HDD','spok_CDD']]
    X=sm.add_constant(X)
    X.index=X['year']
    X_train,X_test,y_train,y_test=tts(X,ann_rev['rev'],test_size=.2)
    est=sm.OLS(y_train,X_train)
    est2=est.fit()
    print(est2.summary())

    #pred=est2.predict(X_test)
    
    #reg=LinearRegression().fit(X_train,y_train)
    #reg.score(X_train,y_train)
    #pred=reg.predict(X_test)
    
    print("month " + str(i) + ", r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(reg.predict(X_train),y_train)))

##or even flow
for i in range(1,13):
    X=mon_fcrps[mon_fcrps['month']==i]
    X=sm.add_constant(X)
    X.index=X['year']
    X_train,X_test,y_train,y_test=tts(X,ann_rev['rev'],test_size=.2)
    #est=sm.OLS(y_train,X_train)
    #est2=est.fit()
    #print(est2.summary())

    #pred=est2.predict(X_test)
    
    
    
    print("month " + str(i) + ", r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(est2.predict(X_train),y_train)))

X=ann_weath['seattle_T']#.loc[:,['year','boi_CDD','boi_HDD','eug_HDD','port_HDD','spok_HDD','spok_CDD']]
X=pd.concat([X,ann_fcrps.loc[:,['Arrow L','Chief Joseph L']]],axis=1)
#X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,ann_hydro['sum_hydro'],test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
print("monthly r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(est2.predict(X_train),y_train)))

X=mon_fcrps.loc[:,['month','Dalls ARF']]
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,mon_hydro['hydro'],test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
print("monthly, r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(est2.predict(X_train),y_train)))

##HERE
may=mon_fcrps[mon_fcrps['month']==5]
may_d=may.loc[:,['Dalls ARF','year']]
may_d.columns=['may','year']
may_d.reset_index(inplace=True)

X=pd.merge(mon_weath,mon_fcrps.loc[:,['year','month','Dalls ARF','Hungry Horse M','Orofino H']])
X=X.loc[:,['Dalls ARF','month','boi_HDD']]#'PORTLAND_T']] #without CDD 16
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,mon_hydro['hydro'],test_size=.2,random_state=5)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
print("monthly, r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(est2.predict(X_train),y_train)))

train_pred=est2.predict(X_train)

fig, ax = plt.subplots()
ax.scatter(y_test,pred, label="test set")
ax.scatter(y_train,train_pred,alpha=.5,label="training set")
plt.ylim(0)
z = np.polyfit(pred, y_test, 1)
p = np.poly1d(z)
ax.plot(pred,p(pred),"b--")
plt.legend(loc='lower right')
plt.title("Predicting Monthly Generation Using Dalles ARF, Month, and Boise HDDs & CDDs",size=16)
plt.xlabel("Synthetic Generation",size=16)
plt.ylabel("Predicted Generation",size=16)


X2=pd.concat([ann_fcrps.loc[:,['Dalls ARF','Orofino H']],may_d['may'],ann_weath.loc[:,['boi_HDD','boi_CDD']]],axis=1)
#X=X.loc[:,['Dalls ARF','month','boi_HDD']]#'PORTLAND_T']] #without CDD 16
X2=sm.add_constant(X2.loc[:,['Dalls ARF','boi_HDD']])
X_train,X_test,y_train,y_test=tts(X2,ann_hydro['sum_hydro'],test_size=.2,random_state=6)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
print("monthly, r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(est2.predict(X_train),y_train)))

fig, ax = plt.subplots()
ax.scatter(y_test,pred, label="test set")
ax.scatter(y_train,train_pred,alpha=.5,label="training set")
plt.ylim(0)
z = np.polyfit(pred, y_test, 1)
p = np.poly1d(z)
ax.plot(pred,p(pred),"b--")
plt.legend(loc='lower right')
plt.title("Predicting Annualh Generation Using Dalles ARF, Month, and Boise HDDs & CDDs",size=16)
plt.xlabel("Synthetic Generation",size=16)
plt.ylabel("Predicted Generation",size=16)


##THIS WORKS?
##remove outlier 10468
X=pd.merge(mon_weath,mon_fcrps.loc[:,['year','month','Dalls ARF','Hungry Horse M']])
X=X.loc[:,['month','boi_CDD','boi_HDD']]#'PORTLAND_T']] #without CDD 16
#X=X.drop(index=10468)
#y=mon_rev['rev']*.000001#.drop(index=10468)
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,mon_rev['hydro'],test_size=.2,random_state=3)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
print("monthly, r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(est2.predict(X_train),y_train)))

train_pred=est2.predict(X_train)

plt.rc('font', size=18)
fig, ax = plt.subplots()
ax.scatter(pred,y_test, label='test set')
ax.scatter(train_pred,y_train,alpha=.4,label='training set')
z = np.polyfit(pred, y_test, 1)
p = np.poly1d(z)
ax.plot(pred,p(pred),"b--")
ax.annotate('test r2=.48',(140000000,500000000))
ax.set_xlabel("Predicted Revenue ($M)")
ax.set_ylabel("Synthetic Revenue ($M)")
plt.title("Monthly Revenue Prediction using Dalls ARF, Boise CDD, Boise HDD")
plt.legend()





##with portland
X=pd.merge(mon_weath,mon_fcrps.loc[:,['year','month','Dalls ARF','Hungry Horse M']])
X=X.loc[:,['month','PORTLAND_T','boi_CDD','boi_HDD']]#'PORTLAND_T']] #without CDD 16
#X=X.drop(index=10468)
y=mon_rev['rev']*.000001#.drop(index=10468)
X=sm.add_constant(X)
X_train,X_test,y_train,y_test=tts(X,y,test_size=.2,random_state=3)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
print("monthly, r2 with test: "+str(r2(pred,y_test))+" r2 with training: " + str(r2(est2.predict(X_train),y_train)))

train_pred=est2.predict(X_train)



plt.rc('font', size=18)
fig, ax = plt.subplots()
ax.scatter(pred,y_test, label='test set')
ax.scatter(train_pred,y_train,alpha=.5,label='training set')
#ax.scatter(hist_pred,hist2,label="historical data")
z = np.polyfit(pred, y_test, 1)
p = np.poly1d(z)
ax.plot(pred,p(pred),"b--")
ax.annotate('test r2=.48',(140000000,500000000))
ax.set_xlabel("Predicted Revenue")
ax.set_ylabel("Synthetic Revenue")
plt.title("Monthly Revenue Prediction using Dalls ARF, Boise CDD, Boise HDD")
plt.legend() 







##seasonal
first=[3]*(3)
apr=[1]
mayjun=[0]*(2)
jul=[1]
end=[3]*(5)
seas=pd.DataFrame(first+apr+mayjun+jul+end)
season=pd.concat([seas]*len(ann_weath))
season.columns=['season']

mon_fcrps2=pd.concat([mon_fcrps,season['season'].reset_index()],axis=1)
seas_data=pd.concat([mon_fcrps.iloc[:,4:],mon_weath.iloc[:,2:],mon_hydro,season['season'].reset_index()],axis=1)
seas_data=seas_data.loc[:,['year','Dworshak M','Dalls ARF','Hungry Horse M','BOISE_T','SPOKANE_T','BOISE_W','boi_CDD','boi_HDD','Orofino H','season']]

zero=seas_data[seas_data['season']==0]
zero=zero.groupby(['year']).agg('mean')
zero.columns=['z_Dwr','z_Dalls','z_HGH','z_BT','z_BW','z_ST','z_CDD','z_HDD','z_Oro','season']

one=seas_data[seas_data['season']==1]
one=one.groupby(['year']).agg('mean')
one.columns=['o_Dwr','o_Dalls','o_HGH','o_BT','o_BW','o_ST','o_CDD','o_HDD','o_Oro','season']

three=seas_data[seas_data['season']==3]
three=three.groupby(['year']).agg('mean')
three.columns=['t_Dwr','t_Dalls','t_HGH','t_BT','t_BW','t_ST','t_CDD','t_HDD','t_Oro','season']

seas_concat=pd.concat([zero,one,three,ann_fcrps,ann_weath],axis=1)

monthly=pd.concat([mon_fcrps2,mon_weath],axis=1)
monthly=monthly.iloc[:,5:]
#monthly=monthly[~monthly['month'].isin([9,12,4,6])]
#mon_hydro3=mon_hydro2[~mon_hydro2['month'].isin([9,12,4,6])]
#X=sm.add_constant(monthly.loc[:,['boi_HDD','BOISE_T','Dalls ARF','month']]) #'Dalls ARF','boi_HDD','Orofino H','BOISE_T'
X_train,X_test,y_train,y_test=tts(seas_concat.loc[:,['t_BW','t_HDD','seat_HDD','Lime Point L',]],ann_hydro['avg_hydro'],test_size=.2,random_state=9)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

wo_weird=monthly[~monthly['month'].isin([9,12])]
wo_hyd=mon_hydro2[~mon_hydro2['month'].isin([9,12])]

pred_wo=est2.predict(wo_weird.loc[:,['boi_HDD','BOISE_T','Dalls ARF','month']])

plt.scatter(pred_wo,wo_hyd['hydro'],label="without 9,12")
plt.scatter(pred,y_test,alpha=.5,label="test")
plt.scatter(pred_tr,y_train,alpha=.25,label="train")
plt.ylim(0)

##let's use this to aggregate then? 
pred_all=pd.DataFrame(est2.predict(X))
#pred_all['year']=monthly['year']
#pred_all=pred_all.groupby('year').agg('mean')
pred_all['month']=monthly['month']
pred_all['year']=monthly['year']
pred_all['diff']=(pred_all.iloc[:,0]-mon_hydro2['hydro'])/mon_hydro2['hydro']
#np.corrcoef(pred_all.iloc[:,1],ann_hydro['avg_hydro'])
#plt.scatter(pred_all.iloc[:,1],ann_hydro['avg_hydro'])

#############################################
##something's happening up to 2500 and above
#############################################
monthly=pd.concat([monthly,mon_hydro2],axis=1)
twof=monthly[monthly['hydro']<250000]
above=monthly[monthly['hydro']>250000]

X_tf=sm.add_constant(twof.loc[:,['boi_HDD','BOISE_T','Dalls ARF','month']]) #'Dalls ARF','boi_HDD','Orofino H','BOISE_T'
X_train,X_test,y_train,y_test=tts(X_tf,twof['hydro'],test_size=.2)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)


X_ab=sm.add_constant(above.loc[:,['boi_HDD','BOISE_T','Dalls ARF','month']]) #'Dalls ARF','boi_HDD','Orofino H','BOISE_T'
X_train,X_test,y_train,y_test=tts(X_ab,above['hydro'],test_size=.2)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)








X=pd.concat([ann_fcrps,ann_weath],axis=1)
X_train,X_test,y_train,y_test=tts(X.loc[:,['Dalls ARF','Orofino H']],ann_hydro['avg_hydro'],test_size=.2)
#X_train,X_test,y_train,y_test=tts(two.iloc[:,4:],two['hydro'],test_size=.2)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)

fc_cor=mon_fcrps.corr()

anns=pd.concat([ann_weath,ann_fcrps,ann_hydro],axis=1)
anns.corr()['Noxon L']

fcrps2['total']=fcrps2.sum(axis=1)
plt.scatter(fcrps2['total'],BPA_hydro['hydro'])


#ann_fcrps['total']=ann_fcrps.iloc[:,3:57].sum(axis=1)
ann_fcrps['total']=ann_fcrps.iloc[:,[15,31,42,43,45,50]].sum(axis=1)
mon_fcrps['total']=mon_fcrps.iloc[:,[4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,52,53,54,55,56,57,58]].sum(axis=1)
plt.scatter(mon_fcrps['total'],mon_hydro['hydro'])
plt.scatter(mon_fcrps['Dworshak M'],mon_hydro['hydro'])
plt.scatter(mon_fcrps['Dalls ARF'],mon_hydro['hydro'])

##CHECK HERE
X=pd.concat([mon_weath.loc[:,['boi_HDD','SEATTLE_T']],mon_fcrps.loc[:,['Dworshak M','Dalls ARF']]],axis=1)
X_train,X_test,y_train,y_test=tts(X,mon_hydro['hydro'],test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)

predall=est2.predict(ann_fcrps['total'])
##LOOK AT WHICH YEARS SPECIFICALLY ARE OFF IN PREDICTIONS AND FIGURE OUT WHAT DRIVING THAT
comp=pd.concat([predall,ann_hydro['avg_hydro']],axis=1)
comp['diff']=(comp.iloc[:,1]-comp.iloc[:,0])/comp.iloc[:,1]
comp=pd.concat([comp,ann_fcrps],axis=1)

son=fcrps2[fcrps2['year']==1168]
diff=(son.mean()-fcrps2.mean())/fcrps2.mean()




########so maybe we need a different regression for each month?############
X=mon_fcrps[mon_fcrps['month']==3]
y=mon_hydro[mon_hydro['month']==3]

cor=pd.concat([X,y],axis=1)
cor.corr()['hydro']

X_train,X_test,y_train,y_test=tts(X.loc[:,['Dalls ARF','Orofino H']],y['hydro'],test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)



##let's take Dworkshak 2,3,6
##seat HDD 11,12
##eug CDD 1,2,11
##eug HDD 8
##dalls ARF 4,5,7,8,9,10

dalls_mons=[4,5,7,8,9,10]
dalls=mon_fcrps[mon_fcrps.month.isin(dalls_mons)]
dalls=dalls.groupby('year').agg('mean')
dalls=dalls['Dalls ARF']
eug=mon_weath[mon_weath.month.isin([1,2,8,11])]
eug=eug.groupby('year').agg('mean')
eug=eug['eug_CDD']
eug2=mon_weath[mon_weath.month.isin([8])]
eug2=eug2.groupby('year').agg('mean')
eug2=eug2['eug_HDD']
seat=mon_weath[mon_weath.month.isin([1,2,3])]
seat=seat.groupby('year').agg('mean')
seat=seat['SEATTLE_W']
dwor=mon_fcrps[mon_fcrps.month.isin([2,3,6])]
dwor=dwor.groupby('year').agg('mean')
dwor=dwor['Dworshak M']


anns=pd.concat([dalls,seat,dwor],axis=1)
X_train,X_test,y_train,y_test=tts(anns,ann_hydro['avg_hydro'],test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)



##make each month  of tda a separate input for the regression
j=mon_fcrps[mon_fcrps.month==1]
j=j['Dalls ARF']
j=pd.DataFrame(j)
j.reset_index(inplace=True)
j=j.iloc[:,1]
j.columns=['jan']

f=mon_fcrps[mon_fcrps.month==2]
f=f['Dalls ARF']
f=pd.DataFrame(f)
f.reset_index(inplace=True)
f=f.iloc[:,1]
f.columns=['feb']

ma=mon_fcrps[mon_fcrps.month==3]
ma=ma['Dalls ARF']
ma=pd.DataFrame(ma)
ma.reset_index(inplace=True)
ma=ma.iloc[:,1]
ma.columns=['mar']

ap=mon_fcrps[mon_fcrps.month==4]
ap=ap['Dalls ARF']
ap=pd.DataFrame(ap)
ap.reset_index(inplace=True)
ap=ap.iloc[:,1]
ap.columns=['apr']

m=mon_fcrps[mon_fcrps.month==5]
m=m['Dalls ARF']
m=pd.DataFrame(m)
m.reset_index(inplace=True)
m=m.iloc[:,1]
m.columns=['may']

ju=mon_fcrps[mon_fcrps.month==6]
ju=ju['Dalls ARF']
ju=pd.DataFrame(ju)
ju.reset_index(inplace=True)
ju=ju.iloc[:,1]
ju.columns=['jun']

jul=mon_fcrps[mon_fcrps.month==7]
jul=jul['Dalls ARF']
jul=pd.DataFrame(jul)
jul.reset_index(inplace=True)
jul=jul.iloc[:,1]
jul.columns=['jul']

a=mon_fcrps[mon_fcrps.month==8]
a=a['Dalls ARF']
a=pd.DataFrame(a)
a.reset_index(inplace=True)
a=a.iloc[:,1]
a.columns=['aug']

s=mon_fcrps[mon_fcrps.month==9]
s=s['Dalls ARF']
s=pd.DataFrame(s)
s.reset_index(inplace=True)
s=s.iloc[:,1]
s.columns=['sep']

o=mon_fcrps[mon_fcrps.month==10]
o=o['Dalls ARF']
o=pd.DataFrame(o)
o.reset_index(inplace=True)
o=o.iloc[:,1]
o.columns=['oct']

n=mon_fcrps[mon_fcrps.month==11]
n=n['Dalls ARF']
n=pd.DataFrame(n)
n.reset_index(inplace=True)
n=n.iloc[:,1]
n.columns=['nov']

d=mon_fcrps[mon_fcrps.month==12]
d=d['Dalls ARF']
d=pd.DataFrame(j)
d.reset_index(inplace=True)
d=d.iloc[:,1]
d.columns=['dec']


sep_tda=pd.concat([f,ma],axis=1)
sep_tda.columns=['feb','mar']

adding=pd.concat([sep_tda,ann_fcrps.loc[:,['Libby M','Orofino H']],ann_weath.loc[:,['SPOKANE_T','boi_HDD','boi_CDD','eug_HDD','eug_CDD']]],axis=1)

X_train,X_test,y_train,y_test=tts(adding,ann_hydro['avg_hydro'],test_size=.2)
est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(r2(pred,y_test))
print(r2(pred_tr,y_train))

plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)





apr_jun=mon_fcrps[mon_fcrps.month.isin([4,5,6])]
apr_jun=apr_jun.drop(columns=['index','month','level_0'])
apr_jun=apr_jun.groupby('year').agg({'mean'})

X=apr_jun[['Dalls ARF']]
X_train,X_test,y_train,y_test=tts(X,ann_hydro['avg_hydro'],test_size=.2,random_state=1)

est=sm.OLS(y_train,X_train)
est2=est.fit()
print(est2.summary())
all_pred=est2.predict(X)
pred=est2.predict(X_test)
pred_tr=est2.predict(X_train)
print(str(r2(pred,y_test)))
print(str(r2(all_pred,ann_hydro['avg_hydro'])))


plt.scatter(pred,y_test,alpha=.5)
plt.scatter(pred_tr,y_train,alpha=.5)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
X=ann_fcrps[['Dalls ARF','Chief Joseph L']]
model.fit(X, ann_hydro['avg_hydro'])
model.score(X, ann_hydro['avg_hydro'])
#display adjusted R-squared
1 - (1-model.score(X, ann_hydro['avg_hydro']))*(len(ann_hydro['avg_hydro'])-1)/(len(ann_hydro['avg_hydro'])-X.shape[1]-1)
X=sm.add_constant(X)
model=sm.OLS(ann_hydro['avg_hydro'],X).fit()
print(model.rsquared_adj)

















