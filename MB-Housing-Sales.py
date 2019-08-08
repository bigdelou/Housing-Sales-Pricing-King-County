# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:15:30 2019

@author: mbigdelou
"""

import os
os.chdir(r'C:\Users\mbigdelou\Desktop\Datamining project')

os.getcwd()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

plt.interactive(False)

#========================
#import dataset
df = pd.read_csv('kc_house_data.csv')

df.shape
df.describe()
df.head()

list(df)
df.info()


#checking if any value is missing
df.isnull().any()

df.notnull()
df.notnull().sum()
df.dropna(how='all')

#=======================Pre-processing
#Check data type
print(df.dtypes)

#distribution of variables and cheching log-normal
from scipy import stats
sns.distplot(df.price);
plt.show()

lnprice = np.log(df.price)
sns.distplot(lnprice);


df['lnprice']=pd.DataFrame(lnprice)

sns.jointplot(x="sqft_above", y="lnprice", xlim=(0,4000), ylim=(11.5,14.5), data=df, kind="hex", color="k");
plt.show()

#sns.jointplot(x="sqft_above", y="price", xlim=(0,4000), ylim=(11.5,14.5), data=df, kind="hex", color="k");
#plt.show()


lnsqft_above = np.log(df.sqft_above)
sns.jointplot(x=lnsqft_above, y=lnprice, kind="hex", color="k");
plt.show()
sns.jointplot(x=lnsqft_above, y=lnprice, kind="kde");
plt.show()

sns.jointplot(x="long", y="lat", data=df, kind="hex", color="k");
plt.show()

#===
# Dropping Unwanted Variables
df = df.drop(['id','date'], axis = 1)
#drop by index number: df = df.drop([21], axis = 1)


#===Correlation
# Correlation of target variable ('price') with all independent variables:
df.corr()['price'].sort_values(ascending=False)
df.corr()['lnprice'].sort_values(ascending=False)

# Correlation Matrix
correlation = df.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.show()

#sns.pairplot(df);
#plt.show()

#===
# Creating Dummy Variables
'''
df.yr_renovated.value_counts()
descript = df.describe(include='all')
descript
 
  
df['yr_renovated'] =df['yr_renovated'].apply(lambda x: 0 if x ==0 else 1)
'''df_dummy = pd.get_dummies(df,columns=['yr_renovated'], drop_first=True, dummy_na=True) '''
'''
#========================
#understanding the distribution with seaborn
with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(df[['sqft_lot','sqft_above','price','sqft_living','bedrooms', 'sqft_living15', 'sqft_lot15']], 
                 hue='bedrooms', palette='tab20',size=6)
plt.show()


#Scatter-plot with price
plt.scatter(df.sqft_lot,lnprice, c='r',marker='*')
plt.show()

plt.scatter(df.sqft_living,lnprice, c='g',marker='*', label='x')
plt.xlabel('sqft_living')
plt.ylabel('lnprice')
plt.show()

plt.scatter(df.lat,lnprice, c='g',marker='*')
plt.show()

plt.scatter(df.long,lnprice, c='g',marker='*')
plt.show()

plt.scatter(df.yr_built,lnprice, c='y',marker='^')
plt.ylabel('lnprice')
plt.show()

plt.scatter(df.bedrooms,lnprice, c='b',marker='+')
plt.xlabel('bedrooms')
plt.ylabel('lnprice')
plt.show()

#dropping the row with an outlier bedroom
df_out = df.drop(df.index[15870])
plt.scatter(df_out.bedrooms,df_out.lnprice, c='b',marker='+')
plt.xlabel('bedrooms')
plt.ylabel('lnprice')
plt.show()

#========================
# Normalization of Variables
#since Price is big number we might have a problem on calculation then we have to normalize it
#Min-Max Normalization
min_max_scaler = MinMaxScaler()

column_names_to_normalize = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']
x = df[column_names_to_normalize].values
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
df[column_names_to_normalize] = df_temp

#==========Feature Selection
#since there were only 21 variables, I didn't run feature selection at this stage

#========================
#Separation of independent and dependent variable
X = df.drop(['price','lnprice'],axis=1)
y = df.lnprice

#========================
#Splitting dataset into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=0) 

#========================
#estimation by LinearRegression
from sklearn.linear_model import LinearRegression
lr = LinearRegression() 
lr.fit(X_train,y_train) 
lr.coef_ 
lr.intercept_

lr.score(X_train,y_train)
lr.score(X_test,y_test) #R-squared for test dataset

y_pred = lr.predict(X_test)


import statsmodels.api as sm
X2=sm.add_constant(X_train)
ols = sm.OLS(y_train,X2) 
lr_model = ols.fit() 
print(lr_model.summary())

#estimation by RLM
#statsmodels.robust.robust_linear_model.RLM(endog, exog, M=<statsmodels.robust.norms.HuberT object>, missing='none', **kwargs)
huber_t = sm.RLM(endog=y_train, exog=X2, M=sm.robust.norms.HuberT())
hub_results = huber_t.fit()
print(hub_results.summary())
print(hub_results.weights)

#==========================Regression diagnostics
#================================================
#================================================
#================================================
#===================Testing for MultiCollinearity
#==============Variance Inflation Factors:

#Code for VIF Calculation
#Writing a function to calculate the VIF values

def VIF_cal(X_train2):
    import statsmodels.formula.api as smf
    x_vars = X_train2
    xvar_names = x_vars.columns
    for i in range(0,len(xvar_names)):
        y=x_vars[xvar_names[i]]
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=smf.ols(formula="y~x", data=x_vars).fit().rsquared
        vif=round(1/(1-rsq),3)
        print(xvar_names[i], "VIF = ", vif)

#Calculating VIF values using the VIF_cal function

X_vif = X_train

VIF_cal(X_vif)

#y_vif = y_train
#X_vif = X_train.loc[:,['season','holiday','workingday','weather','atemp','casual','registered']]
X_vif = X_train.drop(['sqft_living', 'sqft_above', 'sqft_lot15'],axis=1)

X_vif1 = sm.add_constant(X_vif)

#Final Model
ols = sm.OLS(y_train,X_vif1) 
lr = ols.fit() 
print(lr.summary())

''
# Plot line / model
lr2 = LinearRegression()
lr2.fit(X_vif,y_train)
X_test_after_VIF = X_test.drop(['sqft_living', 'sqft_above', 'sqft_lot15'],axis=1)
y_pred_after_VIF = lr2.predict(X_test_after_VIF)
plt.scatter(y_test, y_pred_after_VIF)
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.show()
''

# Plot residuals
residual = y_test - y_pred_after_VIF

import scipy.stats as ss
residual_z = np.array(ss.zscore(residual))

plt.scatter(y_pred_after_VIF, residual_z)
plt.xlabel("Predictions")
plt.ylabel("Residual")
#plt.ylim(-5.0, 5.0)
plt.show()


#================================================
#================================================
#======================Normality of the residuals 
sns.distplot(np.array(residual));
plt.show()

sns.distplot(residual_z);
plt.show()

#======================Jarque-Bera test:
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name1 = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test1 = sms.jarque_bera(lr.resid)
lzip(name1, test1)
#null hypothesis: the data is normally distributed.

#======================Omni test:
name2 = ['Chi^2', 'Two-tail probability']
test2 = sms.omni_normtest(lr.resid)
lzip(name2, test2)
'''

#================================================
#================================================
#=========================Heteroskedasticity test
#======================Breush-Pagan test:
name3 = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test3 = sms.het_breuschpagan(lr.resid, lr.model.exog)
lzip(name3, test3)

#======================Goldfeld-Quandt test:
name5 = ['F statistic', 'p-value']
test5 = sms.het_goldfeldquandt(lr.resid, lr.model.exog)
lzip(name5, test5)


#================================================
#================================================
#==================================Linearity test
#======================Harvey-Collier:
name6 = ['t value', 'p value']
test6 = sms.linear_harvey_collier(lr)
lzip(name6, test6)

import statsmodels.stats.diagnostic as ssd
name6 = ['t value', 'p value']
test6 = ssd.acorr_linear_rainbow(lr)
lzip(name6, test6)

#================================================
#================================================
#====Serial correlation (or) Autocorrelation test
#======================Durbin_watson:
#Durbin-Watson test for no autocorrelation of residuals
#printed with summary()
from statsmodels.stats.stattools import durbin_watson
print("Durbin-Watson: ", durbin_watson(lr.resid))




#================================================
#================================================
#=====================Performance Measures
#====
#R-Squared and Adj. R-Squared of the train set
print ('R-Squared:', lr.rsquared, ';', 'Adj. R-Squared', lr.rsquared_adj)

#R-Squared of Test set
lr2.score(X_test_after_VIF,y_test)

#RMSE
from sklearn.metrics import mean_squared_error
import math

mse_lr = mean_squared_error(y_test,y_pred_after_VIF)
rmse_lr = math.sqrt(mse_lr)
print('RMSE of lr model:', rmse_lr)


# Plot line of y_test and y_pred 
plt.scatter(y_test, y_pred_after_VIF)
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.show()

plt.scatter(y_pred_after_VIF, y_test)
plt.xlabel("Predictions")
plt.ylabel("Actual Values")
plt.show()

#or
#sns.jointplot(x="y_pred_after_VIF", y="y_test", xlim=(0,4000), ylim=(11.5,14.5), data=df, kind="hex", color="k");
h = sns.jointplot(x=y_pred_after_VIF, y=y_test)
h.set_axis_labels('Predictions', 'Actual Values', fontsize=16)
plt.show()
#sns.jointplot(x=y_pred, y=y_test, kind="kde");

h = sns.jointplot(x=y_pred_after_VIF, y=y_test, kind="hex", color="k");
h.set_axis_labels('Predictions', 'Actual Values', fontsize=16)
plt.show()



#correl
#pd: df['A'].corr(df['B'])
print (np.corrcoef(y_test,y_pred_after_VIF))
#or
from scipy.stats.stats import pearsonr   
print (pearsonr(y_test,y_pred_after_VIF)) #with p-value


# Plot residuals
residual = y_test - y_pred_after_VIF 

import scipy.stats as ss
residual_z = np.array(ss.zscore(residual))

plt.scatter(y_pred, residual_z)
plt.xlabel("Predictions")
plt.ylabel("Residual")
#plt.ylim(-5.0, 5.0)
plt.show()

# Plot residuals
plt.figure(figsize=(8, 6), dpi=70)
sns.residplot(y_test, y_pred_after_VIF)
plt.show()



#==== K-Folds Cross Validation (6-fold cross validation)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

scores_lr = cross_val_score(lr2.fit(X_vif,y_train), X_vif, y_train, cv=6)
print ('Cross-validated scores lr:', scores_lr)

# Plot cross validated predictions 
predictions_lr = cross_val_predict(lr2.fit(X_vif,y_train), X_vif, y_train, cv=6)
plt.scatter(y_train, predictions_lr)
plt.show()

#accuracy
accuracy_lr = metrics.r2_score(y_train, predictions_lr)
print ('Cross-Predicted Accuracy_lr:', accuracy_lr)


