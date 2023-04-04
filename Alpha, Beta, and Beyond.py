# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 00:06:53 2023

@author: banik
"""

import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
from fredapi import Fred
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas_datareader.data as web
from pandas_datareader import data
import statsmodels.api as sm
from statsmodels.tools import tools
from statsmodels.tools.validation import array_like, float_like
from scipy.stats import norm as Gaussian
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

# Tickers: UNH, CAH, XOM, LLY, COP

S1 = input("Ticker for stock 1: ").upper()
S2 = input("Ticker for stock 2: ").upper()
S3 = input("Ticker for stock 3: ").upper()
S4 = input("Ticker for stock 4: ").upper()
S5 = input("Ticker for stock 5: ").upper()

Tickers = [S1,S2,S3,S4,S5]
print(f'selected tickers: {Tickers}')
d0 = datetime.date.today()
d1 = d0 - relativedelta(years=11)

# Get API key: Fred
fred = Fred(api_key='0bbc318d3ba2efaf9d4e56708954067d')
pd.set_option('max_columns', 30) 
rf = fred.get_series('DGS1MO',d1,d0).asfreq('M').bfill().tail(-1)/100
rt = yf.Tickers(Tickers).history('11y','1d').Close.dropna().resample("M").last().pct_change().dropna().head(-1)
rm = yf.Ticker('^GSPC').history('11y','1d').Close.dropna().resample("M").last().pct_change().dropna().head(-1)
rm.index = rf.index
re = rt.sub(rf, axis=0)
re['mrp'] = rm - rf
print('\n')
print('Excess return data of 5 stocks and market risk premium')
print("")
print(re)

#Get ordinary least squares estimates
pd.set_option('max_columns', None) 
pd.set_option('max_rows', 30)
beta_OLS = []
for i in range(len(Tickers)):
    model = sm.OLS(re.iloc[:,i], sm.add_constant(re.iloc[:,-1])).fit()
    print(model.summary())
    icpt = round(model.params[0],4)
    coef = round(model.params[1],4)
    t_icpt = round(model.tvalues[0],4)
    t_coef = round(model.tvalues[1],4)
    p_icpt = round(model.pvalues[0],4)
    p_coef = round(model.pvalues[1],4)
    beta_OLS.append([icpt,coef,t_icpt,t_coef,p_icpt,p_coef])
    plt.style.use('seaborn-darkgrid') #print(plt.style.available)
    plt.scatter(x = re.iloc[:,-1], y = re.iloc[:,i])
    plt.xlabel("Market risk premium (%)")
    plt.ylabel(f"Excess return of Asset {i+1}: {Tickers[i]} (%)")
    plt.axline((0,0),(1,1),linewidth = 0.5, color='black',linestyle='--')
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.2,0.5)
    plt.show()
    print(model.params)
beta_OLS = pd.DataFrame(beta_OLS, columns = ['alpha','beta','t (alpha)','t (beta)','p (alpha)','p (beta)']).T
beta_OLS.columns = Tickers
print('\n')
print("OLS beta:::")
print(beta_OLS)

#Machine learning approach: Testing the model
for j in range(len(Tickers)):
    x = np.array(re.iloc[:,-1]).reshape(-1,1)
    y = np.array(re.iloc[:,j]).reshape(-1,1)
    
    # Split between training data and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)
    # create linear regression object and train the model using the training sets
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    print('\n')
    print(f'Running OLS using the training data:::')
    print(f'Coefficients of asset {j+1}, {Tickers[j]}: ', float(np.round(reg.coef_,3)))
    
    # check whether variance score is the perfect predictor (i.e., if var_score = 1)
    print(f'Variance score with test data of the asset {j+1}, {Tickers[j]}: {np.round(reg.score(x_test, y_test),3)}')
    print(f'Variance score with train data of the asset {j+1}, {Tickers[j]}: {np.round(reg.score(x_train, y_train),3)}')
    
    #Get MSE and RMSE for training data
    MSE_train = sum(np.square(reg.predict(x_train) - y_train))/len(y_train)
    RMSE_train = np.sqrt(MSE_train)
    print(f'Mean Squared Error of the asset {j+1}, {Tickers[j]}: {float(np.round(MSE_train,3))}')
    print(f'Root Mean Squared Error of the asset {j+1}, {Tickers[j]}: {float(np.round(RMSE_train,3))}')
    
    #Get MSE and RMSE for training data
    MSE_test = sum(np.square(reg.predict(x_test) - y_test))/len(y_test)
    RMSE_test = np.sqrt(MSE_test)
    print(' ')
    print(f'Test the training model:::')
    print(f'Mean Squared Error of the asset {j+1} (Test data), {Tickers[j]}: {float(np.round(MSE_test,3))}')
    print(f'Root Mean Squared Error of the asset {j+1} (Test data), {Tickers[j]}: {float(np.round(RMSE_test,3))}')
    
    # Plot residual errocumul of training data and test data
    plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train,
                color = "red", s = 50, label = 'Training data') #plotting residual lines for training data
    plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test,
                color = "green", s = 50, label = 'Test data',) #plotting residual lines for test data
    plt.hlines(y = 0, xmax=0.2,xmin=-0.2,linewidth = 1, label = 'Perfect predictor') #plotting line for the perfect predictor
    plt.legend(loc = 'lower left')
    plt.title(f"Residual errors for asset {j+1}, {Tickers[j]}")
    plt.show()

# shrinkage_estimator:
beta_JS = np.zeros((len(Tickers),2))
for i in range(0,len(Tickers)):
    mean_beta = np.matrix(beta_OLS)[1].mean()
    asset_beta = np.matrix(beta_OLS)[1,i]
    var = (asset_beta - mean_beta).T * (asset_beta - mean_beta) # variance of individual asset
    v = (1/len(Tickers))*var # pooled variance
    alpha = 1 - ((v*(len(Tickers)-3))/var) # shrinkage parameter
    beta_JS[i,0] = alpha
    beta_JS[i,1] = mean_beta + alpha*(asset_beta - mean_beta) # James-Stein estimator
    
beta_JS = pd.DataFrame(beta_JS,columns=['alpha','beta']).T 
beta_JS.columns = Tickers # Efficient beta under shrinkage estimator
print('\n')
print('Shrinkage Estimator parameters:::')
print(round(beta_JS,3))
print('\n')
comp = pd.DataFrame()
comp['OLS beta'] = beta_OLS.iloc[1,:]
comp['Shrinkage estimator beta'] = beta_JS.T
print(comp)

#Least absolute deviations
LAD_beta = np.zeros((len(Tickers),1))
LAD_alpha = np.zeros((len(Tickers),1))
for b in range(len(Tickers)):
    X = re.iloc[:,-1]
    Y = re.iloc[:,b]
    
    # Define the Model
    def f(X, a, b): return a * X + b
    
    # The objective Function to minimize (least-squares regression)
    def obj(X, Y, a, b): return np.sum(abs(Y - f(X, a, b)))
    
    # define the bounds -infty < a < infty,  b <= 0
    bounds = [(None, None), (None, 0)]
    
    # res.x contains your coefficients
    res = minimize(lambda coeffs: obj(X, Y, *coeffs), x0=np.zeros(2), bounds=bounds)
    LAD_beta[b] = res.x[0]
    LAD_alpha[b] = res.x[1]
print("")
print("Coefficients under least absolute deviations (LAD):::")
print("")
LAD = pd.DataFrame()
LAD['Alpha'] = pd.DataFrame(LAD_alpha.T, columns=Tickers).T
LAD['Beta'] = pd.DataFrame(LAD_beta.T, columns=Tickers).T
print(LAD)
print("")
comp['LAD beta'] = LAD.iloc[:,1]
print(comp)

# Out-of-sample tests
initial_window = re.iloc[0:60,:]
os_window = re.iloc[60:,:]
cumul_roll = np.zeros((len(os_window),len(Tickers)))#len(os_window.columns)))
cumul_alpha = np.zeros((len(os_window),len(Tickers)))
cumul_beta = np.zeros((len(os_window),len(Tickers)))
zero_rt = pd.DataFrame(np.zeros((len(re),len(Tickers)))) 
zero_rt.index = rm.index
zero_re = zero_rt.subtract(rf, axis=0)
zero_re['mrp'] = rm - rf
zero_roll = np.zeros((len(os_window),len(Tickers)))
for m in range(len(os_window)): 
    for n in range(len(Tickers)):
        cumul_df = re[:len(initial_window)+m]
        x_cumul = np.array(cumul_df.iloc[:,-1]).reshape(-1,1)
        y_cumul = np.array(cumul_df.iloc[:,n]).reshape(-1,1)
        reg = linear_model.LinearRegression()
        reg.fit(x_cumul, y_cumul)
        MSE_cumul = np.matrix(sum(np.square(reg.predict(x_cumul) - y_cumul))/(len(y_cumul)+1))
        RMSE_cumul = np.sqrt(MSE_cumul)
        cumul_roll[m,n]= RMSE_cumul
        cumul_alpha[m,n] = reg.intercept_
        cumul_beta[m,n] = reg.coef_   
        zero_df = zero_re[:len(initial_window)+m]
        zero_x_cumul = np.array(zero_df.iloc[:,-1]).reshape(-1,1)
        zero_y_cumul = np.array(zero_df.iloc[:,n]).reshape(-1,1)
        reg0 = linear_model.LinearRegression()
        reg0.fit(zero_x_cumul, zero_y_cumul)
        MSE_cumul0 = np.matrix(sum(np.square(reg0.predict(zero_x_cumul) - zero_y_cumul))/(len(zero_y_cumul)+1))
        RMSE_cumul0 = np.sqrt(MSE_cumul0)
        zero_roll[m,n] = RMSE_cumul0
cumul_roll = pd.DataFrame(cumul_roll, columns=Tickers)
cumul_roll.index = list(range(len(initial_window)+1,len(re)+1))
cumul_roll.index.name = 'Observations'
print('\n')
print(f'''1-month out of sample RMSE for each stock with 
      a cumulative rolling estimation window:::''')
print("")
print(cumul_roll)

cumul_alpha = pd.DataFrame(cumul_alpha, columns=Tickers)
cumul_alpha.index = list(range(len(initial_window)+1,len(re)+1))
cumul_alpha.index.name = 'Out-of-sample periods (months)'
cumul_beta = pd.DataFrame(cumul_beta, columns=Tickers)
cumul_beta.index = list(range(len(initial_window)+1,len(re)+1))
cumul_beta.index.name = 'Out-of-sample periods (months)'

plt.subplots(figsize=(10, 6))
plt.plot(cumul_alpha.index.values,cumul_alpha.iloc[:,0],color='C1',label=Tickers[0])
plt.plot(cumul_alpha.index.values,cumul_alpha.iloc[:,1],color='C2',label=Tickers[1])
plt.plot(cumul_alpha.index.values,cumul_alpha.iloc[:,2],color='C3',label=Tickers[2])
plt.plot(cumul_alpha.index.values,cumul_alpha.iloc[:,3],color='C4',label=Tickers[3])
plt.plot(cumul_alpha.index.values,cumul_alpha.iloc[:,4],color='C5',label=Tickers[4])
plt.xlabel('Cumulative rolling estimation window (months)')
plt.ylabel('alpha')
plt.legend(labelspacing=1.5,bbox_to_anchor=(1.2,0.50),loc='right')
plt.show()

plt.subplots(figsize=(10, 6))
plt.plot(cumul_beta.index.values,cumul_beta.iloc[:,0],color='C1',label=Tickers[0])
plt.plot(cumul_beta.index.values,cumul_beta.iloc[:,1],color='C2',label=Tickers[1])
plt.plot(cumul_beta.index.values,cumul_beta.iloc[:,2],color='C3',label=Tickers[2])
plt.plot(cumul_beta.index.values,cumul_beta.iloc[:,3],color='C4',label=Tickers[3])
plt.plot(cumul_beta.index.values,cumul_beta.iloc[:,4],color='C5',label=Tickers[4])
plt.xlabel('Cumulative rolling estimation window (months)')
plt.ylabel('beta')
plt.legend(labelspacing=1.5,bbox_to_anchor=(1.2,0.50),loc='right')
plt.show()

zero_roll = pd.DataFrame(zero_roll, columns=Tickers)
zero_roll.index = list(range(len(initial_window)+1,len(re)+1))
zero_roll.index.name = 'Observations'
print('\n')
print(f'''1-month out of sample RMSE for each stock (assuming an expected return of 0) 
      with a cumulative rolling estimation window:::''')
print("")
print(zero_roll)

fix_roll = np.zeros((len(initial_window),len(Tickers)))#len(os_window.columns)))
fix_alpha = np.zeros((len(initial_window),len(Tickers)))
fix_beta = np.zeros((len(initial_window),len(Tickers)))
fix_roll0 = np.zeros((len(initial_window),len(Tickers)))
for k in range(len(initial_window)): 
    for n in range(len(Tickers)):
        fix_df = re[k:len(initial_window)+k]
        x_fix = np.array(fix_df.iloc[:,-1]).reshape(-1,1)
        y_fix = np.array(fix_df.iloc[:,n]).reshape(-1,1)
        reg_fix = linear_model.LinearRegression()
        reg_fix.fit(x_fix, y_fix)
        MSE_fix = np.matrix(sum(np.square(reg_fix.predict(x_fix) - y_fix))/len(y_fix))
        RMSE_fix = np.sqrt(MSE_fix)
        fix_roll[k,n]= RMSE_fix
        fix_alpha[k,n] = reg_fix.intercept_
        fix_beta[k,n] = reg_fix.coef_   
        zero_df1 = zero_re[k:len(initial_window)+k]
        zero_x_fix = np.array(zero_df1.iloc[:,-1]).reshape(-1,1)
        zero_y_fix = np.array(zero_df1.iloc[:,n]).reshape(-1,1)
        reg_fix0 = linear_model.LinearRegression()
        reg_fix0.fit(zero_x_fix, zero_y_fix)
        MSE_fix0 = np.matrix(sum(np.square(reg_fix0.predict(zero_x_fix) - zero_y_fix))/len(zero_y_fix))
        RMSE_fix0 = np.sqrt(MSE_fix0)
        fix_roll0[k,n] = RMSE_fix0

fix_roll = pd.DataFrame(fix_roll, columns=Tickers)
index = np.matrix([list(range(1,len(initial_window)+1)),
                           list(range(len(initial_window),len(initial_window)+len(initial_window)))]).T
fix_roll.index = list(range(1,len(initial_window)+1))
fix_roll.index.name = '60 observations start at month:'
print('\n')
print(f'''1-month out of sample RMSE for each stock with 
      a rolling fixed 60-month estimation window:::''')
print("")
print(fix_roll)

rolling_alpha = pd.DataFrame(fix_alpha, columns=Tickers)
rolling_alpha.index = list(range(1,len(initial_window)+1))
rolling_alpha.index.name = 'Out-of-sample periods (months)'
rolling_beta = pd.DataFrame(fix_beta, columns=Tickers)
rolling_beta.index = list(range(1,len(initial_window)+1))
rolling_beta.index.name = 'Out-of-sample periods (months)'

plt.subplots(figsize=(10, 6))
plt.plot(rolling_alpha.index.values,rolling_alpha.iloc[:,0],color='C1',label=Tickers[0])
plt.plot(rolling_alpha.index.values,rolling_alpha.iloc[:,1],color='C2',label=Tickers[1])
plt.plot(rolling_alpha.index.values,rolling_alpha.iloc[:,2],color='C3',label=Tickers[2])
plt.plot(rolling_alpha.index.values,rolling_alpha.iloc[:,3],color='C4',label=Tickers[3])
plt.plot(rolling_alpha.index.values,rolling_alpha.iloc[:,4],color='C5',label=Tickers[4])
plt.xlabel('Fixed 60-month rolling estimation window starts at month')
plt.ylabel('alpha')
plt.legend(labelspacing=1.5,bbox_to_anchor=(1.2,0.50),loc='right')
plt.show()

plt.subplots(figsize=(10, 6))
plt.plot(rolling_beta.index.values,rolling_beta.iloc[:,0],color='C1',label=Tickers[0])
plt.plot(rolling_beta.index.values,rolling_beta.iloc[:,1],color='C2',label=Tickers[1])
plt.plot(rolling_beta.index.values,rolling_beta.iloc[:,2],color='C3',label=Tickers[2])
plt.plot(rolling_beta.index.values,rolling_beta.iloc[:,3],color='C4',label=Tickers[3])
plt.plot(rolling_beta.index.values,rolling_beta.iloc[:,4],color='C5',label=Tickers[4])
plt.xlabel('Fixed 60-month rolling estimation window starts at month')
plt.ylabel('beta')
plt.legend(labelspacing=1.5,bbox_to_anchor=(1.2,0.50),loc='right')
plt.show()

fix_roll0 = pd.DataFrame(fix_roll0, columns=Tickers)
index = np.matrix([list(range(1,len(initial_window)+1)),
                           list(range(len(initial_window),len(initial_window)+len(initial_window)))]).T
fix_roll0.index = list(range(1,len(initial_window)+1))
fix_roll0.index.name = '60 observations start at month:'
print('\n')
print(f'''1-month out of sample RMSE for each stock (assuming expected return 
      of each stock is 0) with a rolling fixed 60-month estimation window:::''')
print("")
print(fix_roll0)



is_df = np.matrix(initial_window.iloc[:,:-1])
os_df = np.matrix(os_window.iloc[:,:-1])
is_rows = is_df.shape[0]
is_col = is_df.shape[1]
is_row_vector = np.matrix(np.ones(is_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
is_mean_vector = is_df.T*is_row_vector/is_rows  #sample mean vector
is_cov_matrix = (is_df-is_mean_vector.T).T*(is_df-is_mean_vector.T)/(is_rows-1) #sample covariance matrix
is_col_vector=np.matrix(np.ones(is_col)).T #create a col_vectorit vector (i.e., is_col x 1 vector) with columns

#in-sample MSR weights
is_MSR_w = (pow(is_cov_matrix,-1)*is_mean_vector)/(is_col_vector.T*pow(is_cov_matrix,-1)*is_mean_vector) #in-sample MSR portfolio weights
is_MSR_Weights = pd.DataFrame(is_MSR_w,index=Tickers,columns=["Weight"])
is_MSR_Weights = is_MSR_Weights.T
#Get the base sample allocation
print('\n')
print('MSR portfolio weights for 60-month initial window')
print(is_MSR_Weights)


os_rolling_MSR = np.zeros((len(is_df)+1,len(Tickers)+2))
i = 1
while i<=len(is_df):
    rolling_df = np.matrix(rt)[i:len(is_df)+i]
    rolling_rows = rolling_df.shape[0]
    rolling_col = rolling_df.shape[1]
    rolling_row_vector = np.matrix(np.ones(rolling_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
    rolling_mean_vector = rolling_df.T*rolling_row_vector/rolling_rows  #sample mean vector
    rolling_cov_matrix = (rolling_df-rolling_mean_vector.T).T*(rolling_df-rolling_mean_vector.T)/(rolling_rows-1) #sample covariance matrix
    rolling_col_vector=np.matrix(np.ones(rolling_col)).T #create a col_vectorit vector (i.e., is_col x 1 vector) with columns
    rolling_MSR_w = (pow(rolling_cov_matrix,-1)*rolling_mean_vector)/(rolling_col_vector.T*pow(rolling_cov_matrix,-1)*rolling_mean_vector) #MSR portfolio weights
    
    os_rows = os_df.shape[0]
    os_col = os_df.shape[1]
    os_row_vector = np.matrix(np.ones(os_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
    os_mean_vector = os_df.T*os_row_vector/os_rows  #sample mean vector
    
    rolling_MSR_r = rolling_MSR_w.T*os_mean_vector #MSR portfolio return
    rolling_MSR_SD = np.sqrt(rolling_MSR_w.T*rolling_cov_matrix*rolling_MSR_w) #MSR portfolio standard deviation
    os_rolling_MSR[i,0] = rolling_MSR_r
    os_rolling_MSR[i,1] = rolling_MSR_SD
    os_rolling_MSR[i,2:7] = rolling_MSR_w.T
    i = i+1

print("\n")
print("MSR portfolio allocation in the rolling factor model results:::")
print("\n")
os_rolling_MSR = pd.DataFrame(os_rolling_MSR,columns=['Expected Return','Standard Deviation',
                                          f'Weight of {Tickers[0]}',f'Weight of {Tickers[1]}',
                                          f'Weight of {Tickers[2]}',f'Weight of {Tickers[3]}',f'Weight of {Tickers[4]}'])
os_rolling_MSR.index.name = "Rolling fixed 60-months"
os_rolling_MSR = os_rolling_MSR.tail(-1)

#Get the out_of_sample allocation
os_rolling_MSR_allocation = os_rolling_MSR.iloc[:, 2:]
print(os_rolling_MSR_allocation)
#Get the utility function
os_rolling_ER = os_rolling_MSR.iloc[:,0].mean()
os_rolling_Var = pow(os_rolling_MSR.iloc[:,1],2).mean()
os_rolling_ut = os_rolling_ER - ((1/2)*4*os_rolling_Var)
print('\n')
print(f'''Out-of-sample utility for the for a mean variance investor
      in the rolling factor model results is : {round(os_rolling_ut,4)}''')


os_cumul_MSR = np.zeros((len(os_df)+1,len(Tickers)+2))
i = 1
while i<=len(os_df):
    cumul_df = np.matrix(rt)[:len(is_df)+i]
    cumul_rows = cumul_df.shape[0]
    cumul_col = cumul_df.shape[1]
    cumul_row_vector = np.matrix(np.ones(cumul_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
    cumul_mean_vector = cumul_df.T*cumul_row_vector/cumul_rows  #sample mean vector
    cumul_cov_matrix = (cumul_df-cumul_mean_vector.T).T*(cumul_df-cumul_mean_vector.T)/(cumul_rows-1) #sample covariance matrix
    cumul_col_vector=np.matrix(np.ones(cumul_col)).T #create a col_vectorit vector (i.e., is_col x 1 vector) with columns
    cumul_MSR_w = (pow(cumul_cov_matrix,-1)*cumul_mean_vector)/(cumul_col_vector.T*pow(cumul_cov_matrix,-1)*cumul_mean_vector) #MSR portfolio weights
    
    os_rows = os_df.shape[0]
    os_col = os_df.shape[1]
    os_row_vector = np.matrix(np.ones(os_rows)).T #creating a row vector (i.e.,is_rows x 1 vector)
    os_mean_vector = os_df.T*os_row_vector/os_rows  #sample mean vector
    
    cumul_MSR_r = cumul_MSR_w.T*os_mean_vector #MSR portfolio return
    cumul_MSR_SD = np.sqrt(cumul_MSR_w.T*cumul_cov_matrix*cumul_MSR_w) #MSR portfolio standard deviation
    os_cumul_MSR[i,0] = cumul_MSR_r
    os_cumul_MSR[i,1] = cumul_MSR_SD
    os_cumul_MSR[i,2:7] = cumul_MSR_w.T
    i = i+1

print("\n")
print("MSR portfolio allocation in the cumulative factor model results:::")
print("\n")
os_cumul_MSR = pd.DataFrame(os_cumul_MSR,columns=['Expected Return','Standard Deviation',
                                          f'Weight of {Tickers[0]}',f'Weight of {Tickers[1]}',
                                          f'Weight of {Tickers[2]}',f'Weight of {Tickers[3]}',f'Weight of {Tickers[4]}'])
os_cumul_MSR.index = list(range(len(initial_window),len(re)+1))
os_cumul_MSR.index.name = "Cumulative fixed 60-months"
os_cumul_MSR = os_cumul_MSR.tail(-1)

#Get the out_of_sample allocation
os_cumul_MSR_allocation = os_cumul_MSR.iloc[:, 2:]
print(os_cumul_MSR_allocation)
#Get the utility function
os_cumul_ER = os_cumul_MSR.iloc[:,0].mean()
os_cumul_Var = pow(os_cumul_MSR.iloc[:,1],2).mean()
os_cumul_ut = os_cumul_ER - ((1/2)*4*os_cumul_Var)
print('\n')
print(f'''Out-of-sample utility for the for a mean variance investor
      in the cumulative factor model results is : {round(os_cumul_ut,4)}''')

## ---(Tue Feb 14 08:52:39 2023)---

##Kalman Filter
alpha = 0
phi = 0.5
var_eta = 0
mean_beta = 1
var_residual = 0

##Create a univariate linear model whether excess market return is the only 
# independent variable
predicted_estimator = np.zeros((len(re),len(Tickers)))
predicted_MSE = np.zeros((len(re),len(Tickers)))
#Predicting predicted estimators and predicted MSEs
for l in range(1,len(re)):
    for i in range(len(Tickers)):
        uni_lm = sm.OLS(re.iloc[:1,i], sm.add_constant(re.iloc[:1,-1])).fit()
        state_var = uni_lm.params #Set the state variable when t=1 (i.e., first month observation)
        predicted_estimator[0,i] = state_var
        predicted_estimator[l,i] = (1-phi)*mean_beta + predicted_estimator[0:l,i].mean()*phi
        predicted_MSE[0,i] = pow((state_var - mean_beta),2)
        predicted_MSE[l,i] = predicted_MSE[0:l,i].mean()*pow(phi,2) + var_eta
print("predicted Estimators (priori value):::")
print("")
display_estimator = pd.DataFrame(predicted_estimator, columns=Tickers)
print(display_estimator)
print("")
print("predicted Mean Squared Error/variance of residuals (priori value):::")
print("")
display_MSE = pd.DataFrame(predicted_MSE, columns=Tickers)
print(display_MSE)

####Updating the equations
#Innovation (or pre-fit residual) covariance
f = np.square(np.matrix(re)[:,-1].T)*predicted_MSE + var_residual #gives me a 1 x 5 matrix

#Innovation or measurement pre-fit residual
upsilon = np.matrix(re)[:,:-1] -(alpha + np.matrix(re)[:,-1].T*predicted_estimator) #gives me a 5 x 1 matrix

# Optimal Kalman gain with innovation
gain_process = ((np.matrix(re)[:,-1].T*predicted_MSE)*upsilon.T).T*f

#Updated (a posteriori) state estimate
updated_estimator = predicted_estimator + gain_process
updated_estimator = pd.DataFrame(updated_estimator, columns=Tickers)

#Updated (a posteriori) estimate covariance
updated_MSE = predicted_MSE * (1- (predicted_MSE.T*np.square(np.matrix(re)[:,-1]))*f)
updated_MSE = pd.DataFrame(updated_MSE, columns=Tickers).tail(-1)

print("Optimal Estimators (Posterior value):::")
print("")
display_estimator1 = pd.DataFrame(updated_estimator, columns=Tickers)
print(display_estimator1)
print("")
print("Optimal Mean Squared Error/variance of residuals (posterior value):::")
print("")
display_MSE1 = pd.DataFrame(updated_MSE, columns=Tickers)
print(display_MSE1)


#Plot the posteriori state estimate
plt.subplots(figsize=(10, 6))
plt.plot(updated_estimator.index.values,updated_estimator.iloc[:,0],color='C1',label=Tickers[0])
plt.plot(updated_estimator.index.values,updated_estimator.iloc[:,1],color='C2',label=Tickers[1])
plt.plot(updated_estimator.index.values,updated_estimator.iloc[:,2],color='C3',label=Tickers[2])
plt.plot(updated_estimator.index.values,updated_estimator.iloc[:,3],color='C4',label=Tickers[3])
plt.plot(updated_estimator.index.values,updated_estimator.iloc[:,4],color='C5',label=Tickers[4])
plt.xlabel('Observations in months')
plt.ylabel('Optimal Kalman Filter Estimator')
plt.legend(labelspacing=1.5,bbox_to_anchor=(1.2,0.50),loc='right')
plt.show()

#Plot the posteriori estmate covariance
plt.subplots(figsize=(10, 6))
plt.plot(updated_MSE.index.values,updated_MSE.iloc[:,0],color='C1',label=Tickers[0])
plt.plot(updated_MSE.index.values,updated_MSE.iloc[:,1],color='C2',label=Tickers[1])
plt.plot(updated_MSE.index.values,updated_MSE.iloc[:,2],color='C3',label=Tickers[2])
plt.plot(updated_MSE.index.values,updated_MSE.iloc[:,3],color='C4',label=Tickers[3])
plt.plot(updated_MSE.index.values,updated_MSE.iloc[:,4],color='C5',label=Tickers[4])
plt.xlabel('Observations in months')
plt.ylabel('Optimal Kalman Filter Residuals')
plt.legend(labelspacing=1.5,bbox_to_anchor=(1.2,0.50),loc='right')
plt.show()


#Bayessian Regression estimates
import sys
import seaborn as sns
import pymc3 as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
dt = pd.DataFrame(scale.fit_transform(re),columns=['y1','y2','y3','y4','y5','x'])

for i in range(0,len(Tickers)):
    my_model = pm.Model()
    with my_model:
        pm.glm.GLM(x=dt.x,y=dt.iloc[:,i], family = pm.glm.families.Normal())
        #using MAP(MAximum A Posteriori as initial value for MCMC sampler)
        initial = pm.find_MAP() 
        trace = pm.sample(119,chains=2, return_inferencedata=False)
    
    with my_model:
        ppc = pm.sample_posterior_predictive(
            trace, random_seed=42, progressbar=True)
    
    with my_model: 
        trace_updated = az.from_pymc3(trace, posterior_predictive=ppc)
    
    az.style.use("arviz-darkgrid")
    with my_model:
      az.plot_trace(trace_updated, figsize=(17,10), legend=True)
      plt.title(f"Trace plots for Asset_{i+1} ({Tickers[i]})",fontsize=40)
    
    az.style.use("arviz-darkgrid")
    with my_model:
      az.plot_posterior(trace_updated,textsize=16)
      plt.title(f"Posterior distributions for Asset_{i+1} ({Tickers[i]})",fontsize=40)
      print(trace_updated.posterior)