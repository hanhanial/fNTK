# %%
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import os
import datetime
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
import sys

# %%
option_type = sys.argv[1]
n_steps_ahead = int(sys.argv[2])
print('Option type: ',option_type,', number of steps ahead: ',n_steps_ahead)

# lags to be used for prediction
lags = [1,5,22]

# working directory
wdir = "/hpctmp/e0823043/SPX/S02_Bernales2014/Daily_2009_2021_fullAHBS/M_Andersen/"

# output directory
odir = wdir + "S03_VARp/steps" + str(n_steps_ahead) + "ahead/"
os.makedirs(odir,exist_ok=True)  


# %%
options = pd.read_csv(wdir + 'S01_processing_data/' + option_type + "_2009_2021.csv")
options.Date = pd.to_datetime(options['Date'], format='%Y-%m-%d')

dat = pd.read_csv(wdir + 'S02_DeterministicLinearModel/' + option_type + "_2009_2021.csv")
dat.Date = pd.to_datetime(dat['Date'], format='%Y-%m-%d')

dates = {'Date': dat.Date.values,'Day' : np.arange(dat.shape[0])+1}
dates = pd.DataFrame.from_dict(dates)
dates.head()

# set index for the original data
dat = dat.set_index(dat.Date)
dat = dat.drop(columns=['RMSE','Date','R2'])


# %%
# function to compute average of n_lag lags if take_avg=True, otherwise just return all n_lag lagged days
def prepare_data(data,dates,lags,n_steps_ahead,take_avg = True):
    # lags = number of lags to be used as inputs for prediction
    # n_steps_ahead = number of steps (days) ahead to be predicted for
    
    # maximum number of lags to be used as input 
    max_lag = max(lags)
  
    # total number of observations
    num_obs = data.shape[0]
    
    # we need to get the corresponding dates
    # since only data of days with index from max_lag (in python index is (max_lag-1)) 
    # to num_obs (in python index is (num_obs-1))  will have enough lagged input, 
    # and since we need to predict n_steps_ahead days ahead for prediction, 
    # so the latest day we can predict for is (num_obs-1-n_steps_ahead) 
    # but in the slicing index, the last index is not included, so we need to put 
    # (num_obs-1-n_steps_ahead+1) = (num_obs-n_steps_ahead)
    dates1 = dates.iloc[(max_lag-1):(num_obs-n_steps_ahead)]
    dates1['Day'] = range(1,dates1.shape[0]+1)
    
    # initialize input and output matrix
    inp = []
    out = []
    
    for d in range((max_lag-1),(num_obs-n_steps_ahead)):
        # if input day is day d, then output day is day (d+n_steps_ahead)
        y = data.iloc[(d+n_steps_ahead)].to_numpy()
        
        # get corresponding lagged input x for y
        x = []
        for l in lags:
            # day d itself is an input day, so when we use l lags, it includes days d, d-1,..., d-l+1
            # whether to take the average of the l lag(s) (i.e. of days d, d-1,..., d-l+1)
            if take_avg == False:
                # if we don't take the average of the l lag(s), just the l-th lag itself
                tmp = data.iloc[(d-l+1)]
            else:
                # if we take the average of the last l lags
                if l==1:
                    tmp = data.iloc[(d-l+1)] # since there is no need to take average of the l = 1 lag
                else:
                    tmp = data.iloc[(d-l+1):(d+1)] # data of of days d, d-1,..., d-l+1
                    tmp = np.mean(tmp,axis=0)
            x = np.append(x,tmp)
        
        out.append(y)
        inp.append(x)
        
    return dates1, inp, out 


# %%
split_dates, X, Y = prepare_data(data = dat,
                                     dates = dates,
                                     lags = lags,
                                     n_steps_ahead = n_steps_ahead,
                                     take_avg = True)
X = np.array(X)
Y = np.array(Y)

print(X.shape)
print(Y.shape)
print(split_dates.shape)


# %%
# number of bspline coefs (for each day) 
n_bases = Y.shape[1]

# number of inputs used to make prediction at each day
n_in = X.shape[1]

# actual number of days to be used for training and testing
n_days = split_dates.shape[0]

# number of days to be used for training
n_train = 2500

# number of days to be used for testing
n_test = n_days - n_train


# %%
def pred_test_day(i):
    # i = 1
    test_day_idx = i+n_train
    test_date = split_dates.iloc[test_day_idx]

    if (i % 200)==0:
        print("Train and predict for test date: ",test_date.Date)
    
    train_dates = split_dates.iloc[(test_day_idx-n_train):test_day_idx]
    
    # This means that we will predict for n_steps_ahead days: test_day_idx, test_day_idx+1, ..., test_day_idx+n_steps_ahead-1
    # using day (test_day_idx-1) and average of the last k days (test_day_idx-1,...,test_day_idx-k) for each k in lags as input
    # train set
    train_x = X[(test_day_idx-n_train):test_day_idx,:]
    train_y = Y[(test_day_idx-n_train):test_day_idx,:]

    # test set
    test_x = X[test_day_idx,:].reshape((1,n_in))
    test_y = Y[test_day_idx,:].reshape((1,n_bases))

    #### train model ####
    model = LinearRegression().fit(train_x, train_y)

    #### Out-sample predictive errors ####
    test_y_pred = model.predict(test_x)

    # all dates to be predicted for 
    all_test_dates_i = dates.Day[dates['Date']==test_date.Date].values.item()
    all_test_dates_i = dates.loc[dates['Day']==(all_test_dates_i+n_steps_ahead),:]

    # merge predicted basis coefficients and basis values to make prediction for IV
    test_pred_IV = pd.DataFrame(test_y_pred,columns = ["b" + str(i) for i in range(0,n_bases)])
    test_pred_IV['test_date'] = test_date.Date
    test_pred_IV['Date'] = all_test_dates_i.Date.values
    test_pred_IV = test_pred_IV.merge(options,on = "Date", how = "inner")

    tmp1 = test_pred_IV['b1']*test_pred_IV['M']
    tmp2 = test_pred_IV['b2']*test_pred_IV['M2']
    tmp3 = test_pred_IV['b3']*test_pred_IV['tau']
    tmp4 = test_pred_IV['b4']*test_pred_IV['tau2']
    tmp5 = test_pred_IV['b5']*test_pred_IV['Mtau']
    test_pred_IV['fcst_IV'] = test_pred_IV['b0'] + tmp1 + tmp2 + tmp3 + tmp4 + tmp5

    #test_pred_IV = test_pred_IV[['test_date','Date','M', 'Maturity','IV', 'fcst_IV']]
    test_pred_IV = test_pred_IV.drop(columns=['b0', 'b1','b2','b3','b4','b5','IV_atm','Year',
                                              'min_K_F_dist','min_K_F_ratio','M2','tau2','Mtau','ln_IV'])
    test_pred_IV = test_pred_IV.rename(columns={"Date": "test_day_ahead_date"})

    #### In-sample predictive errors ####
    if ((i == (n_test-100)) and (n_steps_ahead >= 10)):
        train_y_pred = model.predict(train_x)

        train_pred_IV = pd.DataFrame(train_y_pred,columns = ["b" + str(i) for i in range(0,n_bases)])
        train_pred_IV["train_date"] = train_dates.Date.values

        all_train_dates = pd.DataFrame()
        for train_day_i in range(train_dates.shape[0]):
            train_day = train_dates.Date.values[train_day_i]
            tmp = dates.Day[dates['Date']==train_day].values.item()
            all_train_days = dates[dates['Day']==(tmp+n_steps_ahead)].Date.values
            tmp1 = pd.DataFrame({'Date': all_train_days})
            tmp1['train_date'] = train_day
            all_train_dates = pd.concat([all_train_dates,tmp1])

        train_pred_IV = train_pred_IV.merge(all_train_dates,how = "inner")

        train_pred_IV = train_pred_IV.merge(options,on = 'Date',how = "inner")

        tmp1 = train_pred_IV['b1']*train_pred_IV['M']
        tmp2 = train_pred_IV['b2']*train_pred_IV['M2']
        tmp3 = train_pred_IV['b3']*train_pred_IV['tau']
        tmp4 = train_pred_IV['b4']*train_pred_IV['tau2']
        tmp5 = train_pred_IV['b5']*train_pred_IV['Mtau']
        train_pred_IV['fcst_IV'] = train_pred_IV['b0'] + tmp1 + tmp2 + tmp3 + tmp4 + tmp5

        train_pred_IV = train_pred_IV.rename(columns={"Date": "pred_day_ahead_date"})
        train_pred_IV = train_pred_IV[['train_date','pred_day_ahead_date',
                                   'M', 'Maturity','IV', 'fcst_IV']]
        
        train_pred_IV.to_csv(odir + option_type + "_pred_train_IV_" + test_date.Date.strftime("%Y%m%d") + ".csv",index=False)
            
    return test_pred_IV


# %%
def main():
    pool_size = 6
    
    with multiprocessing.Pool(processes=pool_size) as pool:
        pool = multiprocessing.Pool(processes=pool_size)
        
        i = [x for x in range(n_test)] 
        
        table1 = pool.map(pred_test_day, i)
          
    return table1



# %%
if __name__ == '__main__':
    o1 = main()    
    
    all_test_preds = pd.concat(o1, axis=0)
    

    overall_RMSE = mean_squared_error(all_test_preds.IV,all_test_preds.fcst_IV,squared=False)
    overall_MAE = mean_absolute_error(all_test_preds.IV,all_test_preds.fcst_IV)
    overall_MAPE = mean_absolute_percentage_error(all_test_preds.IV,all_test_preds.fcst_IV)
    
    overall_accuracy = pd.DataFrame({'period': ['overall'],
                                     'RMSE': [overall_RMSE],
                                     'MAE': [overall_MAE],
                                     'MAPE': [overall_MAPE]})
    
    before_covid = all_test_preds[all_test_preds['test_day_ahead_date'] < '2020-01-01']
    if before_covid.empty == False:
        before_covid_RMSE = mean_squared_error(before_covid.IV,before_covid.fcst_IV,squared=False)
        before_covid_MAE = mean_absolute_error(before_covid.IV,before_covid.fcst_IV)
        before_covid_MAPE = mean_absolute_percentage_error(before_covid.IV,before_covid.fcst_IV)
    
        before_covid_accuracy = pd.DataFrame({'period': ['before_Covid'],
                                              'RMSE': [before_covid_RMSE],
                                              'MAE': [before_covid_MAE],
                                              'MAPE': [before_covid_MAPE]}) 
    else:
        before_covid_accuracy = pd.DataFrame({'period': [],
                                              'RMSE': [],
                                              'MAE': [],
                                              'MAPE': []}) 

    after_covid = all_test_preds[all_test_preds['test_day_ahead_date'] >= '2020-01-01']
    if after_covid.empty == False:
        after_covid_RMSE = mean_squared_error(after_covid.IV,after_covid.fcst_IV,squared=False)
        after_covid_MAE = mean_absolute_error(after_covid.IV,after_covid.fcst_IV)
        after_covid_MAPE = mean_absolute_percentage_error(after_covid.IV,after_covid.fcst_IV)
    
        after_covid_accuracy = pd.DataFrame({'period': ['after_Covid'],
                                              'RMSE': [after_covid_RMSE],
                                              'MAE': [after_covid_MAE],
                                              'MAPE': [after_covid_MAPE]}) 
    else:
        after_covid_accuracy = pd.DataFrame({'period': [],
                                              'RMSE': [],
                                              'MAE': [],
                                              'MAPE': []}) 
    
    all_accuracy = pd.concat([overall_accuracy,before_covid_accuracy,after_covid_accuracy])
    print(all_accuracy)
    
    all_test_preds.to_csv(odir + option_type + "_pred_test_IV.csv",index=False)
    all_accuracy.to_csv(odir + option_type + "_pred_test_IV_accuracy.csv",index=False)


