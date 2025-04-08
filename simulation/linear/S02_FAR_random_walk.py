# %%
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import sys


# %%
seed_id = sys.argv[1] # "seed_20000" # 
n_steps_ahead = int(sys.argv[2]) # 1 # 

print('Number of steps ahead: ',n_steps_ahead)

# working directory
wdir = "/hpctmp/e0823043/SPX/simulation_AHBS/all_params_linear/"

# for RANDOM WALK: to predict for day t+h, use day t
# hence i use lags = 1 here
lags = [1]

# output directory
odir = wdir + "S02_FAR_random_walk/steps" + str(n_steps_ahead) + "ahead/" 
os.makedirs(odir,exist_ok=True)  


# %%
# dates (rows of dates correspond to row of original data)
dates = pd.read_csv("S01_sim_data/" + seed_id + "_date_indexing.csv")
dates.Date = pd.to_datetime(dates['Date'], format='%Y%m%d')


# %%
# dates (rows of dates correspond to row of original data)
dates = pd.read_csv(wdir + "S01_sim_data/" + seed_id + "_date_indexing.csv")
dates.Date = pd.to_datetime(dates['Date'], format='%Y%m%d')

dat = pd.read_csv(wdir + "S01_sim_data/" + seed_id + "_obspline_coeffss.csv")
dat = dat.drop(["Date"],axis = 1)
# set index for the original data
dat = dat.set_index(dates.Date)


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

# %%
# number of equally spaced points (for each day) 
n_out = Y.shape[1]

# number of inputs used to make prediction at each day
n_in = X.shape[1]

# actual number of days to be used for training and testing
n_days = split_dates.shape[0]

# number of days to be used for training
n_trainval = 1600
n_val = 400
n_train = n_trainval - n_val

# number of days to be used for testing
n_test = n_days - n_trainval


# %%
# This function will predict for n_steps_ahead days: test_day_idx, test_day_idx+1, ..., test_day_idx+n_steps_ahead-1
# using day (test_day_idx-1) and average of the last k days (test_day_idx-1,...,test_day_idx-k) for each k in lags as input
def pred_test_day(i):
    # i = 2
    test_day_idx = i+n_trainval
    test_date = split_dates.iloc[test_day_idx]

    # test set
    test_x = X[test_day_idx,:].reshape((1,n_in))
    test_y = Y[test_day_idx,:].reshape((1,n_out))

    # get prediction in the original space
    test_y_pred = test_x

    # all dates to be predicted for 
    all_test_dates_i = dates.Day[dates['Date']==test_date.Date].values.item()
    all_test_dates_i = dates.loc[dates['Day']==(all_test_dates_i+n_steps_ahead),:]

    # merge predicted basis coefficients and basis values to make prediction for IV
    test_pred_IV = pd.DataFrame(test_y_pred,
                                columns = ["basis_" + str(i) for i in range(1,n_out+1)])
    test_pred_IV['test_date'] = test_date.Date.strftime("%Y%m%d")
    test_pred_IV['Date'] = all_test_dates_i.Date.values
    test_pred_IV = test_pred_IV.rename(columns={"Date": "test_day_ahead_date"})
    
    return test_pred_IV


# %%
all_test_preds = pd.DataFrame()
for i in range(n_test): # n_test
    out = pred_test_day(i)
    all_test_preds = pd.concat([all_test_preds,out])


# %%
# read in splines values at grid locations
splines_vals = pd.read_csv(wdir + "S01_sim_data/" + seed_id + "_obspline_values.csv")
splines_vals1 = splines_vals[['basis_' + str(i+1) for i in range(n_out)]].to_numpy()

all_pred_IVs = pd.DataFrame()
for td in list(all_test_preds.test_day_ahead_date):
    pred_basis_coefs = all_test_preds[all_test_preds['test_day_ahead_date']==td][['basis_' + str(i+1) for i in range(n_out)]]
    
    pred_IVs = np.matmul(splines_vals1, pred_basis_coefs.to_numpy().T)
    pred_IVs = pd.DataFrame(pred_IVs, columns=['pred_IV'])
    pred_IVs['Date'] = td
    pred_IVs['option_index'] = splines_vals['option_index']

    all_pred_IVs = all_pred_IVs.append(pred_IVs) 


# "observed" IV values
options = pd.read_csv(wdir + "S01_sim_data/" + seed_id + "_full_data.csv")
options['Date'] = pd.to_datetime(options['Date'], format='%Y%m%d')

# inner join to get pred IV and observed IV of test dates
options = pd.merge(options, all_pred_IVs)


# %%
overall_RMSE = mean_squared_error(y_true = options.IV, y_pred = options.pred_IV, squared=False)
overall_MAE = mean_absolute_error(y_true = options.IV, y_pred = options.pred_IV)
overall_MAPE = mean_absolute_percentage_error(y_true = options.IV, y_pred = options.pred_IV)
overall_OoR2 = r2_score(y_true = options.IV, y_pred = options.pred_IV)

overall_accuracy = pd.DataFrame({'period': ['overall'],
                                 'RMSE': [overall_RMSE],
                                 'MAE': [overall_MAE],
                                 'MAPE': [overall_MAPE],
                                 'OoR2': [overall_OoR2]})

all_accuracy = pd.concat([overall_accuracy])
print("Test IV accuracy (at interpolated points)")
print(all_accuracy)

all_accuracy.to_csv(odir + seed_id + "_pred_test_IV_accuracy.csv",index=False)

# %%