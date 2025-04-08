# %%
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
import warnings
warnings.filterwarnings('ignore')
import sys


# %%
seed_id = sys.argv[1] # "seed_10000" # 
n_steps_ahead = int(sys.argv[2]) # 1 # 
kernel_type = sys.argv[3] # "rbf" # 

# n_trainval should be less than actual_n_trainval
n_trainval = int(sys.argv[4]) # 800 # 

gamma_vals = float(sys.argv[5])
ridge_alpha_vals = float(sys.argv[6])

print('Number of steps ahead: ',n_steps_ahead, ', kernel type: ',kernel_type)

# working directory
wdir = "/hpctmp/e0823043/SPX/simulation_AHBS/all_params_2Sin4Cos/"

# lags to be used for prediction
lags = [1]

# output directory
odir = wdir + "S02_FAR_KRR/" + kernel_type + "/steps" + str(n_steps_ahead) + "ahead_TrainSize" + str(n_trainval) + "/"
os.makedirs(odir,exist_ok=True)  


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

# number of days to be used for training, before starting test period
actual_n_trainval = 1600

n_val = int(np.round(n_trainval*0.2)) # 
n_train = n_trainval - n_val #  

# number of days to be used for testing
n_test = n_days - actual_n_trainval


# This function will predict for n_steps_ahead days: test_day_idx, test_day_idx+1, ..., test_day_idx+n_steps_ahead-1
# using day (test_day_idx-1) and average of the last k days (test_day_idx-1,...,test_day_idx-k) for each k in lags as input
def tuning_KRR(train_x, train_y, val_x, val_y,
               kernel_type,ridge_alpha_values,gamma_values):
    
    #### standardize train and val sets ####
    scaler_x = StandardScaler()
    train_x1 = scaler_x.fit_transform(train_x)
    val_x1 = scaler_x.transform(val_x)

    scaler_y = StandardScaler()
    train_y1 = scaler_y.fit_transform(train_y)
    val_y1 = scaler_y.transform(val_y)

    #### Train model ####
    min_val_rmse = float('inf')
    hyperparam_vals = np.array(np.meshgrid(ridge_alpha_values, gamma_values)).T.reshape(-1,2)
    for j in range(hyperparam_vals.shape[0]): 
        selected_alpha = hyperparam_vals[j][0]
        selected_gamma = hyperparam_vals[j][1]
#         print("Trying %f and %f" % (selected_alpha,selected_gamma))
        
        krr_model = KernelRidge(kernel = kernel_type, 
                                alpha = selected_alpha, 
                                gamma = selected_gamma)
        krr_model.fit(X = train_x1,y = train_y1)

        #### Out-sample predictive errors ####
        val_y_preds = krr_model.predict(val_x1) # Making Predictions
        
        val_rmse = mean_squared_error(y_true = val_y_preds, y_pred = val_y1,squared=False)
    
        if val_rmse < min_val_rmse:
            min_val_rmse = val_rmse
            best_alpha = selected_alpha
            best_gamma = selected_gamma

            print("Current best alpha and gamma is %f and %f with RMSE of %f" % (best_alpha,best_gamma,min_val_rmse))

    return pd.DataFrame({'Ridge_alpha': [best_alpha],
                         'gamma': [best_gamma],
                         'tuned': ['Yes']})


# %%
#### Hyperparameter tuning ####
selected_hyperparams = tuning_KRR(train_x = X[(actual_n_trainval-n_trainval):(actual_n_trainval-n_val),:], 
                                  train_y = Y[(actual_n_trainval-n_trainval):(actual_n_trainval-n_val),:], 
                                  val_x = X[(actual_n_trainval-n_val):actual_n_trainval,:], 
                                  val_y = Y[(actual_n_trainval-n_val):actual_n_trainval,:], 
                                  kernel_type = kernel_type, 
                                  ridge_alpha_values = ridge_alpha_vals,
                                  gamma_values = gamma_vals
                                 ) 


# %%
#### Training with both train + validation data, and make test prediction ####
train_dates = split_dates.iloc[(actual_n_trainval-n_trainval):actual_n_trainval]
test_date = split_dates.iloc[actual_n_trainval:n_days]

# train set
train_x = X[(actual_n_trainval-n_trainval):actual_n_trainval,:]
train_y = Y[(actual_n_trainval-n_trainval):actual_n_trainval,:]

# test set
test_x = X[actual_n_trainval:n_days,:]
test_y = Y[actual_n_trainval:n_days,:]


# %%
#### standardize train and test sets ####
scaler_x = StandardScaler()
train_x1 = scaler_x.fit_transform(train_x)
test_x1 = scaler_x.transform(test_x)

scaler_y = StandardScaler()
train_y1 = scaler_y.fit_transform(train_y)
test_y1 = scaler_y.transform(test_y)


# %%
#### Train model ####
krr_model = KernelRidge(kernel = kernel_type,
                        alpha = float(selected_hyperparams['Ridge_alpha'].values),
                        gamma = float(selected_hyperparams['gamma'].values))
krr_model.fit(X = train_x1,y = train_y1)


# %%
#### Out-sample predictive errors ####
test_y_preds = krr_model.predict(test_x1) # Making Predictions
test_y_preds1 = scaler_y.inverse_transform(test_y_preds) # inverse scaler


# %%
# all dates to be predicted for (i.e. date of day t+h)
all_test_dates_i = dates.loc[dates['Day'].isin(test_date.Day.values + n_steps_ahead),:]


# %%
# merge predicted basis coefficients and basis values to make prediction for IV
test_pred_IV = pd.DataFrame(test_y_preds1,
                            columns = ["basis_" + str(i) for i in range(1,n_out+1)])
test_pred_IV['test_date'] = test_date.Date.dt.strftime("%Y%m%d").values
test_pred_IV['Date'] = all_test_dates_i.Date.values
test_pred_IV = test_pred_IV.rename(columns={"Date": "test_day_ahead_date"})


# %%
# read in splines values at grid locations
splines_vals = pd.read_csv(wdir + "S01_sim_data/" + seed_id + "_obspline_values.csv")
splines_vals1 = splines_vals[['basis_' + str(i+1) for i in range(n_out)]].to_numpy()

all_pred_IVs = pd.DataFrame()
for td in list(test_pred_IV.test_day_ahead_date):
    pred_basis_coefs = test_pred_IV[test_pred_IV['test_day_ahead_date']==td][['basis_' + str(i+1) for i in range(n_out)]]
    
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

# %%
all_accuracy.to_csv(odir + seed_id + "_pred_test_IV_accuracy.csv",index=False)
selected_hyperparams.to_csv(odir  + seed_id + "_best_hyperparams.csv",index=False)

# %%




