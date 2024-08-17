# %%
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
import warnings
warnings.filterwarnings('ignore')
import sys


# %%
obsplines_type = sys.argv[1] # "degree3_tau_NaN_M_NaN" # 
option_type = sys.argv[2] # "Call" # 
n_steps_ahead = int(sys.argv[3]) # 20 # 
kernel_type = sys.argv[4] # "rbf" # 

print('Option type: ',option_type,', number of steps ahead: ',n_steps_ahead,', kernel: ',kernel_type)

# working directory
wdir = "/hpctmp/e0823043/SPX/S04_FAR/OrthogonalBsplines/"

# lags to be used for prediction
lags = [1, 5, 22]

n_neurons = 500

# output directory
odir = wdir + "S03_FAR_KRR/" + kernel_type + "/steps" + str(n_steps_ahead) + "ahead/" 
os.makedirs(odir,exist_ok=True)  


# %%
# read in orthogonal bsplines coefficients
dat = pd.read_csv(wdir + "S01_OrthogonalBsplines/" + obsplines_type + "/" + option_type + "/obspline_coeffss.csv")
dat.Date = pd.to_datetime(dat['Date'], format='%Y%m%d')
dat.columns = [c.replace("basis", "coef") for c in list(dat.columns)]

dates = dat[['Date']]

dat = dat.set_index('Date')

# option data as well as bsplines values at observed (tau,M) in each day
options = pd.read_csv(wdir + "S01_OrthogonalBsplines/" + obsplines_type + "/" + option_type + "/obspline_values.csv")
options.Date = pd.to_datetime(options['Date'], format='%Y%m%d')


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
    dates1['predicting_Date'] = dates['Date'][(max_lag-1+n_steps_ahead):(num_obs-n_steps_ahead+n_steps_ahead)].values

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


# %%
# number of bspline coefs (for each day) 
n_out = Y.shape[1]

# number of inputs used to make prediction at each day
n_in = X.shape[1]

# actual number of days to be used for training and testing
n_days = split_dates.shape[0]

# number of days to be used for training
n_trainval = 2500
n_val = 500
n_train = n_trainval - n_val

# number of days to be used for testing
n_test = n_days - n_trainval


# %%
# Function to tune a given NTK model structure (with given epochs, batch size, # neurons, learning rate, random seed)
# to find the best weight decay rate
def tuning_KRR(train_x, train_y, val_x, val_y,
               kernel_type, ridge_alpha_values, gamma_values):
    
    #### standardize train and val sets ####
    scaler_x = StandardScaler()
    train_x1 = scaler_x.fit_transform(train_x)
    val_x1 = scaler_x.transform(val_x)

    scaler_y = StandardScaler()
    train_y1 = scaler_y.fit_transform(train_y)
    val_y1 = scaler_y.transform(val_y)

    #### Tune KRR model ####
    min_val_rmse = float('inf')
    hyperparam_vals = np.array(np.meshgrid(ridge_alpha_values, gamma_values)).T.reshape(-1,2)
    for j in range(hyperparam_vals.shape[0]): 
        selected_alpha = hyperparam_vals[j][0]
        selected_gamma = hyperparam_vals[j][1]

        krr_model = KernelRidge(kernel = kernel_type, 
                                alpha = selected_alpha, gamma = selected_gamma)
        krr_model.fit(X = train_x1,y = train_y1)

        #### Out-sample predictive errors ####
        val_y_preds = krr_model.predict(val_x1) # Making Predictions
        
        val_rmse = mean_squared_error(y_true = val_y_preds, y_pred = val_y1, squared=False)
        if val_rmse < min_val_rmse:
            min_val_rmse = val_rmse
            best_alpha = selected_alpha
            best_gamma = selected_gamma
            
            print("Current best alpha and gamma is %f and %f" % (best_alpha,best_gamma))

    return pd.DataFrame({'Ridge_alpha': [best_alpha],
                         'gamma': [best_gamma],
                         'tuned': ['Yes']})


# %%
# This function will predict for n_steps_ahead days: test_day_idx, test_day_idx+1, ..., test_day_idx+n_steps_ahead-1
# using day (test_day_idx-1) and average of the last k days (test_day_idx-1,...,test_day_idx-k) for each k in lags as input
def pred_test_day(i, kernel_type, tuned_ridge_alpha, tuned_gamma):
    # i = 0

    test_day_idx = i + n_trainval
    test_date = split_dates.iloc[test_day_idx]['Date']
    test_day_ahead_date = split_dates.iloc[test_day_idx]['predicting_Date']

    if i % 100 == 0:
        print("Train and predict for test date: ",test_date)

     #### Hyperparameter tuning ####
    # tune for every 120 test days
    if i % 120 == 0:
        gamma_vals = [0.02, 0.025, 0.03]
        selected_hyperparams = tuning_KRR(train_x = X[(test_day_idx-n_trainval):(test_day_idx-n_val),:], 
                                          train_y = Y[(test_day_idx-n_trainval):(test_day_idx-n_val),:], 
                                          val_x = X[(test_day_idx-n_val):test_day_idx,:], 
                                          val_y = Y[(test_day_idx-n_val):test_day_idx,:], 
                                          kernel_type = kernel_type, 
                                          ridge_alpha_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                                          gamma_values = gamma_vals
                                         ) 
    else:
        selected_hyperparams = pd.DataFrame({'Ridge_alpha': [tuned_ridge_alpha],
                                             'gamma': [tuned_gamma],
                                             'tuned' : [float("nan")]})
    selected_hyperparams['test_date'] = test_date


    #### Training with both train + validation data, and make test prediction ####
    # train set
    train_x = X[(test_day_idx-n_trainval):test_day_idx,:]
    train_y = Y[(test_day_idx-n_trainval):test_day_idx,:]

    # test set
    test_x = X[test_day_idx,:].reshape((1,n_in))
    test_y = Y[test_day_idx,:].reshape((1,n_out))

    #### standardize train and test sets ####
    scaler_x = StandardScaler()
    train_x1 = scaler_x.fit_transform(train_x)
    test_x1 = scaler_x.transform(test_x)

    scaler_y = StandardScaler()
    train_y1 = scaler_y.fit_transform(train_y)
    test_y1 = scaler_y.transform(test_y)

    #### Train model ####
    krr_model = KernelRidge(kernel = kernel_type,
                            alpha = float(selected_hyperparams['Ridge_alpha'].values),
                            gamma = float(selected_hyperparams['gamma'].values))
    krr_model.fit(X = train_x1, y = train_y1)


    #### Out-sample predictive errors ####
    test_y_pred = krr_model.predict(test_x1) # Making Predictions
    test_y_pred = scaler_y.inverse_transform(test_y_pred)  # undo standardization 

    # orthogonal bsplines values at observed (tau,M) location on day t+h
    obsplines_vals = options.loc[options['Date']==test_day_ahead_date]\
        [['basis_' + str(i+1) for i in range(n_out)]]

    # IV prediction
    IV_pred = np.matmul(obsplines_vals.to_numpy(), test_y_pred.T)

    out = options.loc[options['Date']==test_day_ahead_date].\
        drop(columns = ['basis_' + str(i+1) for i in range(n_out)])
    
    out['fcst_IV'] = IV_pred

    out = out.rename(columns={"Date": "test_day_ahead_date"})
    out['test_date'] = test_date

    return selected_hyperparams, out


# %%
all_test_preds = pd.DataFrame()
best_hyperparams = pd.DataFrame()

for i in range(n_test): # n_test
    if i == 0: 
        new_o1, o2 = pred_test_day(i,
                                   kernel_type = kernel_type, 
                                   tuned_ridge_alpha = 0.1,
                                   tuned_gamma = 0.02)
    else: 
        new_o1, o2 = pred_test_day(i,
                                   kernel_type = kernel_type, 
                                   tuned_ridge_alpha = float(o1['Ridge_alpha'].values),
                                   tuned_gamma = float(o1['gamma'].values)
                                   )
    o1 = new_o1 
    
    # record best hyperparameter and test result for day i
    best_hyperparams = pd.concat([best_hyperparams,o1])
    all_test_preds = pd.concat([all_test_preds,o2])

# remove all test days that we dont use to tune hyperparameter 
best_hyperparams = best_hyperparams.dropna()
best_hyperparams = best_hyperparams.drop(columns=['tuned'])


# %%
overall_RMSE = mean_squared_error(y_true = all_test_preds.IV, 
                                  y_pred = all_test_preds.fcst_IV,squared=False)
overall_MAE = mean_absolute_error(y_true = all_test_preds.IV, 
                                  y_pred = all_test_preds.fcst_IV)
overall_MAPE = mean_absolute_percentage_error(y_true = all_test_preds.IV, 
                                              y_pred = all_test_preds.fcst_IV)

overall_accuracy = pd.DataFrame({'period': ['overall'],
                                 'RMSE': [overall_RMSE],
                                 'MAE': [overall_MAE],
                                 'MAPE': [overall_MAPE]})


before_covid_preds = all_test_preds[all_test_preds["test_day_ahead_date"] < "2020-01-01"]
if before_covid_preds.shape[0] != 0:
    before_covid_RMSE = mean_squared_error(y_true = before_covid_preds.IV, y_pred = before_covid_preds.fcst_IV,squared=False)
    before_covid_MAE = mean_absolute_error(y_true = before_covid_preds.IV, y_pred = before_covid_preds.fcst_IV)
    before_covid_MAPE = mean_absolute_percentage_error(y_true = before_covid_preds.IV, y_pred = before_covid_preds.fcst_IV)

    before_covid_accuracy = pd.DataFrame({'period': ['before_Covid'],
                                          'RMSE': [before_covid_RMSE],
                                          'MAE': [before_covid_MAE],
                                          'MAPE': [before_covid_MAPE]}) 
else:
    before_covid_accuracy = pd.DataFrame({'period': [],
                                          'RMSE': [],
                                          'MAE': [],
                                          'MAPE': []}) 

after_covid_preds = all_test_preds[all_test_preds["test_day_ahead_date"] >= "2020-01-01"]
if after_covid_preds.shape[0] != 0:
    after_covid_RMSE = mean_squared_error(y_true = after_covid_preds.IV, y_pred = after_covid_preds.fcst_IV,squared=False)
    after_covid_MAE = mean_absolute_error(y_true = after_covid_preds.IV, y_pred = after_covid_preds.fcst_IV)
    after_covid_MAPE = mean_absolute_percentage_error(y_true = after_covid_preds.IV, y_pred = after_covid_preds.fcst_IV)

    after_covid_accuracy = pd.DataFrame({'period': ['after_Covid'],
                                          'RMSE': [after_covid_RMSE],
                                          'MAE': [after_covid_MAE],
                                          'MAPE': [after_covid_MAPE]}) 
else:
    after_covid_accuracy = pd.DataFrame({'period': [],
                                          'RMSE': [],
                                          'MAE': [],
                                          'MAPE': []}) 


# %%
all_accuracy = pd.concat([overall_accuracy,before_covid_accuracy,after_covid_accuracy])
print("Observed test IV accuracy: ")
print(all_accuracy)


# %%
all_test_preds.to_csv(odir + option_type + "_pred_test_IV.csv",index=False)
all_accuracy.to_csv(odir + option_type + "_pred_test_IV_accuracy.csv",index=False)
best_hyperparams.to_csv(odir  + option_type + "_best_hyperparams.csv",index=False)


# %%




