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
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchmetrics import MeanAbsoluteError
from torch.utils.data import TensorDataset, DataLoader

# %%
option_type = sys.argv[1] # "Call" # 
n_steps_ahead = int(sys.argv[2]) # 1 # 
weight_decay = float(sys.argv[3]) #0.001

print('Option type: ',option_type,', number of steps ahead: ',n_steps_ahead)

learning_rate = 0.001
EPOCH = 500

# lags to be used for prediction
lags = [1] # only use lag 1 for random walk

# working directory
wdir = "/hpctmp/e0823043/SPX/S02_Bernales2014/Daily_2009_2021_fullAHBS/M_Andersen/"

# output directory
odir = wdir + "S05_DNN_on_AHBSresiduals/steps" + str(n_steps_ahead) + "ahead/"
os.makedirs(odir,exist_ok=True)  


# %% Load in and process data
options = pd.read_csv(wdir + 'S01_processing_data/' + option_type + "_2009_2021.csv")
options.Date = pd.to_datetime(options['Date'], format='%Y-%m-%d')

dat = pd.read_csv(wdir + 'S02_DeterministicLinearModel/' + option_type + "_2009_2021.csv")
dat.Date = pd.to_datetime(dat['Date'], format='%Y-%m-%d')


# %%
dates = {'Date': dat.Date.values,'Day' : np.arange(dat.shape[0])+1}
dates = pd.DataFrame.from_dict(dates)
dates.head()

# set index for the original data
dat = dat.set_index(dat.Date)
dat = dat.drop(columns=['RMSE','Date','R2'])

# dat.head()


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


# %% Moving window prediction
def pred_test_day(i):
    # i = 1

    test_day_idx = i + n_train
    test_date = split_dates.iloc[test_day_idx]

    if (i % 200)==0:
        print("Train and predict for test date: ",test_date.Date)

    # test set
    test_x = X[test_day_idx,:].reshape((1,n_in))
    test_y = Y[test_day_idx,:].reshape((1,n_bases))

    # Random walk prediction
    test_y_pred = test_x

    # date to be predicted for, i.e. date of day t+h
    all_test_dates_i = dates.Day[dates['Date']==test_date.Date].values.item()
    all_test_dates_i = dates.loc[dates['Day']==(all_test_dates_i+n_steps_ahead),:]

    # merge predicted basis coefficients and basis values to make prediction for IV
    AHBS_coefs_t = pd.DataFrame(test_y_pred,columns = ["b" + str(i) for i in range(0,n_bases)])


    ###### FCNN trained on differences (residuales) between fitted IVS (using AHBS) and observed IVS ######
    # fitted IV of day t, using AHBS model of day t
    fitted_IVS_t = AHBS_coefs_t; fitted_IVS_t['Date'] = test_date.Date # day t date
    fitted_IVS_t = fitted_IVS_t.merge(options, on = "Date", how = "inner")
    fitted_IVS_t['AHBS_IV_day_t'] = fitted_IVS_t['b0'] + fitted_IVS_t['b1']*fitted_IVS_t['M'] + \
        fitted_IVS_t['b2']*fitted_IVS_t['M2'] + fitted_IVS_t['b3']*fitted_IVS_t['tau'] + \
        fitted_IVS_t['b4']*fitted_IVS_t['tau2'] + fitted_IVS_t['b5']*fitted_IVS_t['Mtau']

    # compute residuals
    fitted_IVS_t["residual_t"] = fitted_IVS_t['AHBS_IV_day_t'] - fitted_IVS_t['IV']

    # train model with input (tau,m) and output IV of day t
    train_x = fitted_IVS_t[['M','tau']].to_numpy()
    train_y = fitted_IVS_t['residual_t'].to_numpy()
    train_y = np.expand_dims(train_y, axis=1)

    train_x1 = torch.tensor(train_x, dtype = torch.float32)
    train_y1 = torch.tensor(train_y, dtype = torch.float32)

    train_dataset = TensorDataset(train_x1, train_y1)
    train_loader = DataLoader(dataset = train_dataset, 
                                batch_size = train_x1.shape[0],
                                shuffle = True, drop_last = False)

    # FCNN model with 3 hidden layers, following Almeida 2022 architecture
    torch.manual_seed(2468)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    layers = [torch.nn.Linear(in_features = 2, out_features = 32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 32, out_features = 16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 16, out_features = 8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 8, out_features = 1)]
    model = torch.nn.Sequential(*layers)

    optimizer = optim.Adam(model.parameters(), 
                            lr = learning_rate, 
                            weight_decay = weight_decay)
    criterion = torch.nn.MSELoss()

    for epoch in range(EPOCH):
        train_losses = []

        for x, y in train_loader:
            x, y, = x.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            preds = model(x)
            # print(preds)

            loss = criterion(preds, y)
            train_losses.append(loss.item())
            
            loss.backward()
            
            optimizer.step()

        if (epoch % 20 == 0):
            print((f'Epoch {epoch+0:03}: | '
                    f'Train Loss: {np.mean(train_losses):.10f}'))
            

    ###### use AHBS + FCNN trained on day t to predict for day t+h ######
    # predicted (fitted) IV of day (t + h) using AHBS model of day t
    test_pred_IV = AHBS_coefs_t
    test_pred_IV['test_date'] = test_date.Date # day t date
    test_pred_IV['Date'] = all_test_dates_i.Date.values # day (t+h) date

    # AHBS predicted surface of day t+h, using AHBS fitted surface of day t
    test_pred_IV = test_pred_IV.merge(options,on = "Date", how = "inner")
    test_pred_IV['AHBS_IV_t_plus_h'] = test_pred_IV['b0'] + test_pred_IV['b1']*test_pred_IV['M'] + \
        test_pred_IV['b2']*test_pred_IV['M2'] + test_pred_IV['b3']*test_pred_IV['tau'] + \
        test_pred_IV['b4']*test_pred_IV['tau2'] + test_pred_IV['b5']*test_pred_IV['Mtau']
    test_pred_IV = test_pred_IV.rename(columns={"Date": "test_day_ahead_date"})

    # use FCNN (trained on data of day t) to predict residuals 
    test_x = test_pred_IV[['M','tau']].to_numpy()
    test_x1 = torch.tensor(test_x, dtype = torch.float32)

    # 
    model.eval()
    pred_test_y = model(test_x1)
    pred_test_y = pred_test_y.detach().numpy().flatten()

    # final prediction
    test_pred_IV['fcst_IV'] = test_pred_IV['AHBS_IV_t_plus_h'] + pred_test_y

    # remove no longer unimportant columns
    test_pred_IV = test_pred_IV.drop(columns=['b0', 'b1','b2','b3','b4','b5','IV_atm','Year','AHBS_IV_t_plus_h',
                                                'min_K_F_dist','min_K_F_ratio','M2','Mtau','tau2','ln_IV'])
    
    return test_pred_IV


# %%
all_test_preds = pd.DataFrame()
for i in range(n_test): # 
    test_pred = pred_test_day(i)
    
    # test pred for day i
    all_test_preds = pd.concat([all_test_preds,test_pred])
    
    if i % 50 == 0:
        Path(odir + option_type + "_finished_" + str(i) + "_test_days").touch()


# %%    

######
# since Random walk model uses fewer lags than HAR-based model, we need to 
# get the correct starting test date to make fair comparison with other models
actual_starting_test_date = pd.read_csv(wdir + "S03_VARp/steps" + str(n_steps_ahead) + "ahead/" + \
    option_type + "_pred_test_IV.csv")
actual_starting_test_date = actual_starting_test_date.test_date[0]

all_test_preds = all_test_preds.loc[all_test_preds['test_date'] >= actual_starting_test_date]


######
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



# %%
