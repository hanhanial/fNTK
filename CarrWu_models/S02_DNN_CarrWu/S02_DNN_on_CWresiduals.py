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
bspline_type = sys.argv[3] # "degree3_tau_NaN_M_NaN" # 

print('Option type: ',option_type,', number of steps ahead: ',n_steps_ahead)

learning_rate = 0.001
weight_decay = 0.001
EPOCH = 500

# lags to be used for prediction
lags = [1] # only use lag 1 for random walk

# working directory
wdir = "/hpctmp/e0823043/SPX/S03_CarrWu_2016/S02_DNN_on_CarrWu_residuals/"

# output directory
odir = wdir + "S02_DNN_on_CWresiduals/steps" + str(n_steps_ahead) + "ahead/"
os.makedirs(odir,exist_ok=True)  


# %% Load in and process data
wdir1 = "/hpctmp/e0823043/SPX/S02_Bernales2014/Daily_2009_2021/M_Andersen/"
options = pd.read_csv(wdir1 + 'S01_processing_data/' + option_type + "_2009_2021.csv")
options.Date = pd.to_datetime(options['Date'], format='%Y-%m-%d')

# estimated params of Carr Wu model for each day
dat = pd.read_csv(wdir + 'S01_CarrWu_estimated_params/' + option_type + "_CW_est_params.csv")
dat.Date = pd.to_datetime(dat['Date'], format='%Y-%m-%d')


# %%
dates = {'Date': dat.Date.values,'Day' : np.arange(dat.shape[0])+1}
dates = pd.DataFrame.from_dict(dates)
dates.head()

# set index for the original data
dat = dat.set_index(dat.Date)


# %%
# function to compute average of n_lag lags if take_avg=True, otherwise just return all n_lag lagged days
def prepare_data(dates,n_steps_ahead):
    # lags = number of lags to be used as inputs for prediction
    # n_steps_ahead = number of steps (days) ahead to be predicted for
    
    # maximum number of lags to be used as input 
    max_lag = 1 #max(lags)
    # NOTE: in the Almeida 2022 setup, max_lag = 1 always, since they dont use HAR lags
  
    # total number of days
    num_obs = dates.shape[0]
    
    # we need to get the corresponding dates
    # since only data of days with index from max_lag (in python index is (max_lag-1)) 
    # to num_obs (in python index is (num_obs-1))  will have enough lagged input, 
    # and since we need to predict n_steps_ahead days ahead for prediction, 
    # so the latest day we can predict for is (num_obs-1-n_steps_ahead) 
    # but in the slicing index, the last index is not included, so we need to put 
    # (num_obs-1-n_steps_ahead+1) = (num_obs-n_steps_ahead)
    dates1 = dates.iloc[(max_lag-1):(num_obs-n_steps_ahead)]
    dates1['Day'] = range(1,dates1.shape[0]+1)

    # we also have the date we want to predict for (output date)
    dates1['predicting_Date'] = dates['Date'][(max_lag-1+n_steps_ahead):(num_obs-n_steps_ahead+n_steps_ahead)].values
        
    return dates1


# %%
split_dates = prepare_data(dates = dates,n_steps_ahead = n_steps_ahead)

# actual number of days to be used for training and testing
n_days = split_dates.shape[0]

# number of days to be used for training 
# NOTE: this will not really be used, it's just to make it consistent with other models
n_train = 2500

# number of days to be used for testing
n_test = n_days - n_train


# %% 
# function to compute IV, given Carr Wu estimated params
def CW_ImpVol(Strike_tau, v_t, m_t, w_t, n_t, rho_t, Spot_t):
    K_vec = Strike_tau.Strike.to_list()
    T_vec = Strike_tau.tau.to_list()

    CW_ivol_vec = np.zeros(len(K_vec))

    for i in range(len(K_vec)):
        K = K_vec[i]
        T = T_vec[i]
        k = np.log(K/Spot_t)

        p = [(1/4) * np.exp(- 2 * n_t * T) * (w_t**2) * (T**2),
             (1 - 2 * np.exp(- n_t * T) * m_t * T - np.exp(- n_t * T) * w_t * rho_t * np.sqrt(v_t) * T),
             - (v_t + 2 * np.exp(-n_t * T) * w_t * rho_t * np.sqrt(v_t) * k + np.exp(- 2 * n_t * T) * (w_t**2) * (k**2))]

        r = np.roots(p)
        CW_ivol_vec[i] = np.sqrt(r[r > 0])

    return CW_ivol_vec



# %% Moving window prediction
def pred_test_day(i):
    # i = 1

    test_day_idx = i + n_train
    test_date = split_dates['Date'][test_day_idx]
    test_day_ahead_date = split_dates['predicting_Date'][test_day_idx]


    if (i % 200)==0:
        print("Train and predict for test date: ",test_date)

    # CW parameters estimated using test_date data, i.e., data of day t
    CW_params_t = dat[dat['Date'] ==  test_date]

    # option data on day t
    options_t = options[options['Date'] == test_date]

    # fitted IV on day t (using CW estimated on day t)
    IVS_t = CW_ImpVol(Strike_tau = options_t[['Strike', 'tau']],
                    v_t = CW_params_t.v.item(), 
                    m_t = CW_params_t.m.item(), 
                    w_t = CW_params_t.w.item(),
                    n_t = CW_params_t.n.item(), 
                    rho_t = CW_params_t.rho.item(), 
                    Spot_t = np.unique(options_t['Spot']))

    options_t['CW_fitted_IV'] = IVS_t.tolist()
    options_t['residual'] = options_t.IV - options_t.CW_fitted_IV

    # train data with input (tau,m) and output IV of day t
    train_x = options_t[['M','tau']].to_numpy()
    train_y = options_t.residual.to_numpy()
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
            

    ###### use Carr Wu + FCNN trained on day t to predict for day t+h ######
    # option data on day t
    options_th = options[options['Date'] == test_day_ahead_date]
    test_pred_IV = options_th

    # predicted (fitted) IV of day (t + h) using CW model of day t
    CW_IV_t_plus_h = CW_ImpVol(Strike_tau = options_th[['Strike', 'tau']],
                            v_t = CW_params_t.v.item(), 
                            m_t = CW_params_t.m.item(), 
                            w_t = CW_params_t.w.item(),
                            n_t = CW_params_t.n.item(), 
                            rho_t = CW_params_t.rho.item(), 
                            Spot_t = np.unique(options_th['Spot']))
    test_pred_IV['CW_IV_t_plus_h'] = CW_IV_t_plus_h.tolist()

    test_pred_IV = test_pred_IV.rename(columns={"Date": "test_day_ahead_date"})
    test_pred_IV['test_date'] = test_date

    # use FCNN (trained on data of day t) to predict residuals 
    test_x = test_pred_IV[['M','tau']].to_numpy()
    test_x1 = torch.tensor(test_x, dtype = torch.float32)

    # 
    model.eval()
    pred_test_y = model(test_x1)
    pred_test_y = pred_test_y.detach().numpy().flatten()

    # final prediction
    test_pred_IV['fcst_IV'] = test_pred_IV['CW_IV_t_plus_h'] + pred_test_y

    # remove no longer unimportant columns
    test_pred_IV = test_pred_IV.drop(columns=['IV_atm','Year','CW_IV_t_plus_h',
                                                'min_K_F_dist','min_K_F_ratio','M2','Mtau','ln_IV'])
    
    return test_pred_IV


# %%
all_test_preds = pd.DataFrame()
for i in range(n_test): # n_test 
    test_pred = pred_test_day(i)
    
    # test pred for day i
    all_test_preds = pd.concat([all_test_preds,test_pred])
    
    if i % 50 == 0:
        Path(odir + option_type + "_finished_" + str(i) + "_test_days").touch()


# %%    

######
# since Random walk model uses fewer lags than HAR-based model, we need to 
# get the correct starting test date to make fair comparison with other models
actual_starting_test_date = pd.read_csv(wdir1 + "S03_VARp/steps" + str(n_steps_ahead) + "ahead/" + \
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
