# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
from torchmetrics import MeanSquaredError


# %%
seed_id = sys.argv[1] # "seed_20000" # 
n_steps_ahead = int(sys.argv[2]) # 1 # 
n_layers = int(sys.argv[3]) # 3 # 

# n_trainval should be less than actual_n_trainval
n_trainval = int(sys.argv[4]) # 800 # 

print('Number of steps ahead: ',n_steps_ahead, ', number of layers: ',n_layers)

# working directory
wdir = "/hpctmp/e0823043/SPX/simulation_AHBS/all_params_linear/"

# lags to be used for prediction
lags = [1]

n_neurons = 500

# output directory
odir = wdir + "S02_FAR_NTK/steps" + str(n_steps_ahead) + "ahead/HiddenLayers" + str(n_layers) + "_TrainSize" + str(n_trainval) + "/"
os.makedirs(odir,exist_ok=True)  


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NTK_linear(nn.Module):
    def __init__(self, in_features, out_features, beta = 0.1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.beta = beta
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0.0, std=1.0)
        
    def forward(self, input):
        x, y = input.shape
        
        if y != self.in_features:
            print(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
            return 0
        output = input.matmul(1/np.sqrt(y) * self.weight.t())
        
        if self.bias is not None:
            output += self.beta * self.bias
        
        return output
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# %%
class RMSE:
    def __init__(self):
        super().__init__()
        self.squared_errors = 0
        self.total = 0
    
    def update(self, pred, target):
        se = (target-pred).square().sum().item()
        total = len(target)
        self.squared_errors += se
        self.total += total

        return np.sqrt(se/total)
    
    def compute(self):
        rmse = np.sqrt(self.squared_errors / self.total)
        self.squared_errors = 0
        self.total = 0
        return rmse


# %%
def train_epoch(model, criterion, optimizer, data_loader):
    model.train()
    
    train_losses = []
    rmse = RMSE()
    
    for x, y in data_loader:
        x, y, = x.to(device), y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        preds = model(x)
        
        loss = criterion(preds, y)
        train_losses.append(loss.item())
        
        rmse.update(preds, y)
        
        loss.backward()
        
        optimizer.step()

    return train_losses, rmse.compute() 


# %%
def val_epoch(model, criterion, data_loader):
    model.eval()
    
    val_losses = []
    val_rmse = RMSE()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            
            preds = model(x)
            
            loss = criterion(preds, y)
            val_losses.append(loss.item())
            
            val_rmse.update(preds, y)
      
    return val_losses, val_rmse.compute()


# %%
def fit(model, train_criterion, val_criterion, optimizer, train_loader, val_loader, epochs):
    RMSE_stats = {'train': [], "val": []}
    loss_stats = {'train': [], "val": []}
    best_val_rmse = float('inf')
    
    for epoch in range(1, epochs+1):
        train_losses, train_rmse = train_epoch(model, train_criterion, optimizer, train_loader)
        train_loss = np.mean(train_losses)  # Biased if last batch is not full
        
        val_losses, val_rmse = val_epoch(model, val_criterion, val_loader)
        val_loss = np.mean(val_losses)  # Biased ...
    
        loss_stats['train'].append(train_loss)
        loss_stats['val'].append(val_loss)
        RMSE_stats['train'].append(train_rmse)
        RMSE_stats['val'].append(val_rmse)
    
        if epoch % 100 == 0:
            print((f'Epoch {epoch+0:03}: | '
                   f'Train Loss: {train_loss:.5f} | Train RMSE: {train_rmse:.5f} | '
                   f'Val Loss: {val_loss:.5f} | Val RMSE: {val_rmse:.5f} '))

        if val_rmse < best_val_rmse:
#             print(f'Validation RMSE ({min_val_rmse:.5f}--->{val_rmse:.6f}) \t Saving The Model')
            # save_checkpoint(MODEL_PATH, model, epoch, type='best')
            best_val_rmse = val_rmse
            best_num_epoch = epoch
    # save_checkpoint(MODEL_PATH, model, epoch, type='final')

    return loss_stats, RMSE_stats, best_val_rmse, best_num_epoch


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
print(split_dates.shape)


# %%
# number of outputs (for each day) 
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


# %%
# Function to tune a given NTK model structure (with given epochs, batch size, # neurons, learning rate, random seed)
# to find the best weight decay rate
def tuning_NTK(train_x, train_y, val_x, val_y, 
               n_layers, n_neurons,
               epochs, batch_size, learning_rate, seed, 
               weight_decay_rates):
    
    #### standardize train and val sets ####
    scaler_x = StandardScaler()
    train_x1 = scaler_x.fit_transform(train_x)
    val_x1 = scaler_x.transform(val_x)

    scaler_y = StandardScaler()
    train_y1 = scaler_y.fit_transform(train_y)
    val_y1 = scaler_y.transform(val_y)

    #### prepare data for the model ####
    X_train = torch.tensor(train_x1,dtype=torch.float32)
    Y_train = torch.tensor(train_y1,dtype=torch.float32)

    X_val = torch.tensor(val_x1,dtype=torch.float32)
    Y_val = torch.tensor(val_y1,dtype=torch.float32)

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size,shuffle = True, drop_last = False)
    val_dataset = TensorDataset(X_val, Y_val)
    val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, drop_last = False)

    #### Tune NTK model ####
    
    min_val_rmse = float('inf')
    for j in range(len(weight_decay_rates)): 
        print("Running NTK with weight decay = %f " % weight_decay_rates[j])
        torch.manual_seed(seed)

        # Create NTK model with # hidden layers = n_layers and there are n_neurons neurons in each hidden layer
        layers = [NTK_linear(in_features = n_in, out_features = n_neurons), nn.ReLU()] 
        for i in range(n_layers-1):
            layers.append(NTK_linear(in_features = n_neurons, out_features = n_neurons))
            layers.append(nn.ReLU())
        layers.append(NTK_linear(in_features = n_neurons, out_features = n_out))
        model = nn.Sequential(*layers)
        
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr = learning_rate,
                                weight_decay = weight_decay_rates[j])

        loss_stats, rmse_stats, val_rmse, num_epoch = fit(model = model, 
                                                          train_criterion = criterion, val_criterion = criterion,
                                                          optimizer = optimizer,
                                                          train_loader = train_loader, val_loader = val_loader, 
                                                          epochs = epochs)

        model.eval()

        #### Out-sample predictive errors ####
        Y_val_preds = model(X_val) # Making Predictions
        
        torch_RMSE = MeanSquaredError(squared = False)
        val_rmse = torch_RMSE(preds = Y_val_preds, target = Y_val)
        if val_rmse < min_val_rmse:
            min_val_rmse = val_rmse
            best_weight_decay_rate = weight_decay_rates[j]
            
            print("Current best weight decay is %f " % best_weight_decay_rate)
            
            # this is the epoch where validation RMSE is smallest before the final number of epoch (EPOCHS) is reached 
            # just for recording purpose, will not use it for training on whole train data
            selected_num_epochs = num_epoch 

    return pd.DataFrame({'weight_decay_rate': [best_weight_decay_rate],
                         'num_epochs' : [selected_num_epochs]})


# %%
EPOCHS = 500 
BATCH_SIZE = n_trainval
SEED = 2468
learning_rate = 0.05


# %%
#### Hyperparameter tuning ####
selected_hyperparams = tuning_NTK(train_x = X[(actual_n_trainval-n_trainval):(actual_n_trainval-n_val),:], 
                                  train_y = Y[(actual_n_trainval-n_trainval):(actual_n_trainval-n_val),:], 
                                  val_x = X[(actual_n_trainval-n_val):actual_n_trainval,:], 
                                  val_y = Y[(actual_n_trainval-n_val):actual_n_trainval,:], 
                                  n_layers = n_layers, n_neurons = n_neurons,
                                  epochs = 200, batch_size = n_train, 
                                  learning_rate = learning_rate, seed = SEED,
                                  weight_decay_rates = [1e-3] # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] # 
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
#### prepare data for the model ####
X_train = torch.tensor(train_x1,dtype=torch.float32)
Y_train = torch.tensor(train_y1,dtype=torch.float32)

X_test = torch.tensor(test_x1,dtype=torch.float32)
Y_test = torch.tensor(test_y1,dtype=torch.float32)

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE,shuffle = True, drop_last = False)
test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, drop_last = False)


# %%
#### Train model ####
torch.manual_seed(SEED)

# Create model with # hidden layers = n_layers and there are n_neurons neurons in each hidden layer
layers = [NTK_linear(in_features = n_in, out_features = n_neurons),
          nn.ReLU()]
for i in range(n_layers-1):
    layers.append(NTK_linear(in_features = n_neurons, out_features = n_neurons))
    layers.append(nn.ReLU())
layers.append(NTK_linear(in_features = n_neurons, out_features = n_out))
model = nn.Sequential(*layers)

model.to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(),
                        lr = learning_rate,
                        weight_decay = float(selected_hyperparams['weight_decay_rate'].values))

loss_stats, rmse_stats, val_rmse, num_epoch = fit(model = model, 
                                                  train_criterion = criterion, val_criterion = criterion,
                                                  optimizer = optimizer,
                                                  train_loader = train_loader, val_loader = test_loader, 
                                                  epochs = EPOCHS)


# I%%
model.eval()

#### Out-sample predictive errors ####
Y_test_preds = model(X_test) # Making Predictions
Y_test_preds1 = Y_test_preds.detach().numpy() # convert tensor to array
Y_test_preds2 = scaler_x.inverse_transform(Y_test_preds1) # inverse scaler


# %%
# all dates to be predicted for (i.e. date of day t+h)
all_test_dates_i = dates.loc[dates['Day'].isin(test_date.Day.values + n_steps_ahead),:]


# %%
# merge predicted basis coefficients and basis values to make prediction for IV
test_pred_IV = pd.DataFrame(Y_test_preds2,columns = ["basis_" + str(i) for i in range(1,n_out+1)])
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


# %%
all_accuracy = pd.concat([overall_accuracy])
print("Test IV accuracy (at interpolated points)")
print(all_accuracy)


# %%
all_accuracy.to_csv(odir + seed_id + "_pred_test_IV_accuracy.csv",index=False)
selected_hyperparams.to_csv(odir  + seed_id + "_best_hyperparams.csv",index=False)

# %%



