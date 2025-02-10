# %%
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

from jax import jit
import neural_tangents as nt
from neural_tangents import stax


# %%

for obsplines_type in ["degree3_tau_NaN_M_NaN"]:
    for option_type in ["Call", "Put"]:
        for n_steps_ahead in [1,5,10,15,20]:
            for n_layers in [3]:
                # obsplines_type = "degree3_tau_NaN_M_NaN" #sys.argv[1] # 
                # option_type = "Call" # sys.argv[2] # 
                # n_steps_ahead = 1 # int(sys.argv[3]) # 
                # n_layers = 3 # int(sys.argv[4]) # 

                print('Option type: ',option_type,', number of steps ahead: ',\
                    n_steps_ahead,', number of layers: ',n_layers)

                # working directory
                wdir = "/Users/hannahlai/Desktop/Hannah/work/SPX/Analysis/S04_FAR/OrthogonalBsplines/"

                # lags to be used for prediction
                lags = [1, 5, 22]

                n_neurons = 500

                # output directory
                odir = wdir + "S05_FAR_NTKR/" + obsplines_type + "/steps" + \
                    str(n_steps_ahead) + "ahead/HiddenLayers" + str(n_layers) + "/"
                os.makedirs(odir,exist_ok=True)  


                ## %%
                # read in orthogonal bsplines coefficients
                dat = pd.read_csv(wdir + "S01_OrthogonalBsplines/" + obsplines_type + "/" + option_type + "/obspline_coeffss.csv")
                dat.Date = pd.to_datetime(dat['Date'], format='%Y%m%d')
                dat.columns = [c.replace("basis", "coef") for c in list(dat.columns)]

                dates = dat[['Date']]

                dat = dat.set_index('Date')

                # option data as well as bsplines values at observed (tau,M) in each day
                options = pd.read_csv(wdir + "S01_OrthogonalBsplines/" + obsplines_type + "/" + option_type + "/obspline_values.csv")
                options.Date = pd.to_datetime(options['Date'], format='%Y%m%d')


                ## %%
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


                ## %%
                split_dates, X, Y = prepare_data(data = dat,
                                                dates = dates,
                                                lags = lags,
                                                n_steps_ahead = n_steps_ahead,
                                                take_avg = True)

                X = np.array(X)
                Y = np.array(Y)


                ## %%
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


                ## %%
                # Function to tune a given NTK model structure (with given epochs, batch size, # neurons, learning rate, random seed)
                # to find the best weight decay rate
                def tuning_NTK(train_x, train_y, val_x, val_y, 
                                n_layers, n_neurons, learning_rate, 
                                lambda_rates):
                    
                    #### standardize train and val sets ####
                    scaler_x = StandardScaler()
                    train_x1 = scaler_x.fit_transform(train_x)
                    val_x1 = scaler_x.transform(val_x)

                    scaler_y = StandardScaler()
                    train_y1 = scaler_y.fit_transform(train_y)
                    val_y1 = scaler_y.transform(val_y)

                    #### Tune NTK model ####
                    
                    min_val_rmse = float('inf')
                    for j in range(len(lambda_rates)): 
                        print("Running NTK kernel regression with Ridge lambda = %f " % lambda_rates[j])

                        # Fully connected NN
                        if (n_layers == 1):
                            init_fn, apply_fn, kernel_fn = stax.serial(
                                    stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                    stax.Dense(out_dim = train_y1.shape[1], W_std=1, b_std=1)
                            )       
                        elif (n_layers == 3):
                            init_fn, apply_fn, kernel_fn = stax.serial(
                                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                stax.Dense(out_dim = train_y1.shape[1], W_std=1, b_std=1)
                            )
                        else:
                            init_fn, apply_fn, kernel_fn = stax.serial(
                                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                stax.Dense(out_dim = train_y1.shape[1], W_std=1, b_std=1)
                            )

                        # apply_fn = jit(apply_fn)
                        # kernel_fn = jit(kernel_fn, static_argnames='get')

                        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, 
                                                                            x_train = train_x1, 
                                                                            y_train = train_y1, 
                                                                            learning_rate = learning_rate,
                                                                            
                                                                            # strength of the diagonal regularization for kernel_fn
                                                                            diag_reg = lambda_rates[j]
                                                                            )

                        #### Out-sample predictive errors ####
                        # compute the result of doing gradient descent on our infinite network for an *infinite* amount of time 
                        # predict_fn with "NTK" mode compute the distribution of networks after gradient descent training
                        ntk_mean, ntk_covariance = predict_fn(x_test = val_x1, get = 'ntk', 
                                                            compute_cov = True)
                        Y_val_preds = np.array(ntk_mean) 
                        if np.isnan(Y_val_preds).any():
                            next
                        else:
                            val_rmse = mean_squared_error(y_pred = Y_val_preds, y_true = val_y1)
                            if val_rmse < min_val_rmse:
                                min_val_rmse = val_rmse
                                best_lambda_rate = lambda_rates[j]
                                
                                print("Current best weight decay is %f " % best_lambda_rate)

                    return pd.DataFrame({'lambda_rate': [best_lambda_rate]})


                ## %%
                # This function will predict for n_steps_ahead days: test_day_idx, test_day_idx+1, ..., test_day_idx+n_steps_ahead-1
                # using day (test_day_idx-1) and average of the last k days (test_day_idx-1,...,test_day_idx-k) for each k in lags as input
                def pred_test_day(i,n_neurons,n_layers,learning_rate,tuned_lambda_rate):
                    # i = 0

                    test_day_idx = i + n_trainval
                    test_date = split_dates.iloc[test_day_idx]['Date']
                    test_day_ahead_date = split_dates.iloc[test_day_idx]['predicting_Date']


                    if i % 100 == 0:
                        print("Train and predict for test date: ",test_date)

                    #### Hyperparameter tuning ####
                    # tune for every 120 test days
                    if i % 120 == 0:
                        selected_hyperparams = tuning_NTK(train_x = X[(test_day_idx-n_trainval):(test_day_idx-n_val),:], 
                                                            train_y = Y[(test_day_idx-n_trainval):(test_day_idx-n_val),:], 
                                                            val_x = X[(test_day_idx-n_val):test_day_idx,:], 
                                                            val_y = Y[(test_day_idx-n_val):test_day_idx,:], 
                                                            n_layers = n_layers, n_neurons = n_neurons,
                                                            learning_rate = learning_rate, 
                                                            lambda_rates = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
                                                            ) 
                        selected_hyperparams['test_date'] = test_date
                    else:
                        selected_hyperparams = pd.DataFrame({'lambda_rate': [tuned_lambda_rate]})
                        selected_hyperparams['test_date'] = np.NAN


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
                    # Fully connected NN
                    if (n_layers == 1):
                        init_fn, apply_fn, kernel_fn = stax.serial(
                                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                                stax.Dense(out_dim = train_y1.shape[1], W_std=1, b_std=1)
                        )       
                    elif (n_layers == 3):
                        init_fn, apply_fn, kernel_fn = stax.serial(
                            stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                            stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                            stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                            stax.Dense(out_dim = train_y1.shape[1], W_std=1, b_std=1)
                        )
                    else:
                        init_fn, apply_fn, kernel_fn = stax.serial(
                            stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                            stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                            stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                            stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                            stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                            stax.Dense(out_dim = train_y1.shape[1], W_std=1, b_std=1)
                        )

                    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, 
                                                                        x_train = train_x1, 
                                                                        y_train = train_y1, 
                                                                        learning_rate = learning_rate,
                                                                        
                                                                        # strength of the diagonal regularization for kernel_fn
                                                                        diag_reg = float(selected_hyperparams['lambda_rate'].values)
                                                                    )

                    #### Out-sample predictive errors ####
                    # compute the result of doing gradient descent on our infinite network for an *infinite* amount of time 
                    # predict_fn with "NTK" mode compute the distribution of networks after gradient descent training
                    ntk_mean, ntk_covariance = predict_fn(x_test = test_x1, get = 'ntk',
                                                        compute_cov = True)
                    test_y_pred = np.array(ntk_mean) # Making Predictions
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


                ## %%
                all_test_preds = pd.DataFrame()
                best_hyperparams = pd.DataFrame()

                for i in range(n_test): # n_test
                    if i == 0: 
                        # initialize for i=0 since NTK is not tuned yet at all
                        new_o1, o2 = pred_test_day(i,
                                                n_layers = n_layers,
                                                n_neurons = n_neurons,
                                                learning_rate = 0.01,
                                                tuned_lambda_rate = 0.1)
                    else: 
                        new_o1, o2 = pred_test_day(i,
                                                n_layers = n_layers,
                                                n_neurons = n_neurons,
                                                learning_rate = 0.01,
                                                tuned_lambda_rate = float(o1['lambda_rate'].values))
                    o1 = new_o1 
                    
                    # record best hyperparameter and test result for day i
                    best_hyperparams = pd.concat([best_hyperparams,o1])
                    all_test_preds = pd.concat([all_test_preds,o2])
                    
                    if i % 50 == 0:
                        Path(odir + "finished_" + str(i) + "_test_days").touch()

                # remove all test days that we dont use to tune hyperparameter 
                best_hyperparams = best_hyperparams.dropna()

                ## %%
                # function to compute prediction accuracy for each year
                def compute_pred_accuracy(preds, period):
                    if preds.empty == False:
                        RMSE = root_mean_squared_error(preds.IV, preds.fcst_IV)
                        MAE = mean_absolute_error(preds.IV, preds.fcst_IV)
                        MAPE = mean_absolute_percentage_error(preds.IV, preds.fcst_IV)
                        OoR2 = 1 - np.sum(pow(preds.fcst_IV.values - preds.IV.values,2)) / np.sum(pow(preds.IV.values - np.mean(preds.IV.values),2))
                        
                        accuracy = pd.DataFrame({'period': [period],
                                                'RMSE': [RMSE],
                                                'MAE': [MAE],
                                                'MAPE': [MAPE],
                                                'OoR2': [OoR2]}) 
                    else:
                        accuracy = pd.DataFrame({'period': [],
                                                'RMSE': [],
                                                'MAE': [],
                                                'MAPE': [],
                                                'OoR2': []}) 

                    return accuracy


                ## %%

                overall_accuracy = compute_pred_accuracy(preds = all_test_preds, period = 'overall')


                # list of uniquie years in the prediction period
                all_years = np.unique(all_test_preds['Year'].values)

                # initialize a list to store prediction accuracies across different periods
                all_accuracy = list()

                # store overall pred accuracy too
                all_accuracy.append(overall_accuracy)

                # compute and store pred accuracy for each year
                for year in all_years:
                    tmp = compute_pred_accuracy(preds = all_test_preds[all_test_preds['Year'] == year], 
                                                period = year)
                    all_accuracy.append(tmp)

                all_accuracy = pd.concat(all_accuracy)
                print("Observed test IV accuracy: ")
                print(all_accuracy)

                ## %%
                all_test_preds.to_csv(odir + option_type + "_pred_test_IV.csv",index=False)
                all_accuracy.to_csv(odir + option_type + "_pred_test_IV_accuracy.csv",index=False)
                best_hyperparams.to_csv(odir  + option_type + "_best_hyperparams.csv",index=False)


# %%




