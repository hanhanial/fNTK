# %%
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import sys

from jax import jit
import neural_tangents as nt
from neural_tangents import stax

import glob

# %%
# working directory
wdir = "/Users/hannahlai/Desktop/Hannah/work/SPX/Analysis/simulation_AHBS/all_params_linear/"

all_seed_ids = glob.glob(wdir + "S01_sim_data/*_full_data.csv")
all_seed_ids = [f.replace(wdir + "S01_sim_data/", "") for f in all_seed_ids]
all_seed_ids = [f.replace("_full_data.csv", "") for f in all_seed_ids]

# n_trainval should be less than actual_n_trainval
n_trainval = 800

for seed_id in all_seed_ids:
    for n_layers in [1, 3, 5]: # 
        # seed_id = "seed_20000" # sys.argv[1] # 
        n_steps_ahead = 1 # int(sys.argv[2]) # 
        # n_layers = 3 # int(sys.argv[3]) # 

        # lags to be used for prediction
        lags = [1]

        n_neurons = 500

        # output directory
        odir = wdir + "S03_FAR_NTK_KernelRegression/steps" + str(n_steps_ahead) + "ahead/HiddenLayers" +  str(n_layers) + "_TrainSize" + str(n_trainval) + "/"
        os.makedirs(odir,exist_ok=True)  

        # dates (rows of dates correspond to row of original data)
        dates = pd.read_csv(wdir + "S01_sim_data/" + seed_id + "_date_indexing.csv")
        dates.Date = pd.to_datetime(dates['Date'], format='%Y%m%d')

        dat = pd.read_csv(wdir + "S01_sim_data/" + seed_id + "_obspline_coeffss.csv")
        dat = dat.drop(["Date"],axis = 1)
        # set index for the original data
        dat = dat.set_index(dates.Date)

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


        # number of outputs (for each day) 
        n_out = Y.shape[1]

        # number of inputs used to make prediction at each day
        n_in = X.shape[1]

        # actual number of days to be used for training and testing
        n_days = split_dates.shape[0]

        # number of days to be used for training
        actual_n_trainval = 1600

        n_val = int(np.round(n_trainval*0.2))
        n_train = n_trainval - n_val

        # number of days to be used for testing
        n_test = n_days - actual_n_trainval

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
                elif (n_layers == 2):
                    init_fn, apply_fn, kernel_fn = stax.serial(
                        stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                        stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                        stax.Dense(out_dim = train_y1.shape[1], W_std=1, b_std=1)
                    )
                else:
                    init_fn, apply_fn, kernel_fn = stax.serial(
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
                        best_weight_decay_rate = lambda_rates[j]
                        
                        print("Current best weight decay is %f " % best_weight_decay_rate)

            return pd.DataFrame({'lambda_rate': [best_weight_decay_rate]})

        learning_rate = 0.1

        #### Ridge lambda tuning ####
        """
        train_x = X[0:(n_trainval-n_val),:]
        train_y = Y[0:(n_trainval-n_val),:] 
        val_x = X[(n_trainval-n_val):n_trainval,:]
        val_y = Y[(n_trainval-n_val):n_trainval,:] 
        n_layers = n_layers; n_neurons = n_neurons
        learning_rate = learning_rate
        lambda_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] 
        """
        #### Hyperparameter tuning ####
        selected_hyperparams = tuning_NTK(train_x = X[(actual_n_trainval-n_trainval):(actual_n_trainval-n_val),:], 
                                        train_y = Y[(actual_n_trainval-n_trainval):(actual_n_trainval-n_val),:], 
                                        val_x = X[(actual_n_trainval-n_val):actual_n_trainval,:], 
                                        val_y = Y[(actual_n_trainval-n_val):actual_n_trainval,:], 
                                        n_layers = n_layers, n_neurons = n_neurons, 
                                        learning_rate = learning_rate,
                                        lambda_rates = [1e-4, 1e-3, 1e-2, 1e-1] #
                                        ) 


        #### Training with both train + validation data, and make test prediction ####
        train_dates = split_dates.iloc[(actual_n_trainval-n_trainval):actual_n_trainval]
        test_date = split_dates.iloc[actual_n_trainval:n_days]

        # train set
        train_x = X[(actual_n_trainval-n_trainval):actual_n_trainval,:]
        train_y = Y[(actual_n_trainval-n_trainval):actual_n_trainval,:]

        # test set
        test_x = X[actual_n_trainval:n_days,:]
        test_y = Y[actual_n_trainval:n_days,:]


        #### standardize train and test sets ####
        scaler_x = StandardScaler()
        train_x1 = scaler_x.fit_transform(train_x)
        test_x1 = scaler_x.transform(test_x)

        scaler_y = StandardScaler()
        train_y1 = scaler_y.fit_transform(train_y)
        test_y1 = scaler_y.transform(test_y)


        # Fully connected NN
        if (n_layers == 1):
            init_fn, apply_fn, kernel_fn = stax.serial(
                    stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                    stax.Dense(out_dim = train_y1.shape[1], W_std=1, b_std=1)
            )       
        elif (n_layers == 2):
            init_fn, apply_fn, kernel_fn = stax.serial(
                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                stax.Dense(out_dim = n_neurons, W_std=1, b_std=1), stax.Relu(),
                stax.Dense(out_dim = train_y1.shape[1], W_std=1, b_std=1)
            )
        else:
            init_fn, apply_fn, kernel_fn = stax.serial(
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
                                                            diag_reg = float(selected_hyperparams['lambda_rate'].values)
                                                            )

        #### Out-sample predictive errors ####
        # compute the result of doing gradient descent on our infinite network for an *infinite* amount of time 
        # predict_fn with "NTK" mode compute the distribution of networks after gradient descent training
        ntk_mean, ntk_covariance = predict_fn(x_test = test_x1, get = 'ntk',
                                            compute_cov = True)
        Y_test_preds = np.array(ntk_mean) # Making Predictions
        Y_test_preds1 = scaler_x.inverse_transform(Y_test_preds) # inverse scaler


        # all dates to be predicted for (i.e. date of day t+h)
        all_test_dates_i = dates.loc[dates['Day'].isin(test_date.Day.values + n_steps_ahead),:]


        # merge predicted basis coefficients and basis values to make prediction for IV
        test_pred_IV = pd.DataFrame(Y_test_preds1,columns = ["basis_" + str(i) for i in range(1,n_out+1)])
        test_pred_IV['test_date'] = test_date.Date.dt.strftime("%Y%m%d").values
        test_pred_IV['Date'] = all_test_dates_i.Date.values
        test_pred_IV = test_pred_IV.rename(columns={"Date": "test_day_ahead_date"})


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

            all_pred_IVs = all_pred_IVs._append(pred_IVs) 


        # "observed" IV values
        options = pd.read_csv(wdir + "S01_sim_data/" + seed_id + "_full_data.csv")
        options['Date'] = pd.to_datetime(options['Date'], format='%Y%m%d')

        # inner join to get pred IV and observed IV of test dates
        options = pd.merge(options, all_pred_IVs)

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
        selected_hyperparams.to_csv(odir  + seed_id + "_best_hyperparams.csv",index=False)


# %%



