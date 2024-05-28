# fNTK

## Sample data
Folder: sample_data 
* Refer to sub-folder downloading_cleaning_codes for codes to download options data from Wharton Research Data Services (WRDS) and to clean/filter raw data
* SPX.data contains a sample of (clean) SPX data, for both put and call options, between 01 Apr 2020 and 15 Apr 2020. All models are performed for put and call options separately.


## Simulation studies
Folder: simulation
* Two sub-folders: linear and nonlinear, for linear and nonlinear setup, respectively
* In each sub-folder:
  + Run S01_sim_data.r to simulate IVS data


## Nonlinear functional autoregressive (NFAR) models
Folder: NFAR_models
* Run S01_orthogonalsplinebasis.r to obtain basis coefficients of IVS by projecting on orthogonal splines (Redd 2012)
* Utilize different kernels to perform lead-lag regression on the basis coefficients
  + Run S02_FAR_NTK.py for NTK parameterized neural networks, which is equivelent to kernel regression using NTK (Jacot 2018)
  + Run S03_FAR_KRR.py for parametric kernels, e.g., RBF, Laplacian, or Linear kernel
  + Run S04_FAR_random_walk.py for random walk, i.e., use basis coefficients of day $t$ as predictions of basis coefficients of day $t+h$ in $h$-step ahead forecasting


## Other alternative models
### Ad-hoc Black-Scholes (AHBS) models
Folder: AHBS_models
* Run S01_processing_data.r to prepare data to be in correct format 
* Run S02_DeterministicLinearModel.ipynb for the deterministic linear regression for each day, i.e., to obtain the coefficients using the set of (polynomial) basis functions $(1, m, \tau, m^2, \tau^2, m \tau)$
* Run S03_VARp.py for AHBS with vector autoregressive (VAR) as predicting model, proposed by Bernales 2014
* Run S04_RandomWalk.py for AHBS with random walk (similar to S04_FAR_random_walk.py in NFAR_models, but using the AHBS coefficients instead)
* Run S05_DNN_on_AHBSresiduals.py for AHBS with DNN correction, proposed by Almeida 2022


### Carr and Wu (CW) models
Folder: CarrWu_models \
Following the CW models proposed by Almeida 2022, we have:
* S01_CarrWu_pred.m for CW with random walk
* S02_DNN_CarrWu for codes for CW with DNN correction


## Trading strategies
Folder: trading_strategies
* First, extract trading signals with S02_trading_signals.r code: for each model, buying or selling an option depends on whether the model predicts the IV (of the same option) to increase or decrease on day $t+h$, compared to the IV on day $t$
* After extracting the trading signals for each option, use S03_*.r codes to perform different trading strategies, such as delta-hedging (DH) short call, or delta-neutral short straddle
* The mean (simple) returns and Sharpe ratio of each trading strategy are computed in codes S04_*.r
* To see whether the Sharpe ratios of different models are statistically significantly different, use codes in folder SharpeRatio_test


## References
1. Almeida, C., J. Fan, G. Freire, and F. Tang (2022). “Can a Machine Correct Option Pricing Models?” Journal of business & economic statistics: a publication of the American Statistical Association, 1–14.
2. Bernales, A. and M. Guidolin (2014). “Can we forecast the implied volatility surface dynamics of equity options? Predictability and economic value tests”. Journal of Banking & Finance, 46, 326–342.
3. Jacot, A., F. Gabriel, and C. Hongler (2018). “Neural tangent kernel: Convergence and generalization in neural networks”. Advances in neural information processing systems, 31.
4. Redd, A. (2012). “A comment on the orthogonalization of B-spline basis functions and their derivatives”. Statistics and Computing, 22.1, 251–257.
