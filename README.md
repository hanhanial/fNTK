# fNTK

## Sample data
Folder: sample_data 
* Refer to sub-folder downloading_cleaning_codes for codes to download options data from Wharton Research Data Services (WRDS) and to clean/filter raw data
* SPX.data contains a sample of (clean) SPX data, for both put and call options, between 01 Apr 2020 and 15 Apr 2020. All models are performed for put and call options separately.
* Run S01_orthogonalsplinebasis.r to obtain basis coefficients of IVS by projecting on orthogonal splines (Redd 2012)

## Simulated data
Folder: simulation
* Two sub-folders: linear and nonlinear, for linear and nonlinear setup, respectively
* In each sub-folder: Run S01_sim_data.r to simulate IVS data. Run S02_* code for random walk (RF), neural tangent kernel (NTK) and classical kernel Ridge regression (KRR) models

### Prediction accuracy for each model
File: compute_pred_accuracy.r


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
