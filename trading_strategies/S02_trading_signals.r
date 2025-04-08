library(tidyverse)
library(lubridate)
library(ggplot2)
theme_set(theme_bw())

args = commandArgs(trailingOnly = TRUE) 
print(args)

steps_ahead = args[1]

##### original option data ##### 
options = read_csv('/hpctmp/e0823043/SPX/DailyOptions/S03_moneyness/SPX_2009_2021.csv')
options = options %>% 
  filter(Moneyness_M>=-2, Moneyness_M<=2,
         Maturity>=5, Maturity<=252)

# all dates 
options = options %>% 
  rename(test_date = Date) %>% 
  mutate(test_date = as.Date(as.character(test_date),format = "%Y%m%d"),
         Exp_date = as.Date(as.character(Exp_date),format = "%Y%m%d"))

##### deltas and option ID ##### 
deltas = read_csv("/hpctmp/e0823043/SPX/DailyOptions/SPX_delta_2019_2021.csv")

options = inner_join(options,deltas %>% rename(test_date = Date))

options = options %>% 
  select(test_date, PC_flag, Option_ID, Strike, Maturity, ForwardPrice, Bid, Ask,
         Moneyness_M, delta, IV, IR, DividendYield, Volume) %>% 
  rename(delta_t = delta,
         Bid_t = Bid, Ask_t = Ask,
         IV_t = IV, 
         ForwardPrice_t = ForwardPrice,
         Moneyness_M_t = Moneyness_M,
         Maturity_t = Maturity,
         IR_t = IR,
         Volume_t = Volume,
         DividendYield_t = DividendYield)

# take absolute values of delta
options = options %>% 
  mutate(delta_t = abs(delta_t))


##### S&P 500 prices ##### 
spots = read_csv("/hpctmp/e0823043/SPX/DailyOptions/SPX_SecurityPrice_2000_2022.csv")
spots = spots %>% 
  select(date,low,high,close) %>% 
  rename(Date = date, Spot_low = low, Spot_high = high, Spot_close = close)

# merge to get low (bid) and high (ask) values of Spot on day t
options = left_join(options, 
                    spots %>% 
                      rename(test_date = Date)) %>% 
  rename(Spot_low_t = Spot_low, Spot_high_t = Spot_high, Spot_close_t = Spot_close)


##### interest rate data ##### 
IR = read_csv("/hpctmp/e0823043/SPX/DailyOptions/interest_rate.csv")
IR = IR %>% 
  mutate(Date = as.Date(as.character(Date),format = "%Y%m%d"),
         IR = IR/100) %>% 
  rename(test_date = Date)

##### predicted data ##### 
# functional models
functional_dir = "/hpctmp/e0823043/SPX/S04_FAR/OrthogonalBsplines/"
functional_res = tibble(model = c("fRW","fLinK","fLapK","fGauK","fNTK3")) %>% 
  mutate(dir = c(paste0(functional_dir,"S04_FAR_random_walk/steps",steps_ahead,"ahead/"),
                 paste0(functional_dir,"S03_FAR_KRR/linear/steps",steps_ahead,"ahead/"),
                 paste0(functional_dir,"S03_FAR_KRR/laplacian/steps",steps_ahead,"ahead/"),
                 paste0(functional_dir,"S03_FAR_KRR/rbf/steps",steps_ahead,"ahead/"),
                 paste0(functional_dir,"S02_FAR_NTK/degree3_tau_NaN_M_NaN/steps",steps_ahead,"ahead/HiddenLayers3/"))
         )


# Carr & Wu models
carrwu_dir = "/hpctmp/e0823043/SPX/S03_CarrWu_2016/"
CW_res = tibble(model = c("CW_RW","CW_DNN")) %>% 
  mutate(dir = c(paste0(carrwu_dir,"S01_CarrWu_pred/steps",steps_ahead,"ahead/"),
                 paste0(carrwu_dir,"S02_DNN_on_CarrWu_residuals/S02_DNN_on_CWresiduals/steps",steps_ahead,"ahead/"))
  )


# AHBS models
AHBS_dir = "/hpctmp/e0823043/SPX/S02_Bernales2014/Daily_2009_2021/M_Andersen/"
AHBS_res = tibble(model = c("AHBS","AHBS_RW","AHBS_DNN","AHBS_LSTM_interp")) %>% 
  mutate(dir = c(paste0(AHBS_dir,"S03_VARp/steps",steps_ahead,"ahead/"),
                 paste0(AHBS_dir,"S04_RandomWalk/steps",steps_ahead,"ahead/"),
                 paste0(AHBS_dir,"S05_DNN_on_AHBSresiduals/steps",steps_ahead,"ahead/"),
                 paste0(AHBS_dir,"S06_Zhang2022/S01_LSTM_interpIVs/steps",steps_ahead,"ahead/"))
  )

# dimension reduction models
dimred_dir = "/hpctmp/e0823043/SPX/S06_PCA_vs_Autoencoders/"
dimred_res = tibble(model = c("splines_PCA_LSTM", "splines_AE_LSTM")) %>% 
  mutate(dir = c(paste0(dimred_dir,"S01_PCA_LSTM_predict_actual_IV/PCA_thrsh0.99/steps",steps_ahead,"ahead/"),
                 paste0(dimred_dir,"S02_Autoencoder_LSTM_predict_actual_IV/AEhiddfeats8/steps",steps_ahead,"ahead/"))
  )

# other models
others_dir = "/hpctmp/e0823043/SPX/S05_other_models/"
others_res = tibble(model = c("DNN_RW", "splines_DNN_interp", "splines_LSTM_interp")) %>% 
  mutate(dir = c(paste0(others_dir,"S01_pureFCNN_on_IV_tau_m/steps",steps_ahead,"ahead/"),
                 paste0(others_dir,"S02_FCNN_on_interpolatedIVs_predict_actual_IV/steps",steps_ahead,"ahead/"),
                 paste0(others_dir,"S03_LSTM_on_interpolatedIVs_predict_actual_IV/steps",steps_ahead,"ahead/"))
  )


files = bind_rows(functional_res, CW_res) %>% 
  bind_rows(AHBS_res) %>%
  bind_rows(dimred_res) %>% 
  bind_rows(others_res) 


# output directory
odir = "/hpctmp/e0823043/SPX/S04_FAR/trading_strategies/S02_trading_signals/"
dir.create(file.path(odir), recursive = TRUE, showWarnings = FALSE)

# interest rate for steps_ahead day maturity
IR_for_s_days = IR %>% 
  group_by(test_date) %>% 
  summarise(RF_rate_for_s_days = approx(x = Maturity, y = IR, xout = steps_ahead, rule = 2)$y) 


net_gains = tibble()
all_dates_with_no_trades = tibble()
for (f in 1:nrow(files)) {
  # f = 5
  
  ### read in prediction data
  # test_date is date of test day, test_day_ahead_date is the day we want to actually predict IV for
  # given that we are on test_date day
  call_dat = read_csv(paste0(files$dir[f],"Call_pred_test_IV.csv"))
  call_dat = call_dat %>% 
    mutate(Exp_date = as.Date(as.character(Exp_date),format = "%Y%m%d")) %>%
    inner_join(deltas %>% rename(test_day_ahead_date = Date)) %>% 
    select(test_date,test_day_ahead_date,Option_ID,IV,fcst_IV,Maturity,Bid,Ask) %>%
    rename(IV_th = IV,
           fcst_IV_th = fcst_IV,
           Maturity_th = Maturity,
           Bid_th = Bid, Ask_th = Ask)
  
  put_dat = read_csv(paste0(files$dir[f],"Put_pred_test_IV.csv"))
  put_dat = put_dat %>% 
    mutate(Exp_date = as.Date(as.character(Exp_date),format = "%Y%m%d")) %>%
    inner_join(deltas %>% rename(test_day_ahead_date = Date)) %>% 
    select(test_date,test_day_ahead_date,Option_ID,IV,fcst_IV,Maturity,Bid,Ask) %>% 
    rename(IV_th = IV,
           fcst_IV_th = fcst_IV,
           Maturity_th = Maturity,
           Bid_th = Bid, Ask_th = Ask)
  
  dat = bind_rows(call_dat,put_dat) 
  
  # merge to get low (bid) and high (ask) values of Spot on day t+h
  dat = left_join(dat, 
                  spots %>% 
                    rename(test_day_ahead_date = Date)) %>% 
    rename(Spot_low_th = Spot_low, Spot_high_th = Spot_high, Spot_close_th = Spot_close)
  
  # check if the test_date column is in date format: if not, convert to date
  if (! is.Date(dat$test_date)) {
    dat = dat %>% mutate(test_date = as.Date(as.character(test_date),format = "%Y%m%d"))
  }
  
  
  dat1 = inner_join(dat,options) %>% 
    arrange(test_date) 
  
  # compute the difference btw forecasted premium on day (t+h) and premium on day t
  Q = dat1 %>% 
    mutate(difference = fcst_IV_th/IV_t - 1)
  
  ##### compute V_t = cost of the portfolio on day t
  # set of call options to purchase
  Q_call_plus = Q %>% 
    filter(PC_flag == 1, difference > 0) %>% 
    mutate(signal = "long_call")
  
  # set of call options to sell
  Q_call_minus = Q %>% 
    filter(PC_flag == 1, difference < 0) %>% 
    mutate(signal = "short_call")
  
  # set of put options to purchase
  Q_put_plus = Q %>% 
    filter(PC_flag == -1, difference > 0) %>% 
    mutate(signal = "long_put")
  
  # set of put options to sell
  Q_put_minus = Q %>% 
    filter(PC_flag == -1, difference < 0) %>% 
    mutate(signal = "short_put")
  
  Q = bind_rows(Q_call_plus,Q_call_minus) %>% 
    bind_rows(Q_put_plus) %>% 
    bind_rows(Q_put_minus)
  
  # combine to get risk-free interest rates for steps_ahead days
  Q = inner_join(Q,IR_for_s_days)
  
  Q = Q %>% 
    mutate(model = files$model[f]) %>% 
    select(-test_day_ahead_date,-Option_ID,-Maturity_th,
           -ForwardPrice_t,
           -IR_t,-DividendYield_t,
           -IV_th,-IV_t,-fcst_IV_th,-PC_flag)
  
  # for days without selling or purchasing signals, 
  # invest $1000 in risk-free interest rate for h days
  all_dates = unique(dat1$test_date) # all dates originally in the data set
  dates_with_no_trades = all_dates[!(all_dates %in% Q$test_date)]
  if (length(dates_with_no_trades) > 0) {
    dates_with_no_trades = tibble(test_date = dates_with_no_trades) %>% 
      inner_join(IR_for_s_days) %>% 
      mutate(model = files$model[f],
             signal = "no_trading_signals") 
    all_dates_with_no_trades = bind_rows(all_dates_with_no_trades,dates_with_no_trades)
  }
  
  net_gains = bind_rows(net_gains,Q)
  
  
}

short_calls = net_gains %>% filter(signal == "short_call") %>% select(-signal)
short_puts = net_gains %>% filter(signal == "short_put") %>% select(-signal)
long_calls = net_gains %>% filter(signal == "long_call") %>% select(-signal)
long_puts = net_gains %>% filter(signal == "long_put") %>% select(-signal)

write_csv(short_calls,paste0(odir,"steps",as.character(steps_ahead),"ahead_short_calls.csv"))
write_csv(short_puts,paste0(odir,"steps",as.character(steps_ahead),"ahead_short_puts.csv"))
write_csv(long_calls,paste0(odir,"steps",as.character(steps_ahead),"ahead_long_calls.csv"))
write_csv(long_puts,paste0(odir,"steps",as.character(steps_ahead),"ahead_long_puts.csv"))

if (nrow(all_dates_with_no_trades)>0) {
  write_csv(all_dates_with_no_trades,paste0(odir,"steps",as.character(steps_ahead),"ahead_dates_with_no_trades.csv"))
}
