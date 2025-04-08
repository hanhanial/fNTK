library(tidyverse)
library(ggplot2)

##### original option data #####
# Use this to get total available number of option pairs
options = read_csv('~/Desktop/Hannah/work/SPX/Analysis/S01_data_filtering/S03_moneyness/SPX_2009_2021.csv')
options = options %>%
  filter(PC_flag == -1, # keep only put options
         Maturity >= 5, Maturity <= 252, 
         abs(Moneyness_M) <= 2,
         Volume > 0)

options1 = options %>%
  select(Date,Exp_date,PC_flag,Strike,Spot,Maturity,Moneyness_M,Ask,Bid,Volume)
options1 = options1 %>% 
  arrange(Date,Exp_date,Strike,PC_flag)


###### Overall summary ######
# count number of options with maturity of at least steps_ahead (only those options will be available for trading
# in steps_ahead forecasting)
overall_num_avai_ops = list()
for (s in c(1,5,10,15,20)) {
  overall_num_avai_ops[[paste0("total_num_ops_for",as.character(s),"steps_ahead")]] = options1 %>% 
    filter(Maturity >= s) %>% 
    group_by(Date) %>% 
    summarise(total_num_ops = n()) %>% # number of options 
    ungroup() %>% 
    mutate(Date = as.Date(as.character(Date),format = "%Y%m%d"),
           steps_ahead = s) %>%
    rename(test_date = Date)
}
overall_num_avai_ops = Reduce(x = overall_num_avai_ops,f = bind_rows)


filtering_thrsh = 0.005 # threshold for filtering trading signals
effective_spread_measure =  0 # trading cost


###### Short Put DH ######
short_put = read_csv(paste0("S03_trading_returns/ShortPutDH_overall_thrs",
                            as.character(filtering_thrsh),"_EffSpread",
                            as.character(effective_spread_measure),".csv"))

unique(short_put$model)
sorted_models = c("DNN_RW",
                  "CW_RW", "CW_DNN",
                  "AHBS_RW", "AHBS_DNN", "AHBS",
                  "fRW", "fLinK", "fGauK", "fLapK", "fNTK3",
                  "AHBS_LSTM_interp",
                  "splines_DNN_interp", "splines_LSTM_interp",
                  "splines_PCA_LSTM","splines_AE_LSTM")

short_put = short_put %>% 
  mutate(model = factor(x = model,
                        levels = sorted_models))
short_put = short_put %>% 
  inner_join(overall_num_avai_ops) %>% 
  mutate(perc_traded_ops = num_options/total_num_ops*100)


short_put_overall = short_put %>% 
  # mean of simple return (R_t) and Sharpe ratio SR_t
  group_by(steps_ahead,model) %>% 
  summarise(mean_perc_traded_ops = mean(perc_traded_ops),
            mean_return = mean(R_t)*100,
            std_return = sqrt(var(R_t)),
            sharpe_ratio = mean(ER_t)/sqrt(var(ER_t)),
            mean_net_gain = mean(G_t)) %>% 
  ungroup() %>% 
  filter(! model %in% c("fNTK1","fNTK5")) %>% 
  mutate(sharpe_ratio = sharpe_ratio*sqrt(252/steps_ahead)) %>%  # annualized Sharpe ratio
  gather(key = "Type",value = "Value",-steps_ahead,-model) %>% 
  mutate(steps_ahead = paste0("$h = ",as.character(steps_ahead),"$"),
         Type = factor(Type,levels = c("mean_perc_traded_ops","mean_net_gain","mean_return","std_return","sharpe_ratio"))) %>% 
  spread(key = "steps_ahead",value = "Value") %>% 
  arrange(Type,model) %>% 
  select(Type,model,`$h = 1$`,`$h = 5$`,`$h = 10$`,`$h = 15$`,`$h = 20$`) %>% 
  mutate(across(where(is.numeric), round, 2)) %>% 
  filter(Type %in% c("mean_return","sharpe_ratio"))


###### Long Put DH ######
long_put = read_csv(paste0("S03_trading_returns/LongPutDH_overall_thrs",
                           as.character(filtering_thrsh),"_EffSpread",
                           as.character(effective_spread_measure),".csv"))
long_put = long_put %>% 
  mutate(model = factor(x = model,
                        levels = sorted_models))
long_put = long_put %>% 
  inner_join(overall_num_avai_ops) %>% 
  mutate(perc_traded_ops = num_options/total_num_ops*100)


long_put_overall = long_put %>% 
  # mean of simple return (R_t) and Sharpe ratio SR_t
  group_by(steps_ahead,model) %>% 
  summarise(mean_perc_traded_ops = mean(perc_traded_ops),
            mean_return = mean(R_t)*100,
            std_return = sqrt(var(R_t)),
            sharpe_ratio = mean(ER_t)/sqrt(var(ER_t)),
            mean_net_gain = mean(G_t)) %>% 
  ungroup() %>% 
  filter(! model %in% c("fNTK1","fNTK5")) %>% 
  mutate(sharpe_ratio = sharpe_ratio*sqrt(252/steps_ahead)) %>%  # annualized Sharpe ratio
  gather(key = "Type",value = "Value",-steps_ahead,-model) %>% 
  mutate(steps_ahead = paste0("$h = ",as.character(steps_ahead),"$"),
         Type = factor(Type,levels = c("mean_perc_traded_ops","mean_net_gain","mean_return","std_return","sharpe_ratio"))) %>% 
  spread(key = "steps_ahead",value = "Value") %>% 
  arrange(Type,model) %>% 
  select(Type,model,`$h = 1$`,`$h = 5$`,`$h = 10$`,`$h = 15$`,`$h = 20$`) %>% 
  mutate(across(where(is.numeric), round, 2)) %>% 
  filter(Type %in% c("mean_return","sharpe_ratio"))

