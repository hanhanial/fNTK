library(tidyverse)

source("helper_funcs/straddles_helper_funcs.r")

idir = "S02_trading_signals/"

filtering_thrsh = 0.1 # threshold for filtering trading signals
effective_spread_measure = 0 # trading cost

compute_breakdown = F # whether to breakdown returns and sharpe ratio by (tau,M) values
all_steps_ahead = c(1,5,10,15,20)


##### #####
overall_day_by_day_res = list();

if (compute_breakdown) {
  long_straddle_by_M = list(); long_straddle_by_tau = list()
}

for (s in 1:length(all_steps_ahead)) {
  # s = 4
  steps_ahead = all_steps_ahead[s]
  
  long_calls = read_csv(paste0(idir,"steps",steps_ahead,"ahead_long_calls.csv")) %>% 
    filter(Volume_t != 0,
           abs(difference) > filtering_thrsh #,test_date >= "2020-01-01"
    ) %>% 
    mutate(MidPrice_t = (Ask_t + Bid_t)/2, Spread_t = abs(Ask_t - Bid_t),
           MidPrice_th = (Ask_th + Bid_th)/2, Spread_th = abs(Ask_th - Bid_th))
  # since for long straddles, we buy on day t and sell on day t+h
  # we compute the buy price (BuyPrice_t) on day t and sell price (SellPrice_th) on day t+h
  # according to the effective spread measure accordingly
  long_calls = long_calls %>% 
    mutate(BuyPrice_t = MidPrice_t + Spread_t*effective_spread_measure/2,
           SellPrice_th = MidPrice_th - Spread_th*effective_spread_measure/2)
  
  long_puts = read_csv(paste0(idir,"steps",steps_ahead,"ahead_long_puts.csv"))%>% 
    filter(Volume_t != 0,
           abs(difference) > filtering_thrsh #,test_date >= "2020-01-01"
    ) %>% 
    mutate(MidPrice_t = (Ask_t + Bid_t)/2, Spread_t = abs(Ask_t - Bid_t),
           MidPrice_th = (Ask_th + Bid_th)/2, Spread_th = abs(Ask_th - Bid_th))
  # since for long straddles, we buy on day t and sell on day t+h
  # we compute the buy price (BuyPrice_t) on day t and sell price (SellPrice_th) on day t+h
  # according to the effective spread measure accordingly
  long_puts = long_puts %>% 
    mutate(BuyPrice_t = MidPrice_t + Spread_t*effective_spread_measure/2,
           SellPrice_th = MidPrice_th - Spread_th*effective_spread_measure/2)
  
  
  ####### Overall returns ####### 
  long_straddle_dat = bind_rows(long_calls %>% mutate(Option_type = "Call"),
                                long_puts %>% mutate(Option_type = "Put")) %>% 
    # filter and keep options with both call and put
    group_by(test_date,model,Strike,Maturity_t) %>%
    mutate(num_ops = n()) %>%
    ungroup() %>%
    filter(num_ops == 2) 
  
  # get the actual value of delta of put options (which are < 0) - since in the previous code,
  # I keep the absolute values of deltas of puts
  long_straddle_dat = long_straddle_dat %>% 
    mutate(delta_t = ifelse(Option_type == "Put", - delta_t, delta_t))
  hist(long_straddle_dat$delta_t[long_straddle_dat$Option_type=="Call"])
  hist(long_straddle_dat$delta_t[long_straddle_dat$Option_type=="Put"])
  
  # for each option pair, we compute number of call options to be traded,
  # and number of put options to be traded, to ensure delta-neutral
  quantity_of_calls = long_straddle_dat %>% 
    group_by(test_date,model,Strike,Maturity_t) %>%
    summarise(num_traded_calls = - delta_t[Option_type == "Put"] / (delta_t[Option_type == "Call"] - delta_t[Option_type == "Put"])) %>% 
    ungroup() %>% 
    mutate(num_traded_puts = 1 - num_traded_calls)
  
  # assume that we buy (or sell) num_traded_puts units of put options and 
  # buy (or sell) a total of num_traded_calls units of call options
  # to ensure delta-neutral straddle portfolio
  long_straddle_dat = inner_join(long_straddle_dat,quantity_of_calls) %>% 
    mutate(BuyPrice_t = ifelse(Option_type=="Put", BuyPrice_t*num_traded_puts, BuyPrice_t*num_traded_calls),
           SellPrice_th = ifelse(Option_type=="Put", SellPrice_th*num_traded_puts, SellPrice_th*num_traded_calls))
  
  overall_res = long_returns(data = long_straddle_dat,steps_ahead = steps_ahead)
  overall_day_by_day_res[[paste0("h = ",steps_ahead)]] = overall_res$day_by_day_res
  
  ########## Straddle returns by (tau,M) categories ########## 
  if (compute_breakdown) {
    long_straddle_dat = long_straddle_dat %>% 
      mutate(tau_group = ifelse(Maturity_t <=60, "[5,60]",
                                ifelse(Maturity_t <= 120, "(60,120]",
                                       ifelse(Maturity_t <= 180,"(120,180]","(180,252]"))),
             M_group = ifelse(Moneyness_M_t <= -0.5, "[-2,-0.5]",
                              ifelse(Moneyness_M_t <= 0, "(-0.5,0]",
                                     ifelse(Moneyness_M_t <= 0.5,"(0,0.5]","(0.5,2]"))))
    
    
    tau_groups = unique(long_straddle_dat$tau_group)
    M_groups = unique(long_straddle_dat$M_group)
    
    for (i in 1:length(M_groups)) {
      # i = 1
      
      straddle_out = long_returns(data = long_straddle_dat %>% 
                                    filter(M_group == M_groups[i]) %>% 
                                    group_by(test_date,model,Strike,Maturity_t) %>%
                                    mutate(num_ops = n()) %>%
                                    ungroup() %>%
                                    filter(num_ops == 2),
                                  steps_ahead = steps_ahead)
      long_straddle_by_M[[paste0(M_groups[i],"_steps",steps_ahead,"ahead")]] = straddle_out$day_by_day_res %>% 
        mutate(M_group = M_groups[i])
    }
    
    for (i in 1:length(tau_groups)) {
      # i = 1
      
      straddle_out = long_returns(data = long_straddle_dat %>% 
                                    filter(tau_group == tau_groups[i]) %>% 
                                    group_by(test_date,model,Strike,Maturity_t) %>%
                                    mutate(num_ops = n()) %>%
                                    ungroup() %>%
                                    filter(num_ops == 2),
                                  steps_ahead = steps_ahead)
      long_straddle_by_tau[[paste0(tau_groups[i],"_steps",steps_ahead,"ahead")]] = straddle_out$day_by_day_res %>% 
        mutate(tau_group = tau_groups[i])
    }
  }
  
  
}

overall_day_by_day_res = Reduce(x = overall_day_by_day_res,f = bind_rows)
write_csv(overall_day_by_day_res,paste0("S03_trading_returns/DN_LongStraddle_overall_thrs",
                                        as.character(filtering_thrsh),"_EffSpread",
                                        as.character(effective_spread_measure),".csv"))

if (compute_breakdown) {
  long_straddle_by_M = Reduce(x = long_straddle_by_M,f = bind_rows)
  write_csv(long_straddle_by_M,paste0("S03_trading_returns/DN_LongStraddle_by_M_thrs",
                                      as.character(filtering_thrsh),"_EffSpread",
                                      as.character(effective_spread_measure),".csv"))
  
  long_straddle_by_tau = Reduce(x = long_straddle_by_tau,f = bind_rows)
  write_csv(long_straddle_by_tau,paste0("S03_trading_returns/DN_LongStraddle_by_tau_thrs",
                                        as.character(filtering_thrsh),"_EffSpread",
                                        as.character(effective_spread_measure),".csv"))
}

