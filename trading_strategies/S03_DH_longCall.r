library(tidyverse)
library(ggplot2)
library(gridExtra)
theme_set(theme_bw())

source("helper_funcs/long_delta_hedging_helper_funcs.r")

filtering_thrsh = 0.1 # threshold for filtering trading signals
effective_spread_measure = 0 # trading cost

compute_breakdown = F # whether to breakdown returns and sharpe ratio by (tau,M) values
all_steps_ahead = c(1,5,10,15,20)


##### #####
overall_day_by_day_res = list();

if (compute_breakdown) {
  long_call_DH_by_M = list(); long_call_DH_by_tau = list()
}

for (i in 1:length(all_steps_ahead)) {
  # i = 1
  steps_ahead = all_steps_ahead[i]
  
  dat = read_csv(paste0("S02_trading_signals/steps",as.character(steps_ahead),
                        "ahead_long_calls.csv")) 
  dat = dat %>% 
    filter(Volume_t != 0,
           abs(difference) > filtering_thrsh 
    ) %>% 
    mutate(MidPrice_t = (Ask_t + Bid_t)/2, Spread_t = abs(Ask_t - Bid_t),
           MidPrice_th = (Ask_th + Bid_th)/2, Spread_th = abs(Ask_th - Bid_th),
           
           MidSpot_t = (Spot_high_t + Spot_low_t)/2, SpotSpread_t = abs(Spot_high_t - Spot_low_t),
           MidSpot_th = (Spot_high_th + Spot_low_th)/2, SpotSpread_th = abs(Spot_high_th - Spot_low_th)
    )
  # since for long call DH, we buy options and sell underlying stock on day t and 
  # sell options and buy underlying stock on day t+h
  # we compute the buy option price (BuyPremium_t) and the sell stock price (SellSpot_t) on day t 
  # and sell price (SellPremium_th) and the buy stock price (BuySpot_th) on day t+h
  # according to the effective spread measure accordingly
  dat = dat %>% 
    mutate(BuyPremium_t = MidPrice_t + Spread_t*effective_spread_measure/2,
           SellPremium_th = MidPrice_th - Spread_th*effective_spread_measure/2,
           
           SellSpot_t = Spot_close_t,
           BuySpot_th = Spot_close_th 
    )
  
  ########## DH returns overall ########## 
  out = long_call_DH(data = dat, 
                      steps_ahead = steps_ahead)
  
  overall_day_by_day_res[[paste0("h = ",steps_ahead)]] = out$day_by_day_res
  
  
  ########## DH returns by (tau,M) categories ########## 
  if (compute_breakdown) {
    dat = dat %>% 
      mutate(tau_group = ifelse(Maturity_t <=60, "[5,60]",
                                ifelse(Maturity_t <= 120, "(60,120]",
                                       ifelse(Maturity_t <= 180,"(120,180]","(180,252]"))),
             M_group = ifelse(Moneyness_M_t <= -0.5, "[-2,-0.5]",
                              ifelse(Moneyness_M_t <= 0, "(-0.5,0]",
                                     ifelse(Moneyness_M_t <= 0.5,"(0,0.5]","(0.5,2]"))))
    
    tau_groups = unique(dat$tau_group)
    M_groups = unique(dat$M_group)
    
    
    for (i in 1:length(M_groups)) {
      # i = 1
      
      long_call_DH_out = long_call_DH(data = dat %>% 
                                           filter(M_group == M_groups[i]),
                                         steps_ahead = steps_ahead)
      long_call_DH_by_M[[paste0(M_groups[i],"_steps",steps_ahead,"ahead")]] = long_call_DH_out$day_by_day_res %>% 
        mutate(M_group = M_groups[i])
    }
    
    for (i in 1:length(tau_groups)) {
      # i = 1
      
      long_call_DH_out = long_call_DH(data = dat %>% 
                                             filter(tau_group == tau_groups[i]),
                                           steps_ahead = steps_ahead)
      long_call_DH_by_tau[[paste0(tau_groups[i],"_steps",steps_ahead,"ahead")]] = long_call_DH_out$day_by_day_res %>% 
        mutate(tau_group = tau_groups[i])
    }
  }
  
  
}


overall_day_by_day_res = Reduce(x = overall_day_by_day_res,f = bind_rows)
write_csv(overall_day_by_day_res,paste0("S03_trading_returns/LongCallDH_overall_thrs",
                                        as.character(filtering_thrsh),"_EffSpread",
                                        as.character(effective_spread_measure),".csv"))



if (compute_breakdown) {
  long_call_DH_by_M = Reduce(x = long_call_DH_by_M,f = bind_rows)
  write_csv(long_call_DH_by_M,paste0("S03_trading_returns/LongCallDH_by_M_thrs",
                                      as.character(filtering_thrsh),"_EffSpread",
                                      as.character(effective_spread_measure),".csv"))
  
  long_call_DH_by_tau = Reduce(x = long_call_DH_by_tau,f = bind_rows)
  write_csv(long_call_DH_by_tau,paste0("S03_trading_returns/LongCallDH_by_tau_thrs",
                                        as.character(filtering_thrsh),"_EffSpread",
                                        as.character(effective_spread_measure),".csv"))
}

