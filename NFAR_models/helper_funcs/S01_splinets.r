library(tidyverse)
source("helper_funcs/helpers.r")

# NOTE: "orthogonalsplinebasis" gives a lot more stable orthogonal bsplines than "Splinets"

###### ###### 
# Moneyness M and Maturity tau cut-offs
M_lb = -2.5; M_ub = 2.5
tau_lb = 1; tau_ub = 280

for (PC in c(-1,1)) {
  
  selected_PC_flag = PC # PC flag: 1 for Call, -1 for Put
  option_type = ifelse(selected_PC_flag==1, "Call", "Put")
  
  # output directory
  odir = paste0(getwd(),"/S01_OrthogonalBsplines/",option_type,"/")
  dir.create(file.path(odir), recursive = TRUE)
  
  ###### read in option data ###### 
  dat = read_csv("~/Desktop/Hannah/work/SPX/Analysis/S01_data_filtering/S03_moneyness/SPX_2009_2021.csv")
  dat = dat %>% 
    filter(Moneyness_M >= M_lb, Moneyness_M <= M_ub, 
           Maturity >= tau_lb, Maturity <= tau_ub, 
           PC_flag == selected_PC_flag)
  all_dates = unique(dat$Date)
  
  
  ###### fitting for each day d ###### 
  all_splinets_coefs = tibble()
  all_fitting_accuracy = tibble()
  all_splinets_vals = tibble()
  
  for (d in 1:length(all_dates)) {
    if (d %% 200 == 0) {
      print(paste0("Processing for day: ", as.character(d),"/",as.character(length(all_dates))))
    }
    
    out = splinets_reg(date = all_dates[d],
                       chosen_PC_flag = selected_PC_flag)
    
    all_splinets_coefs = bind_rows(all_splinets_coefs, out$splinets_coefs)
    all_fitting_accuracy = bind_rows(all_fitting_accuracy, out$fitting_accuracy)
    all_splinets_vals = bind_rows(all_splinets_vals, out$splinets_vals)
    
  }
  
  
  ###### output ###### 
  # fitting accuracy
  summary(all_fitting_accuracy$MAPE)
  png(paste0(odir, option_type,"_obases_MAPE.png"))
  hist(all_fitting_accuracy$MAPE,
       main = paste0(option_type," options - MAPE of fitting b-spline on ",length(all_dates)," days"))
  dev.off()
  png(paste0(odir, option_type,"_obases_RMSE.png"))
  hist(all_fitting_accuracy$RMSE,
       main = paste0(option_type," options - RMSE of fitting b-spline on ",length(all_dates)," days"))
  dev.off()
  
  
  # coefficients
  print(all_splinets_coefs %>% select(-Date) %>% summary())
  write_csv(all_splinets_coefs,paste0(odir,"obspline_coeffss.csv"))
  
  
  # final Moneyness M and Maturity tau cut-offs to be used for forecasting
  final_M_lb = -2; final_M_ub = 2
  final_tau_lb = 5; final_tau_ub = 252
  all_splinets_vals = all_splinets_vals %>% 
    filter(Moneyness_M >= final_M_lb, Moneyness_M <= final_M_ub, 
           Maturity >= final_tau_lb, Maturity <= final_tau_ub)
  write_csv(all_splinets_vals,paste0(odir,"obspline_values.csv"))
}