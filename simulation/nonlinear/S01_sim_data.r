library(tidyverse)

wdir = "/hpctmp/e0823043/SPX/simulation_AHBS/all_params_2Sin4Cos/"
# wdir = "~/Desktop/Hannah/work/SPX/Analysis/simulation_AHBS/all_params_2Sin4Cos/"

source(paste0(wdir,"helpers/helpers.r"))
source(paste0(wdir,"helpers/helper_funcs.r"))
source(paste0(wdir,"helpers/simulate_nonlinear_series.r"))

odir = paste0(wdir,"S01_sim_data/")
dir.create(odir, showWarnings = FALSE, recursive = TRUE)

params = read_csv("/hpctmp/e0823043/SPX/S02_Bernales2014/Daily_2009_2021/M_Andersen/S02_DeterministicLinearModel/Call_2009_2021.csv")
# params = read_csv("~/Desktop/Hannah/work/SPX/Analysis/S02_Bernales2014/Daily_2009_2021/M_Andersen/S02_DeterministicLinearModel/Call_2009_2021.csv")

params1 = params %>% 
  # filter(Date <= "2019-12-31") %>% 
  select(b0,b1,b2,b3,b4,b5) %>% 
  gather(key = "coef",value = "value")

params_stats = params1 %>% 
  group_by(coef) %>% 
  summarise(
    min = min(value),
    quantile_lb = quantile(x = value,0.15),
    mean = mean(value),
    quantile_ub = quantile(x = value,0.85),
    max = max(value),
    std = sqrt(var(value))
  )

num_points = 2000
num_sims = 100
innovation_var = 0.0001

for (i in 1:num_sims) {
  # i = 1
  
  # initial seed - for reproduction
  init_seed = i*10000
  
  out = tibble(coef_name = character(),coef_value = numeric(),Day = numeric())
  
  coef_names = params_stats$coef
  for (k in 1:length(coef_names)) {
    # k = 2
    coef = coef_names[k]
    
    if (k %in% c(1,3,4)) {
      sim_coef = simulate_2sin4cos(init = runif(n = 1,
                                                min = params_stats$quantile_lb[params_stats$coef==coef],
                                                max = params_stats$quantile_ub[params_stats$coef==coef]),
                                   inno_var = innovation_var,
                                   starting_seed = init_seed,
                                   burnin = 200,nT = num_points)
    } else {
      sim_coef = simulate_2sin4cos(init = runif(n = 1,
                                                min = params_stats$quantile_lb[params_stats$coef==coef],
                                                max = params_stats$quantile_ub[params_stats$coef==coef]),
                                   inno_var = innovation_var,
                                   starting_seed = init_seed,
                                   burnin = 200,nT = num_points)
    }
    
    sim_coef = rescaling(x = sim_coef,lb_old = min(sim_coef),ub_old = max(sim_coef),
                         lb_new = params_stats$quantile_lb[params_stats$coef==coef],
                         ub_new = params_stats$quantile_ub[params_stats$coef==coef])
    # scatter.smooth(x = sim_coef[1:(length(sim_coef)-1)],y = sim_coef[2:length(sim_coef)],
    #                xlab = paste0("lag ",coef), ylab = paste0("lead ",coef))
    
    out = bind_rows(out,tibble(coef_name = coef,
                               coef_value = sim_coef,
                               Day = 1:num_points))
  }
  out = out %>% spread(key = "coef_name",value = "coef_value")
  
  
  M_vals = seq(from = -2.5, to = 2.5, by = 0.1)
  tau_vals = seq(from = 0.02, to = 1, by = 0.05)
  
  tau_M = as_tibble(expand.grid(tau_vals,M_vals)) %>% 
    rename(tau = Var1, M = Var2)
  
  out1 = lapply(1:num_points, function(x) {
    sim_IV = tau_M %>% 
      mutate(b0 = out$b0[out$Day==x],
             b1 = out$b1[out$Day==x],
             b2 = out$b2[out$Day==x],
             b3 = out$b3[out$Day==x],
             b4 = out$b4[out$Day==x],
             b5 = out$b5[out$Day==x],
             Day = x) %>% 
      mutate(IV = b0 + b1*M + b2*M^2 + b3*tau + b4*tau^2 + b5*M*tau + 
               (b1 + b2)*(M^3) + (b2 + b3)*(tau^3) + 
               (b0 + b1*b2)*(M^2)*(tau^3) + (b2 + b3*b4)*(M^3)*(tau^2) 
             # (b1 + b2)*(M^2)*tau + (b2 + b3)*M*(tau^2) + (b3 + b4)*(M^2)*(tau^2) +      
      )
    return(sim_IV)
  })
  
  out2 = bind_rows(out1, .id = "column_label")
  hist(out2$IV)
  sum(out2$IV < 0)/nrow(out2)
  
  # add noise to IV
  out2$IV = out2$IV + rnorm(1,0, innovation_var)
  
  out2$IV = rescaling(x = out2$IV,lb_old = min(out2$IV),ub_old = max(out2$IV),
                      lb_new = 0.1,
                      ub_new = 0.95)
  
  # out3 = out2 %>% 
  #   filter(tau == 0.52,
  #          M == 0) %>% 
  #   arrange(Day)
  # scatter.smooth(x = out3$IV[1:(nrow(out3)-1)],y = out3$IV[2:nrow(out3)],
  #                xlab = paste0("lag IV"), ylab = paste0("lead IV"))
   
  
  out2 = out2 %>% 
    arrange(Day,tau,M) %>%
    group_by(Day) %>% 
    mutate(option_index = 1:n()) %>% 
    ungroup() 
  
  IV = out2 %>% 
    select(Day,option_index,IV) %>% 
    spread(key = option_index,value = IV) %>% 
    arrange(Day) %>% 
    select(-Day) %>% 
    t() %>% 
    as.data.frame()
  
  dates = out2 %>% 
    select(Day) %>% 
    distinct() %>% 
    mutate(Date = seq(from = as.Date("2009-01-01"), 
                      to = as.Date("2009-01-01") + num_points - 1,
                      by = "day")) %>% 
    mutate(Date = gsub(pattern = "-",replacement = "",x = Date))
  
  M_IV = out2 %>% 
    filter(Day == 1) %>% 
    select(tau,M,option_index) %>% 
    arrange(option_index)
  M = M_IV %>% select(M)
  tau = M_IV %>% select(tau)
  
  params = out2 %>% 
    select(Day,b0,b1,b2,b3,b4,b5) %>% 
    distinct()
  
  out2 = out2 %>% 
    inner_join(dates) %>% 
    select(-column_label,-b0,-b1,-b2,-b3,-b4,-b5) %>% 
    rename(Moneyness_M = M, Maturity = tau) %>% 
    mutate(Maturity = Maturity*252)
  
  write_csv(params,paste0(odir,"seed_",init_seed,"_AHBSparams.csv"))
  write_csv(out2,paste0(odir,"seed_",init_seed,"_full_data.csv"))
  write_csv(IV,paste0(odir,"seed_",init_seed,"_IV.csv"),col_names = FALSE)
  write_csv(M,paste0(odir,"seed_",init_seed,"_M.csv"),col_names = FALSE)
  write_csv(tau,paste0(odir,"seed_",init_seed,"_tau.csv"),col_names = FALSE)
  write_csv(dates,paste0(odir,"seed_",init_seed,"_date_indexing.csv"))
  
  
  #### orthogonal b-splines ####
  all_dates = unique(dates$Date)
  M_lb = -2.5; M_ub = 2.5
  tau_lb = 5; tau_ub = 252
  
  # fitting for each day d
  all_obases_coefs = tibble()
  all_fitting_accuracy = tibble()
  all_obases_vals = tibble()
  for (d in 1:length(all_dates)) {
    # if (d %% 200 == 0) {
    #   print(paste0("Processing for day: ", as.character(d),"/",as.character(length(all_dates))))
    # }
    
    out = obases_reg(dat = out2,
                     date = all_dates[d])
    
    all_obases_coefs = bind_rows(all_obases_coefs, out$obases_coefs)
    all_fitting_accuracy = bind_rows(all_fitting_accuracy, out$fitting_accuracy)
    all_obases_vals = bind_rows(all_obases_vals, out$obases_vals)
  }
  
  # since all days (in each simulation) share exactly the same grids of (tau, M)
  # so we can just save it once
  # NOTE: this is no longer correct if we observed IV at irregular grids of (tau, M)
  all_obases_vals = all_obases_vals %>% 
    select(contains("basis_"),option_index) %>% 
    distinct()
  
  if (nrow(all_obases_vals) != nrow(M)) {
    stop("It seems that observed IV are at different grids of (tau, M) for each day, pls check!!!")
  }
  
  write_csv(all_obases_coefs,paste0(odir,"seed_",init_seed,"_obspline_coeffss.csv"))
  write_csv(all_obases_vals,paste0(odir,"seed_",init_seed,"_obspline_values.csv"))
  
}
