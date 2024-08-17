library(tidyverse)

source("/hpctmp/e0823043/SPX/S04_FAR/trading_strategies/S05_Sharpe_Ratio_tests/helper_funcs/SharpeR_Ledoit2008_HPC.r")

idir = "/hpctmp/e0823043/SPX/S04_FAR/trading_strategies/S03_trading_returns/"
strategy = "DN_ShortStraddle"
threshold = 0.005
effective_spread = 0

odir = "/hpctmp/e0823043/SPX/S04_FAR/trading_strategies/S05_Sharpe_Ratio_tests/S01_SharpeRatio_test/"
dir.create(file.path(odir), recursive = TRUE)

#### ####
ret_dat = read_csv(paste0(idir, strategy, "_overall_thrs", as.character(threshold),
                          "_EffSpread", as.character(effective_spread), ".csv"))
ret_dat = ret_dat %>% 
  mutate(Year = format(test_date,'%Y'))

# list of all unique years
years = unique(ret_dat$Year)

# list of all models
models = unique(ret_dat$model)
models = grep(pattern = "splines_",x = models,invert = T,value = T)

# list of forecasting horizons
fcst_horizons = unique(ret_dat$steps_ahead)

# combinations of 2 models
combs = combn(models, 2)
combs = tibble("model_1" = combs[1,],
               "model_2" = combs[2,])

###### tune for block size for each forecasting horizon ###### 
block_sizes_for_each_h = tibble(block_size = numeric(),
                                steps_ahead = numeric())
for (step in c(1, 5, 10, 15, 20)) {
  print(paste0('-- Tuning block size for ',as.character(step),' steps ahead --'))
  selected_block_sizes = c()
  
  # randomly select some pairs of models 
  for (i in sample(x = 1:nrow(combs),size = 10,replace = FALSE)) {
    m1 = combs$model_1[i]
    m2 = combs$model_2[i]
    print(paste0("selecting block size using ", m1, " and ", m2))
    
    dat = ret_dat %>% 
      filter(model %in% c(m1, m2),
             steps_ahead == step) %>% 
      select(test_date, model, ER_t) %>% 
      spread(key = model, value = ER_t) %>% 
      select(-test_date) %>% 
      as.matrix()
    dat = replace(dat,is.na(dat),0)
    
    block_size_res = block.size.calibrate(ret = dat,
                                          b.vec = c(5, 10, 15, 20, 25, 30), # 35, 40, 50
                                          
                                          # try with small value of K first, 
                                          # since it takes very long to calibrate block size if K is large
                                          K = 100 # 1000
    )
    selected_block_sizes = c(selected_block_sizes, block_size_res$b.optimal)
  }
  
  # get the most selected block size across all model comparisons
  final_block_size = as.numeric(names(sort(-table(selected_block_sizes)))[1]) 
  
  block_sizes_for_each_h = bind_rows(block_sizes_for_each_h,
                                     tibble(block_size = final_block_size,
                                            steps_ahead = step))
  print(paste0('--> final block size is ',as.character(final_block_size)))
  
}
print(block_sizes_for_each_h)
write_csv(block_sizes_for_each_h,
          paste0(odir, strategy,"_thsh",threshold,"_EffSpread",effective_spread,"_BlockSizes.csv"))

block_sizes_for_each_h = read_csv(paste0(odir, strategy,"_thsh",threshold,"_EffSpread",effective_spread,"_BlockSizes.csv"))

pval_list = list()
for (test_type in c("HAC", "bstrTS")) {
  for (hz in fcst_horizons) {
    for (p in c("overall")) { # , years
      tmp = matrix(0, nrow = length(models), ncol = length(models))
      colnames(tmp) = rownames(tmp) = models
      
      pval_list[[paste0(p, "_steps", hz, "ahead_",test_type)]] = tmp
    }
  } 
}
all_res = list()
for (i in 1:nrow(combs)) {
  # i = 17
  m1 = combs$model_1[i]
  m2 = combs$model_2[i]
  print(paste0(as.character(i), "/", as.character(nrow(combs)),
               ": comparing ",m1," and ",m2))
  
  for (hz in fcst_horizons) {
    for (p in c("overall")) { # years
      res = SR_test(data = ret_dat,
                    model_1 = m1, model_2 = m2, 
                    h = hz, period = p,
                    block_size = block_sizes_for_each_h$block_size[block_sizes_for_each_h$steps_ahead == hz])
      print(res)
      
      all_res[[paste0(p, "_steps", hz, "ahead_", m1,"_", m2)]] = res
      
      pval_list[[paste0(p, "_steps", hz, "ahead_HAC")]][m1, m2] = pval_list[[paste0(p, "_steps", hz, "ahead_HAC")]][m2, m1] = res$HAC_pval
      
      pval_list[[paste0(p, "_steps", hz, "ahead_bstrTS")]][m1, m2] = pval_list[[paste0(p, "_steps", hz, "ahead_bstrTS")]][m2, m1] = res$bstr_ts_pval
      res$bstr_ts_pval
    }
  }
}


all_res = bind_rows(all_res)
write_csv(all_res,
          paste0(odir,strategy,"_thsh",threshold,"_EffSpread",effective_spread,"_SRres.csv"))
saveRDS(pval_list,
        paste0(odir, strategy,"_thsh",threshold,"_EffSpread",effective_spread,"_SRpvals.rds"))
