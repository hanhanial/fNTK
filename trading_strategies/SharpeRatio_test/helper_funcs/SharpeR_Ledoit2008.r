load("/Users/hannahlai/Desktop/Hannah/work/SPX/Analysis/S04_FAR/trading_strategies/S05_Sharpe_Ratio_tests/helper_funcs/SharpeR/Sharpe.RData")

SR_test = function(data, model_1, model_2, h, period, block_size) {

  if (period == "overall") {
    dat = data %>% 
      filter(model %in% c(model_1, model_2),
             steps_ahead == h) %>% 
      select(test_date, model, ER_t) %>% 
      spread(key = model, value = ER_t) %>% 
      select(-test_date) %>% 
      as.matrix()
  } else {
    dat = data %>% 
      filter(model %in% c(model_1, model_2),
             Year == period,
             steps_ahead == h) %>% 
      select(test_date, model, ER_t) %>% 
      spread(key = model, value = ER_t) %>% 
      select(-test_date) %>% 
      as.matrix()
  }
  
  dat = replace(dat,is.na(dat),0)
  
  hac_res = hac.inference(dat)
  bstr_ts_res = boot.time.inference(ret = dat,b = block_size, M = 1000)
  
  return(tibble(Model_1 = model_1,
                Model_2 = model_2,
                steps_ahead = h,
                Period = period,
                
                annualSR_1 = hac_res$Sharpe.Ratios['SR1.hat']*sqrt(252/h),
                annualSR_2 = hac_res$Sharpe.Ratios['SR2.hat']*sqrt(252/h),
                
                HAC_pval = hac_res$p.Values['HAC'],
                bstr_ts_pval = bstr_ts_res$p.Value,
                bstr_ts_blocksize = block_size))
}
