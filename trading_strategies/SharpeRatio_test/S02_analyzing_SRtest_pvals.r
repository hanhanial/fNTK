library(tidyverse)

strategy = "DN_ShortStraddle"
threshold = 0.005
effective_spread = 0

all_SR_tests = tibble()
for (s in c("1","5","10","15","20")) {
  pvals = readRDS(paste0("S01_SharpeRatio_test/", strategy,"_thsh",threshold,"_EffSpread",effective_spread,"_SRpvals.rds"))
  
  sorted_models = c("DNN_RW",
                    "CW_RW", "CW_DNN",
                    "AHBS_RW", "AHBS_DNN", "AHBS",
                    "fRW", "fLinK", "fGauK", "fLapK", "fNTK3")
  
  tmp = pvals[[paste0("overall_steps",s,"ahead_bstrTS")]]
  diag(tmp) = "-"
  tmp = tmp[sorted_models,sorted_models]
  
  sorted_models = gsub(pattern = "_",replacement = "-",x = sorted_models)
  sorted_models = gsub(pattern = "fNTK3",replacement = "fNTK",x = sorted_models)
  sorted_models[sorted_models == "AHBS"] = "AHBS-VAR"
  
  colnames(tmp) = rownames(tmp) = sorted_models
  
  tmp = as.data.frame(tmp) %>% 
    rownames_to_column("model") %>% 
    mutate(steps_ahead = s)
  
  all_SR_tests = bind_rows(all_SR_tests, tmp)
  
}
