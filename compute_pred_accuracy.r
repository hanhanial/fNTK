library(tidyverse)
library(forecast)
library(lubridate)

args = commandArgs(trailingOnly = TRUE) 
print(args)

option_type = args[1]

odir = "/hpctmp/e0823043/SPX/Figures/S05_prediction_accuracy/"
dir.create(file.path(odir), recursive = TRUE)

steps_ahead = as.character(c(1,5,10,15,20))


DM_res = list(); DM_types = c("MAPE","RMSE")
overall_accuracy = tibble()
accuracy_by_tau = tibble(); accuracy_by_M = tibble()

for (s in steps_ahead) {
  # s = steps_ahead[2]
  
  # functional models
  functional_dir = "/hpctmp/e0823043/SPX/S04_FAR/OrthogonalBsplines/"
  functional_res = tibble(model = c("fRW","fLinK","fLapK","fGauK",
                                    
                                    # testing number of hidden layers in NTK
                                    "fNTK1", "fNTK3", "fNTK5", 
                                    
                                    # testing Kernel Ridge regression with empirical NTK
                                    "fNTK3_KR",
                                    
                                    # sieve method to tune o-splines degree
                                    "deg2_fNTK3", "deg4_fNTK3", "deg5_fNTK3", "deg6_fNTK3",
                                    
                                    # try different number of training samples
                                    "fNTK3_1000TrainSams", "fNTK3_2000TrainSams",
                                    
                                    "fLinK_1000TrainSams","fLapK_1000TrainSams","fGauK_1000TrainSams",
                                    
                                    "fLinK_2000TrainSams","fLapK_2000TrainSams","fGauK_2000TrainSams"
  )) %>% 
    
    mutate(dir = c(paste0(functional_dir,"S04_FAR_random_walk/steps",s,"ahead/"),
                   paste0(functional_dir,"S03_FAR_KRR/linear/steps",s,"ahead/"),
                   paste0(functional_dir,"S03_FAR_KRR/laplacian/steps",s,"ahead/"),
                   paste0(functional_dir,"S03_FAR_KRR/rbf/steps",s,"ahead/"),
                   
                   # testing number of hidden layers in NTK
                   paste0(functional_dir,"S02_FAR_NTK/degree3_tau_NaN_M_NaN/steps",s,"ahead/HiddenLayers1/"),
                   paste0(functional_dir,"S02_FAR_NTK/degree3_tau_NaN_M_NaN/steps",s,"ahead/HiddenLayers3/"),
                   paste0(functional_dir,"S02_FAR_NTK/degree3_tau_NaN_M_NaN/steps",s,"ahead/HiddenLayers5/"),
                   
                   # testing Kernel Ridge regression with empirical NTK
                   paste0(functional_dir,"S05_FAR_NTKR/degree3_tau_NaN_M_NaN/steps",s,"ahead/HiddenLayers3/"),
                   
                   # sieve method to tune o-splines degree
                   paste0(functional_dir,"S02_FAR_NTK/degree2_tau_NaN_M_NaN/steps",s,"ahead/HiddenLayers3/"),
                   paste0(functional_dir,"S02_FAR_NTK/degree4_tau_NaN_M_NaN/steps",s,"ahead/HiddenLayers3/"),
                   paste0(functional_dir,"S02_FAR_NTK/degree5_tau_NaN_M_NaN/steps",s,"ahead/HiddenLayers3/"),
                   paste0(functional_dir,"S02_FAR_NTK/degree6_tau_NaN_M_NaN/steps",s,"ahead/HiddenLayers3/"),
                   
                   # try different number of training samples
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S04_FAR/S02_FAR_NTK/TrainingSize1000/steps",s,"ahead/"),
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S04_FAR/S02_FAR_NTK/TrainingSize2000/steps",s,"ahead/"),
                   
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S04_FAR/S03_FAR_KRR/TrainingSize1000/linear/steps",s,"ahead/"),
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S04_FAR/S03_FAR_KRR/TrainingSize1000/laplacian/steps",s,"ahead/"),
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S04_FAR/S03_FAR_KRR/TrainingSize1000/rbf/steps",s,"ahead/"),
                   
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S04_FAR/S03_FAR_KRR/TrainingSize2000/linear/steps",s,"ahead/"),
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S04_FAR/S03_FAR_KRR/TrainingSize2000/laplacian/steps",s,"ahead/"),
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S04_FAR/S03_FAR_KRR/TrainingSize2000/rbf/steps",s,"ahead/")
    )
    )
  
  
  # Carr & Wu models
  carrwu_dir = "/hpctmp/e0823043/SPX/S03_CarrWu_2016/"
  CW_res = tibble(model = c("CW_RW","CW_DNN")) %>% 
    mutate(dir = c(paste0(carrwu_dir,"S01_CarrWu_pred/steps",s,"ahead/"),
                   paste0(carrwu_dir,"S02_DNN_on_CarrWu_residuals/S02_DNN_on_CWresiduals/steps",s,"ahead/"))
    )
  
  
  # AHBS models
  AHBS_dir = "/hpctmp/e0823043/SPX/S02_Bernales2014/Daily_2009_2021/M_Andersen/"
  AHBS_res = tibble(model = c("AHBS_VAR","AHBS_RW","AHBS_DNN", 
                              "AHBS_VAR_1000TrainSams", "AHBS_VAR_2000TrainSams")) %>% 
    mutate(dir = c(paste0(AHBS_dir,"S03_VARp/steps",s,"ahead/"),
                   paste0(AHBS_dir,"S04_RandomWalk/steps",s,"ahead/"),
                   paste0(AHBS_dir,"S05_DNN_on_AHBSresiduals/steps",s,"ahead/"),
                   
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S02_Bernales2014/TrainingSize1000/S03_VARp/steps",s,"ahead/"),
                   paste0("/hpctmp/e0823043/SPX/varying_training_size/S02_Bernales2014/TrainingSize2000/S03_VARp/steps",s,"ahead/"))
    )
  
  # dimension reduction models
  dimred_dir = "/hpctmp/e0823043/SPX/S06_PCA_vs_Autoencoders/"
  dimred_res = tibble(model = c("PCA_eigenvecs1_LSTM",
                                "PCA_eigenvecs3_LSTM",
                                "PCA_thrsh0.90_LSTM", 
                                "PCA_thrsh0.95_LSTM", 
                                "PCA_thrsh0.99_LSTM", 
                                
                                "AE_feats1_LSTM",
                                "AE_feats2_LSTM",
                                "AE_feats3_LSTM",
                                "AE_feats4_LSTM",
                                "AE_feats8_LSTM")) %>%
    
    mutate(dir = c(paste0(dimred_dir,"S01_PCA_LSTM_predict_actual_IV/PCA_eigenvecs1/steps",s,"ahead/"),
                   paste0(dimred_dir,"S01_PCA_LSTM_predict_actual_IV/PCA_eigenvecs3/steps",s,"ahead/"),
                   paste0(dimred_dir,"S01_PCA_LSTM_predict_actual_IV/PCA_thrsh0.9/steps",s,"ahead/"),
                   paste0(dimred_dir,"S01_PCA_LSTM_predict_actual_IV/PCA_thrsh0.95/steps",s,"ahead/"),
                   paste0(dimred_dir,"S01_PCA_LSTM_predict_actual_IV/PCA_thrsh0.99/steps",s,"ahead/"),
                   
                   paste0(dimred_dir,"S02_Autoencoder_LSTM_predict_actual_IV/AEhiddfeats1/steps",s,"ahead/"),
                   paste0(dimred_dir,"S02_Autoencoder_LSTM_predict_actual_IV/AEhiddfeats2/steps",s,"ahead/"),
                   paste0(dimred_dir,"S02_Autoencoder_LSTM_predict_actual_IV/AEhiddfeats3/steps",s,"ahead/"),
                   paste0(dimred_dir,"S02_Autoencoder_LSTM_predict_actual_IV/AEhiddfeats4/steps",s,"ahead/"),
                   paste0(dimred_dir,"S02_Autoencoder_LSTM_predict_actual_IV/AEhiddfeats8/steps",s,"ahead/"))
    )
  
  # other models
  others_dir = "/hpctmp/e0823043/SPX/S05_other_models/"
  others_res = tibble(model = c("DNN_RW",
                                
                                # predict interpolated IV values using standard FCNN, LSTM, or, NTK
                                "DNN_interp","LSTM_interp","NTK_interp",
                                
                                # fNTK using unrestricted HAR, i.e., 22 lags individually as inputs
                                "fNTK_unrestrictedHAR", 
                                
                                # predict o-splines coefs using standard FCNN or LSTM model
                                "fFCNN", "fLSTM")
  ) %>% 
    mutate(dir = c(paste0(others_dir,"S01_pureFCNN_on_IV_tau_m/steps",s,"ahead/"),
                   
                   # predict interpolated IV values using standard FCNN, LSTM, or, NTK
                   paste0(others_dir,"S02_FCNN_on_interpolatedIVs_predict_actual_IV/steps",s,"ahead/"),
                   paste0(others_dir,"S03_LSTM_on_interpolatedIVs_predict_actual_IV/steps",s,"ahead/"),
                   paste0(others_dir,"S04_NTK_FCNN_on_interpolatedIVs_predict_actual_IV/steps",s,"ahead/"),
                   
                   # fNTK using unrestricted HAR, i.e., 22 lags individually as inputs
                   paste0(others_dir,"S05_FAR_NTK_22InpsSeparately/steps",s,"ahead/"),
                   
                   # predict o-splines coefs using standard FCNN or LSTM model
                   paste0(others_dir,"S06_FAR_FCNN/steps",s,"ahead/"),
                   paste0(others_dir,"S07_FAR_LSTM/steps",s,"ahead/")
    )
    )
  
  
  files = bind_rows(functional_res, CW_res) %>% 
    bind_rows(AHBS_res) %>% 
    bind_rows(dimred_res) %>%
    bind_rows(others_res)
  
  
  res = lapply(files$model, function(x) {
    # x = files$model[2]
    tmp = read_csv(paste0(files$dir[files$model==x],option_type,"_pred_test_IV.csv"))
    
    colnames(tmp)[colnames(tmp) == "Moneyness_M"] = "M"
    
    tmp = tmp %>% 
      mutate(option_type = ifelse(PC_flag==1,"Call","Put")) %>% 
      select(test_date,test_day_ahead_date,M,Maturity,IV,fcst_IV)
    tmp = tmp %>% 
      mutate(model = x)
    
    # check if the test_date column is in date format: if not, convert to date
    if (! is.Date(tmp$test_date)) {
      tmp = tmp %>% mutate(test_date = as.Date(as.character(test_date),format = "%Y%m%d"))
    }
    
    return(tmp)
  })
  
  all_res = bind_rows(res) %>% 
    select(-test_day_ahead_date) %>% 
    mutate(step_ahead = as.numeric(s),
           AE = abs(IV-fcst_IV),
           APE = abs((IV-fcst_IV)/IV)
    ) 
  
  ####### Overall prediction accuracy by models for step s #######
  (accuracy = all_res %>%
     group_by(step_ahead,model) %>% 
     summarise(MAPE = mean(abs(AE/IV)), 
               MAE = mean(abs(AE)),
               RMSE = sqrt(mean(AE^2)),
               R2 = 1 - sum((fcst_IV - IV)^2) / sum((IV - mean(IV))^2) ) %>% 
     mutate(Year = "all_years"))
  
  
  (accuracy_by_year = all_res %>%
      mutate(Year = format(test_date, format = "%Y")) %>% 
      group_by(Year,step_ahead,model) %>% 
      summarise(MAPE = mean(abs(AE/IV)), 
                MAE = mean(abs(AE)),
                RMSE = sqrt(mean(AE^2)),
                R2 = 1 - sum((fcst_IV - IV)^2) / sum((IV - mean(IV))^2) ))
  
  # merge overall prediction accuracy by models across all steps ahead
  overall_accuracy = overall_accuracy  %>% 
    bind_rows(bind_rows(accuracy, accuracy_by_year))
  
  
  ####### Prediction accuracy by models and (tau,M) #######
  accuracy_by_tau_M = all_res %>% 
    mutate(Year = format(test_date, format = "%Y"),
           tau_group = ifelse(Maturity <=60, "[5,60]",
                              ifelse(Maturity <= 120, "(60,120]",
                                     ifelse(Maturity <= 180,"(120,180]","(180,252]"))),
           M_group = ifelse(M <= -0.5, "[-2,-0.5]",
                            ifelse(M <= 0, "(-0.5,0]",
                                   ifelse(M <= 0.5,"(0,0.5]","(0.5,2]"))))
  
  accuracy_by_tau_all_years = accuracy_by_tau_M %>% 
    group_by(step_ahead,model,tau_group) %>% 
    summarise(MAPE = mean(abs(AE/IV)), 
              MAE = mean(abs(AE)),
              RMSE = sqrt(mean(AE^2)),
              R2 = 1 - sum((fcst_IV - IV)^2) / sum((IV - mean(IV))^2) ) %>% 
    ungroup() %>% 
    mutate(Year = "all_years")
  
  accuracy_by_tau_by_year = accuracy_by_tau_M %>% 
    group_by(Year,step_ahead,model,tau_group) %>% 
    summarise(MAPE = mean(abs(AE/IV)), 
              MAE = mean(abs(AE)),
              RMSE = sqrt(mean(AE^2)),
              R2 = 1 - sum((fcst_IV - IV)^2) / sum((IV - mean(IV))^2) ) %>% 
    ungroup() 
  
  accuracy_by_M_all_years = accuracy_by_tau_M %>% 
    group_by(step_ahead,model,M_group) %>% 
    summarise(MAPE = mean(abs(AE/IV)), 
              MAE = mean(abs(AE)),
              RMSE = sqrt(mean(AE^2)),
              R2 = 1 - sum((fcst_IV - IV)^2) / sum((IV - mean(IV))^2) ) %>% 
    ungroup() %>% 
    mutate(Year = "all_years")
  
  accuracy_by_M_by_year = accuracy_by_tau_M %>% 
    group_by(Year,step_ahead,model,M_group) %>% 
    summarise(MAPE = mean(abs(AE/IV)), 
              MAE = mean(abs(AE)),
              RMSE = sqrt(mean(AE^2)),
              R2 = 1 - sum((fcst_IV - IV)^2) / sum((IV - mean(IV))^2) ) %>% 
    ungroup()
  
  accuracy_by_tau = accuracy_by_tau %>% 
    bind_rows(bind_rows(accuracy_by_tau_all_years,accuracy_by_tau_by_year))
  accuracy_by_M = accuracy_by_M %>% 
    bind_rows(bind_rows(accuracy_by_M_all_years,accuracy_by_M_by_year))
  
  
  ##### Diebold Mariano test #####
  combs = combn(x = files$model,m = 2)
  for (d in DM_types) {
    
    if (d == "MAPE") {
      tmp = all_res %>%
        group_by(test_date,model) %>% 
        summarise(error_g = mean(abs(AE/IV)) ) # mean(AE) 
    } else {
      tmp = all_res %>%
        group_by(test_date,model) %>% 
        summarise(error_g = sqrt(mean(AE^2)) ) # mean(AE) 
    }
    
    
    # record number of times models in rows are better than models in columns
    tmp1 = matrix(data = NaN,nrow = length(files$model),ncol = length(files$model))
    rownames(tmp1) = files$model
    colnames(tmp1) = files$model
    
    
    for (j in 1:ncol(combs)) {
      m1 = tmp %>%
        filter(model == combs[1,j]) %>%
        pull(error_g)
      m2 = tmp %>%
        filter(model == combs[2,j]) %>%
        pull(error_g)
      
      # "less" --> alternative hypothesis: method 2 (column) is less accurate than method 1 (row)
      # so if p-val is <0.05 in an entry, it means the method 1 (row) is better than method 2 (column)
      p12 = dm.test(m1,m2,alternative="less",h = as.numeric(s),power = 1)$p.value
      p21 = dm.test(m2,m1,alternative="less",h = as.numeric(s),power = 1)$p.value
      tmp1[combs[1,j],combs[2,j]] = p12
      tmp1[combs[2,j],combs[1,j]] = p21
      
    }
    
    DM_res[[paste0(d,"_",s,"_stepahead")]] = as.data.frame(tmp1) %>% 
      rownames_to_column("model") %>% 
      mutate(steps_ahead = as.numeric(s))
    
  }
  
}


write_csv(overall_accuracy,paste0("S05_prediction_accuracy/",option_type,"_overall_accuracy.csv"))

write_csv(accuracy_by_tau,
          paste0("S05_prediction_accuracy/",option_type,"_accuracy_by_tau.csv"))
write_csv(accuracy_by_M,
          paste0("S05_prediction_accuracy/",option_type,"_accuracy_by_M.csv"))



MAPE_DM = Reduce(x = DM_res[grepl(x = names(DM_res),pattern = "MAPE")],
                 f = bind_rows)
RMSE_DM = Reduce(x = DM_res[grepl(x = names(DM_res),pattern = "RMSE")],
                 f = bind_rows)

write_csv(MAPE_DM,paste0("S05_prediction_accuracy/",option_type,"_MAPE_DM_test_pvals.csv"))
write_csv(RMSE_DM,paste0("S05_prediction_accuracy/",option_type,"_RMSE_DM_test_pvals.csv"))

