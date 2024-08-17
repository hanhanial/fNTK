library(orthogonalsplinebasis)
library(Splinets)

rescaling = function(x, current_bounds, new_bounds) {
  # Function to rescale x values to a new range
  
  lb1 = current_bounds[1]
  ub1 = current_bounds[2]
  lb2 = new_bounds[1]
  ub2 = new_bounds[2]
  
  if (lb1 > ub1 || lb2 > ub2) {
    stop('Invalid lower and upper bounds!!!')
  }
  
  x_new = ((x - lb1) / (ub1 - lb1)) * (ub2 - lb2) + lb2
  
  return(x_new)
}


rowwise_kronecker = function(rowA, rowB) {
  kron_row = kronecker(rowA, rowB)
  return(kron_row)
}


# Function to compute Mean Absolute Percentage Error (MAPE)
mape = function(actual, forecast) {
  if(length(actual) != length(forecast)) {
    stop("Length of actual and forecast vectors must be the same.")
  }
  
  if(any(actual == 0)) {
    stop("Actual values cannot be zero for MAPE calculation.")
  }
  
  abs_percentage_error = abs((actual - forecast) / actual)
  mean_abs_percentage_error = mean(abs_percentage_error, na.rm = TRUE)
  
  return(mean_abs_percentage_error)
}

# Function to compute Root Mean Squared Error (RMSE)
rmse = function(actual, forecast) {
  if(length(actual) != length(forecast)) {
    stop("Length of actual and forecast vectors must be the same.")
  }
  
  squared_error = (actual - forecast)^2
  mean_squared_error = mean(squared_error, na.rm = TRUE)
  root_mean_squared_error = sqrt(mean_squared_error)
  
  return(root_mean_squared_error)
}


###### function to obtain values of orthogonal bsplines (two-sided O-splines) ###### 
# NOTE: This is for (orthogonal) b-splines with NO interior knots!!!
obases_reg = function(date, chosen_PC_flag, degree = 3) {
  # extract data for selected date, and rescale M and tau to [0,1] range
  dat1 = dat %>% 
    filter(Date == date) %>% 
    mutate(rescaled_M = rescaling(x = Moneyness_M, 
                                  current_bounds = c(M_lb, M_ub),
                                  new_bounds = c(0, 1)),
           rescaled_tau = rescaling(x = Maturity, 
                                    current_bounds = c(tau_lb, tau_ub),
                                    new_bounds = c(0, 1)))
  
  #### evaluate orthogonal bsplines at M and tau
  knots = c(rep(x = 0, degree), 0, 1, rep(x = 1, degree))
  obases = OBasis(knots = knots, order = degree + 1)
  
  obases_at_M = evaluate(obases, x = dat1$rescaled_M)
  obases_at_tau = evaluate(obases, x = dat1$rescaled_tau)
  
  kronc_obases = t(mapply(rowwise_kronecker, 
                          as.data.frame(t(obases_at_tau)), 
                          as.data.frame(t(obases_at_M))))
  
  
  kronc_obases = as_tibble(kronc_obases)
  colnames(kronc_obases) = paste0(paste0("basis_", 1:ncol(kronc_obases)))
  
  obases_dat = kronc_obases %>% 
    mutate(IV = dat1$IV)
  
  # get basis coefficients
  obases_fitted = lm(IV ~ . +0, data = obases_dat)
  obases_coefs = ifelse(is.na(obases_fitted$coefficients),0,obases_fitted$coefficients)
  fitted_IVs = as.matrix(obases_dat[,names(obases_coefs)]) %*% as.matrix(obases_coefs)
  
  fitted_mape = mape(actual = obases_dat$IV, forecast = fitted_IVs)
  fitted_rmse = rmse(actual = obases_dat$IV, forecast = fitted_IVs)
  
  return(list(obases_coefs = as_tibble(t(as.matrix(obases_coefs))) %>% 
                mutate(Date = date),
              fitting_accuracy = tibble(MAPE = fitted_mape,
                                        RMSE = fitted_rmse),
              obases_vals = bind_cols(kronc_obases, dat1) %>% 
                select(-min_K_F_dist, -IV_atm, -min_K_F_ratio,
                       -rescaled_M, -rescaled_tau)
  )
  )
  
}


###### function to obtain values of splinets orthogonal bsplines ###### 
splinets_reg = function(date, chosen_PC_flag) {
  # extract data for selected date, and rescale M and tau to [0,1] range
  dat1 = dat %>% 
    filter(Date == date) %>% 
    mutate(rescaled_M = rescaling(x = Moneyness_M, 
                                  current_bounds = c(M_lb, M_ub),
                                  new_bounds = c(0, 1)),
           rescaled_tau = rescaling(x = Maturity, 
                                    current_bounds = c(tau_lb, tau_ub),
                                    new_bounds = c(0, 1)))
  
  #### splinets
  k = 2 # order
  n_knots = 7 # number of knots
  # xi = seq(0, 1, length.out = n_knots)
  xi = c(-0.5, -0.4, -0.3, 0.5, 1.3, 1.4, 1.5)
  so = splinet(knots = xi, smorder = k) 
  # plot(so$bs) # Plotting B-splines
  # plot(so$os) # Plotting Splinets
  
  
  #### evaluate this splinets at M and tau
  obases_at_M = evspline(object = so$os, x = dat1$rescaled_M)
  obases_at_tau = evspline(object = so$os, x = dat1$rescaled_tau)
  
  kronc_obases = t(mapply(rowwise_kronecker, 
                            as.data.frame(t(obases_at_tau[,-1])), 
                            as.data.frame(t(obases_at_M[,-1]))))
  
  kronc_obases = as_tibble(kronc_obases)
  colnames(kronc_obases) = paste0(paste0("basis_", 1:ncol(kronc_obases)))
  
  obases_dat = kronc_obases %>% 
    mutate(IV = dat1$IV)
  
  # get basis coefficients
  obases_fitted = lm(IV ~ . +0, data = obases_dat)
  obases_coefs = ifelse(is.na(obases_fitted$coefficients),0,obases_fitted$coefficients)
  fitted_IVs = as.matrix(obases_dat[,names(obases_coefs)]) %*% as.matrix(obases_coefs)
  
  fitted_mape = mape(actual = obases_dat$IV, forecast = fitted_IVs)
  fitted_rmse = rmse(actual = obases_dat$IV, forecast = fitted_IVs)
  
  return(list(obases_coefs = as_tibble(t(as.matrix(obases_coefs))) %>% 
                mutate(Date = date),
              fitting_accuracy = tibble(MAPE = fitted_mape,
                                        RMSE = fitted_rmse),
              obases_vals = bind_cols(kronc_obases, dat1) %>% 
                select(-min_K_F_dist, -IV_atm, -min_K_F_ratio,
                       -rescaled_M, -rescaled_tau)
  )
  )
  
}
