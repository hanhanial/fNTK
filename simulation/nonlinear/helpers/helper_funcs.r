simulateVAR = function(num_processes, p, init,
                       pars, means,
                       inno_var,nT,
                       starting_seed,
                       burnin = 200) {
  '
supposed we want to simulate a certain number of VAR(p) processes, i.e. num_processes
where num_processes = 1 for univariate VAR(p) and num_processes>=2 for multivariate VAR(p)

- pars = list of (square) coefficient matrices, length(pars) = p for VAR(p)
--> each element of pars must be a square matrix, where nrow = ncol = num_processes

- means = vector of unconditional means (same length as num_processes)

- p = number of lags 
--> number of list in pars should be the same as lags
--> if some lag j (j<=p) is not actiave, then the corresponding j-th element in pars is a square, zero matrix

- nT = final number of simulated points (after excluding burn-in interations)

- init = initial values of VAR processes, 
--> init should have number of rows = p and number of columns = num_processes

- inno_var = variance (or covariance) matrix of innovation
--> inno_var is a square matrix, with nrow = ncol = num_processes
--> inno_var = diagonal matrix if innovation terms for different VAR processes are uncorrelated (i.e. inno_var = variance matrix)

- burnin = number of burn-in iterations
--> total number of iterations (i.e. simulated time points) = Nt + burnin
'
  
  # total number of simulation time points
  totTime = nT + burnin
  
  # initialize a matrix to store VAR(p) processes
  # nrow(res) = total number of time points to be simulated (including burn-in)
  # ncol(res) = num_processes = total number of VAR(p) processes to be simulated 
  res = matrix(0,totTime,num_processes)
  
  # initial values of VAR(p), since we need the first few p lags to generate sequence of VAR(p) processes
  res[1:p,] = init
  
  # to generate sequence of VAR(p) processes, from time point (p+1) to totTime
  for (t in (p+1):(totTime))
  {
    # t = p+1
    set.seed(starting_seed+t)
    
    # innovation terms
    inno_term = mvtnorm::rmvnorm(1,rep(0,num_processes), inno_var)
    
    # for each lag j, compute its contribution to the current value of the VAR processes (at time t)
    lag_terms = rowSums(do.call(cbind,lapply(seq_along(p),function(j){pars[[j]]%*%(res[t-p[j],]-means)})))
    
    # add up unconditional means, lag terms and innovation term to get current value at time t
    res[t,] = means + lag_terms + inno_term
  }
  
  return(res[(burnin+1):totTime,1:num_processes])
}

rescaling = function(x,lb_old,ub_old,lb_new,ub_new) {
  if (lb_old>ub_old | lb_new>ub_new) {
    stop('Invalid lower and upper bounds!!!')
  }
  x_new = ((x-lb_old)/(ub_old-lb_old))*(ub_new-lb_new) + lb_new
  return(x_new)
}
