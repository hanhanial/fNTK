simulate_2sin3cos = function(init,inno_var,
                             starting_seed,
                             burnin = 200,nT = 1000) {
  # total number of simulation time points
  totTime = nT + burnin
  
  x = rep(0,totTime)
  
  # initial values of the series
  x[1] = init
  
  # to generate time series
  for (t in 2:totTime)
  {
    set.seed(starting_seed+t)
    
    # innovation terms
    inno_term = rnorm(1,0, inno_var)
    
    x[t] = 2*sin(x[t-1]) + 3*cos(x[t-1]) + inno_term
  }
  
  return(x[(burnin+1):totTime])
}


simulate_2sin4cos = function(init,inno_var,
                             starting_seed,
                             burnin = 200,nT = 1000) {
  # total number of simulation time points
  totTime = nT + burnin
  
  x = rep(0,totTime)
  
  # initial values of the series
  x[1] = init
  
  # to generate time series
  for (t in 2:totTime)
  {
    set.seed(starting_seed+t)
    
    # innovation terms
    inno_term = rnorm(1,0, inno_var)
    
    x[t] = 2*sin(x[t-1]) + 4*cos(x[t-1]) + inno_term
  }
  
  return(x[(burnin+1):totTime])
}


simulate_sin2cos = function(init,inno_var = 0,
                            starting_seed,
                            burnin = 200,nT = 1000) {
  # total number of simulation time points
  totTime = nT + burnin
  
  x = rep(0,totTime)
  
  # initial values of the series
  x[1] = init
  
  # to generate time series
  for (t in 2:totTime)
  {
    set.seed(starting_seed+t)
    
    # innovation terms
    inno_term = rnorm(1,0, inno_var)
    
    x[t] = sin(x[t-1]) + 2*cos(x[t-1]) + inno_term
  }
  
  return(x[(burnin+1):totTime])
}
