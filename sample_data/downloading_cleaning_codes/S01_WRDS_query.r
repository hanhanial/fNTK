library(tidyverse)
library(bizdays)
library(zoo)

# NOTE: files in the shared scratch directory are deleted after one week (168 hours)!!!!
wdir = "/scratch/eur/hanhanial/"
odir = paste0(wdir,"fGNN_data/DataDaily/")
dir.create(odir, showWarnings = FALSE, recursive = TRUE)

# weekdays = vector of characters that define which days of the week are not business days
business_calendar = create.calendar('my_calendar', weekdays = c('saturday','sunday'))

###### Interest rate ###### 
IR_query = dbSendQuery(wrds,
                       paste0("select date,days,rate",
                              " from OPTIONM.ZEROCD")) # " where days <= 500")
IR_data = dbFetch(IR_query, n=-1)
dbClearResult(IR_query)

IR_data = as_tibble(IR_data) %>% 
  rename(Date = date, Maturity_calendar_days = days, IR = rate) %>% 
  mutate(IR = IR/100) # divide by 100 since the reported IR is in percentage

###### Retrieve for each ticker ###### 
# list of tickers
ticker_list = read_csv("ticker_list/clean_tickerlist.csv")

# we also want to download SPX index options too
ticker_list = bind_rows(tibble(Ticker = "SPX", SecID = 108105),
                        ticker_list)

# select years to be downloaded
years = 2009:2022

for (i in 1:nrow(ticker_list)) { 
  # i = 1
  selected_ticker = ticker_list$Ticker[i]
  selected_Sec_ID = ticker_list$SecID[i]
  
  print(paste0("Retrieving data for ticker ",i,"/",nrow(ticker_list),
               ": ",selected_ticker))
  
  #### check ticker unique secID ####
  # to ensure we get correct data (i.e. correct ticker and secID)
  secID_query = dbSendQuery(wrds,
                            paste0("select secid,effect_date",
                                   " from OPTIONM.SECNMD",
                                   " where ticker = \'",selected_ticker,"\'"))
  secID_data = dbFetch(secID_query, n=-1)
  dbClearResult(secID_query)
  
  if (sum(secID_data$secid==selected_Sec_ID) == 0) {
    print("There is no valid Sec ID corresponding to the given ticker and sec ID!!!")
    break
  }
  
  #### download data for each of selected years #### 
  all_years_data = tibble()
  for (j in 1:length(years)) {
    # j = 1
    
    selected_year = years[j]
    print(paste0("Retrieving option data for year ",selected_year))
    
    option_vars = c('secid', 'optionid', 'date', 'exdate', 'cp_flag', 
                    'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest',
                    'impl_volatility', 'delta', 'gamma', 'vega','theta')
    option_query = dbSendQuery(wrds,
                               paste0("select ",paste(option_vars,collapse=","),
                                      " from OPTIONM.OPPRCD",selected_year,
                                      " where secid = ",selected_Sec_ID))
    option_data = dbFetch(option_query, n=-1)
    dbClearResult(option_query)
    
    option_data = as_tibble(option_data) %>% 
      # since the recorded value is strike price of the option times 1000
      mutate(strike_price = strike_price/1000) %>% 
      rename(SecID = secid, Option_ID = optionid,
             Date = date, Exp_date = exdate,
             PC_flag = cp_flag, Strike = strike_price,
             Volume = volume, ImpVol = impl_volatility,
             Delta = delta, Gamma = gamma, Vega = vega, Theta = theta)
    
    # Maturity in business days and also in calendar days
    option_data = option_data %>% 
      mutate(Maturity = bizdays(Date, Exp_date, cal = business_calendar),
             Maturity_calendar_days = as.numeric(Exp_date - Date)) 
    
    #### Spot price #### 
    spot_query = dbSendQuery(wrds,
                             paste0("select date,close",
                                    " from OPTIONM.SECPRD",selected_year,
                                    " where secid = ",selected_Sec_ID))
    spot_data = dbFetch(spot_query, n=-1)
    dbClearResult(spot_query)
    
    spot_data = as_tibble(spot_data) %>% 
      rename(Date = date, Spot = close)
    
    option_data = left_join(option_data,spot_data)
    
    #### Forward price #### 
    forward_query = dbSendQuery(wrds,
                                paste0("select date,expiration,forwardprice",
                                       " from OPTIONM.FWDPRD",selected_year,
                                       " where secid = ",selected_Sec_ID))
    forward_data = dbFetch(forward_query, n=-1)
    dbClearResult(forward_query)
    
    forward_data = as_tibble(forward_data) %>% 
      rename(Date = date, Exp_date = expiration, Forward = forwardprice) %>% 
      mutate(Maturity_calendar_days = as.numeric(Exp_date - Date)) # Maturity in calendar days
    
    #### IR and forward price corresponding to each option on each day #### 
    # only keep options with maturity <= 400 (calendar days), this to make sure no NA values
    # from IR (i.e. no options with maturity longer than downloaded IR) -->
    # to include longer maturity options, need to re-download IR
    option_data1 = option_data %>% 
      filter(Maturity_calendar_days <= 450)
    
    # get forward price for all tau in each day t in dat
    t_tau = option_data1 %>% 
      select(Date,Maturity_calendar_days) %>% 
      distinct() %>% 
      full_join(forward_data %>% select(-Exp_date) %>% rename(WRDS_Forward = Forward)) %>% 
      full_join(IR_data %>% rename(WRDS_IR = IR) %>% filter(Date %in% option_data1$Date)) %>% 
      arrange(Date,Maturity_calendar_days)
    
    # interpolate to get Forward price for each tau on each day t
    t_tau = t_tau %>%
      group_by(Date) %>%
      mutate(Forward = na.approx(WRDS_Forward, na.rm=FALSE, rule = 2),
             IR = na.approx(WRDS_IR, na.rm=FALSE, rule = 2)) %>% 
      select(-WRDS_Forward, -WRDS_IR) %>% 
      
      # in WRDS data, sometimes there are more than one Forward Price or IR for a (Date,Maturity_calendar_days) 
      # in this case, take mean
      group_by(Date,Maturity_calendar_days) %>% 
      summarise(Forward = mean(Forward),
                IR = mean(IR)) %>% 
      ungroup()
    
    option_data1 = left_join(option_data1,t_tau) 
    
    # keep only maturity in trading days
    option_data1 = option_data1 %>% select(-Maturity_calendar_days)
    
    print(paste0("There are ",sum(is.na(option_data1$Forward)),
                 " without Forward price and ",sum(is.na(option_data1$IR))," options without IR"))
    
    # remove options with NA or zero values for IV
    print(paste0("There are ",round(sum(is.na(option_data1$ImpVol) | option_data1$ImpVol==0) / nrow(option_data1) *100,2),
                 "% of options with NA or zero values IV"))
    option_data1 = option_data1 %>% 
      filter(!is.na(ImpVol),ImpVol!=0)
    
    # remove zero or negative Spot price
    print(paste0("There are ",round(sum(option_data1$Spot <= 0) / nrow(option_data1) *100,2),
                 "% of options with non-positive spot price"))
    option_data1 = option_data1 %>% 
      filter(Spot > 0)
    
    all_years_data = bind_rows(all_years_data,
                               option_data1)
  }
  
  # write downloaded data to file
  write_csv(all_years_data,paste0(odir,selected_ticker,'_',years[1],'_',years[length(years)],'.csv'))
}
