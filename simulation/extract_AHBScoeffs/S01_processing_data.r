library(tidyverse)

options = read_csv("~/Desktop/Hannah/work/SPX/Analysis/S01_data_filtering/DailyOptions/S03_moneyness/SPX_2009_2021.csv")

options = options %>% 
  filter(Moneyness_M>=-2,Moneyness_M<=2,
         Maturity>=5, Maturity<=252) %>% 
  rename(M = Moneyness_M) %>% # using Andersen's moneyness M definition  
  mutate(tau = Maturity/252) %>% 
  mutate(M2 = M^2,
         Mtau = M*tau,
         ln_IV = log(IV),
         Date = as.character(as.Date(as.character(Date), '%Y%m%d'))) #%>% 
  # select(Date,PC_flag,Strike,Spot,Maturity,IV,ln_IV,M,M2,Mtau,tau)
head(options)
nrow(options)
#  options that have call-put parity relationships
dups = options %>%
  group_by(Date,tau,Strike) %>%
  mutate(n=n()) %>%
  filter(n>1) %>%
  select(PC_flag,Date,tau,Strike,M,IV) %>%
  arrange(Date,Strike) %>%
  group_by(Date,tau,Strike,IV) %>%
  mutate(n=n())
print(paste0("There are ",nrow(dups)/2,
             " call-put option pairs with call-put parity relationships - of which ",
             sum(dups$n>1)/2," pairs have the same IV"))
print(paste0("About ",round((nrow(dups) - sum(dups$n>1))/nrow(options)*100,2),
             "% of options violated call-put parity relationship"))

puts = options %>% 
  filter(PC_flag==-1)
calls = options %>% 
  filter(PC_flag==1)

write_csv(puts,paste0("S01_processing_data/Put_2009_2021.csv"))
write_csv(calls,paste0("S01_processing_data/Call_2009_2021.csv"))

