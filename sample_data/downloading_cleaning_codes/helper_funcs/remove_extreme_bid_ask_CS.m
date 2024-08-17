function [tabP1, tabC1] = remove_extreme_bid_ask_CS(tabP, tabC)
  putspread = sortrows(tabP.best_offer - tabP.best_bid);
  tabP((tabP.best_offer - tabP.best_bid) > putspread(end - 1)*10, :) = [];
  callspread = sortrows(tabC.best_offer - tabC.best_bid);
  tabC((tabC.best_offer - tabC.best_bid) > callspread(end - 1)*10, :) = [];
  tabP1 = tabP;
  tabC1 = tabC;
end
            