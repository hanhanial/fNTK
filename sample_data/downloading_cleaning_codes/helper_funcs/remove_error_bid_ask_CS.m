function [tabP1, tabC1] = remove_error_bid_ask_CS(tabP, tabC)
       putspread = tabP.best_offer - tabP.best_bid;
       tabP(putspread < 0, :)=[];
       callspread = tabC.best_offer - tabC.best_bid;
       tabC(callspread < 0, :) = [];
       tabP1 = tabP;
       tabC1 = tabC;
end