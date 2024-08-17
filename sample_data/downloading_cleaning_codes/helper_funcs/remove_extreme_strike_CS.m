function [tabP1, tabC1] = remove_extreme_strike_CS(tabP, tabC)
   tabP(tabP.Strike < tabP.Spot*0.3, :) = [];
   tabP(tabP.Strike > tabP.Spot*1.7, :) = [];
   tabC(tabC.Strike < tabC.Spot*0.3, :) = [];
   tabC(tabC.Strike > tabC.Spot*1.7, :) = [];
   tabP1 = tabP;
   tabC1 = tabC;
end