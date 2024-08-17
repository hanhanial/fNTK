function [tabP1, tabC1] = remove_double_strike_CS(tabP, tabC)
    if (length(unique(tabP.Strike))) < length(tabP.Strike) || (length(unique(tabC.Strike))) < length(tabC.Strike) 
            tabC = tabC(~isnan(tabC.ImpVol), :);
            tabP = tabP(~isnan(tabP.ImpVol), :);
            if (length(unique(tabP.Strike))) < length(tabP.Strike) || (length(unique(tabC.Strike))) < length(tabC.Strike) 
              [num,uniqstrike] = hist(tabC.Strike, unique(tabC.Strike));
              idx = find(ismember(tabC.Strike, uniqstrike(num>1)));
              tabC(idx, :) = [];
              [num,uniqstrike] = hist(tabP.Strike, unique(tabP.Strike));
              idx = find(ismember(tabP.Strike, uniqstrike(num>1)));
              tabP(idx, :) = [];
            end
    end
    tabP1 = tabP;
    tabC1 = tabC;
end