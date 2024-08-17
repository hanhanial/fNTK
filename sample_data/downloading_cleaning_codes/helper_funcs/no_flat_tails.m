function [mP1, mC1] = no_flat_tails(mP, mC)
%% mid prices
mC.mid_price = 0.5*(mC.best_bid + mC.best_offer);
mP.mid_price = 0.5*(mP.best_bid + mP.best_offer);

%% Put options
for i = 1:(size(mP, 1)-1)
    if mP.mid_price(1) >= mP.mid_price(2)
        mP(1, :) = [];
    else
        break;
    end
end

%% Call options
for i=1:(size(mC, 1)-1)
    if mC.mid_price(end) >= mC.mid_price(end-1)
        mC(end, :) = [];
    else
        break;
    end
end

%%
mP1 = removevars(mP,{'mid_price'}); mC1 = removevars(mC,{'mid_price'});
end
