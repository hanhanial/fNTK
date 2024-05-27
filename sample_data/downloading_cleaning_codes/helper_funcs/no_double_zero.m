function [mP1, mC1] = no_double_zero(mP, mC)
%% Put options
vElimidx = zeros(size(mP, 1), 1);
for p = size(mP, 1):-1:1
    if mP.best_bid(p) == 0
        vElimidx(p) = 1;
        if p > 1 && mP.best_bid(p-1) == 0
            vElimidx(1:p-1) = 1;
            break;
        end
    end
end
mP(logical(vElimidx), :) = [];

%% Call options
vElimidx = zeros(size(mC, 1),1);
for p = size(mC, 1):-1:1
    if mC.best_bid(p) == 0
        vElimidx(p) = 1;
        if p > 1 && mC.best_bid(p-1) == 0
            vElimidx(1:p-1) = 1;
            break;
        end
    end
end
mC(logical(vElimidx),:) = [];

%%
mP1 = mP; mC1 = mC;
end