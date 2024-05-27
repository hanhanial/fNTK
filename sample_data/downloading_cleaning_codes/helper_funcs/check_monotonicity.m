function [mP1, mC1] = check_monotonicity(mP, mC, ttm)

%check monotonicity from the right and left of the spot price 
% (this includes the split case and double price curves)
dS0 = mP.Spot(1)*exp(mP.IR(1)*ttm); % Forward price = Spot*exp(r*tau)

%% Put option
mPut_temp0 = mP(mP.Strike < dS0, :);
vElimidx = zeros(size(mPut_temp0, 1), 1);
for m = size(mPut_temp0, 1):-1:2
    if mPut_temp0.best_offer(m) < mPut_temp0.best_bid(m-1)
        vElimidx(1:m-1) = 1;
        break;
    end
end
mPut_temp0(logical(vElimidx), :) = [];

mPut_temp1 = mP(mP.Strike >= dS0, :);
vElimidx = zeros(size(mPut_temp1, 1),1);
for m = 1:size(mPut_temp1, 1)-1
    if mPut_temp1.best_bid(m) > mPut_temp1.best_offer(m+1)
        vElimidx(m+1:end) = 1;
        break;
    end
end
mPut_temp1(logical(vElimidx),:)=[];

mP1 = [mPut_temp0; mPut_temp1];

%% Call options
mCall_temp0 = mC(mC.Strike < dS0, :);
vElimidx = zeros(size(mCall_temp0, 1), 1);
for m = size(mCall_temp0, 1):-1:2
    if mCall_temp0.best_bid(m) > mCall_temp0.best_offer(m-1)
        vElimidx(1:m-1) = 1;
        break;
    end
end
mCall_temp0(logical(vElimidx), :) = [];

mCall_temp1 = mC(mC.Strike >= dS0, :);
vElimidx = zeros(size(mCall_temp1, 1),1);
for m = 1:size(mCall_temp1, 1)-1
    if mCall_temp1.best_bid(m+1) > mCall_temp1.best_offer(m)
        vElimidx(m+1:end) = 1;
        break;
    end
end
mCall_temp1(logical(vElimidx),:) = [];

mC1 = [mCall_temp0; mCall_temp1];

end