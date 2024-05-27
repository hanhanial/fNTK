function [IV_atm] = get_IV_ATM(dat)

% absolute distance between Strike price and the given forward price
dat.dist = abs(dat.Strike - dat.Forward);

% IV_atm of options with the given traded date and maturity
min_dist = min(dat.dist);
IV_atm = dat.ImpVol(find(dat.dist==min_dist));

if length(IV_atm)>1
    disp("There are more than one IV_atm...")
    IV_atm = mean(IV_atm);
end

end