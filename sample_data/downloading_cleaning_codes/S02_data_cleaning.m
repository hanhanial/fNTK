%{
 Filtering options by different criteria.
%}

clear
clc
close all

cd '/hpctmp/e0823043/fGNN/Analysis/S01_data_filtering/'
addpath helper_funcs/

idir = "/hpctmp/e0823043/fGNN/Data_download/DataDaily/";

% make sure output directory exists
if ~exist('S01_cleaned_data/', 'dir')
    mkdir('S01_cleaned_data/')
end
odir = 'S01_cleaned_data/';

%% ticker list
% original clean list of tickers (provided by Maria)
tickerlist = readtable('/hpctmp/e0823043/fGNN/Data_download/ticker_list/clean_tickerlist.csv');

% list of summary stats with liquidity ranking from highest to lowest (by average # options per day)
%tickers = readtable('summary_stats_LIQUIDITY.csv');
%tickers = tickers.Name(1:101); % keep top 100 most liquid options (including SPX, hence 101)
%tickers = string(tickerlist.Ticker(ismember(tickerlist.Company_name,tickers)));

tickers = string(tickerlist.Ticker);
tickers = [tickers; "SPX"]; % including SPX index (since tickerlist does not have SPX in it)

for i = 1:length(tickers)
    %     i = 1;
    ticker = tickers(i);

    data = readtable(strcat(idir,ticker,"_2009_2022.csv"));

    % remove options with NA values for ImpVol, or ImpVol==0
    data(isnan(data.ImpVol),:) = [];
    data(data.ImpVol==0,:) = [];

    data(data.best_bid == 0, :) = [];
    data(data.open_interest == 0, :) = [];

    % all unique dates of all options 
    dates = unique(data.Date);
    num_dates = length(dates); % number of unique dates

    all_clean_data = [];

    % loop over option traded dates 
    for d = 1:num_dates
        % d = 1;
        date = dates(d);

        % extract options on this given date
        day_data = data(data.Date == date, :);

        % get unique maturities of options traded on this date
        maturities = unique(day_data.Maturity);

        clean_day_data = [];

        % Loop over maturities for each date
        for m = 1:length(maturities)
            % m = 1

            % Pick contracts for the selected maturity
            tabPutMat = day_data(day_data.Maturity == maturities(m) & day_data.PC_flag == "P", :);
            tabCallMat = day_data(day_data.Maturity == maturities(m) & day_data.PC_flag == "C", :);

            % if there is call or put option(s) with negative or zero spot prices,
            % exclude this maturity
            if sum(tabPutMat.Spot <= 0) > 0 || sum(tabCallMat.Spot <=0) > 0
                continue;
            end

            % Exclude double strikes
            [tabPutMat, tabCallMat] = remove_double_strike_CS(tabPutMat, tabCallMat);

            % Exclude extreme strikes (Spot +/- 70%)
            [tabPutMat, tabCallMat] = remove_extreme_strike_CS(tabPutMat, tabCallMat);

            % Exclude best_bid > best_offer
            [tabPutMat, tabCallMat] = remove_error_bid_ask_CS(tabPutMat, tabCallMat);

            % Exclude this maturity if there are less than 2 (call or put) options left
            if height(tabPutMat) < 2 || height(tabCallMat) < 2
                continue;
            end

            % Exclude extreme bid and ask
            % spread = (best_offer – best_bid) --> remove all options with spreads > 10
            % times of the second largest spread (i.e. remove options with largest spread
            % if this spread is > 10 times of the second largest spread?)
            [tabPutMat, tabCallMat] = remove_extreme_bid_ask_CS(tabPutMat, tabCallMat);

            % if there are less than 2 put or call option left, after
            % all the exclusions, ignore this maturity and go to next
            % one
            if height(tabPutMat) < 2 || height(tabCallMat) < 2
                continue;
            end

            tabCallMat = sortrows(tabCallMat, 'Strike');
            tabPutMat = sortrows(tabPutMat, 'Strike');

            [tabPutMat, tabCallMat] = no_double_zero(tabPutMat, tabCallMat); % is this filtering really necessary, given that we already remove options with best_bid == 0????
            [tabPutMat, tabCallMat] = check_monotonicity(tabPutMat, tabCallMat, maturities(m)/252);
            [tabPutMat, tabCallMat] = no_flat_tails(tabPutMat, tabCallMat);

            % get IV_atm
            tabCallMat.IV_atm = repelem(get_IV_ATM(tabCallMat),height(tabCallMat))';
            tabPutMat.IV_atm = repelem(get_IV_ATM(tabPutMat),height(tabPutMat))';

            clean_day_data = [clean_day_data; tabCallMat; tabPutMat];
        end

        % combine all the dates 
        all_clean_data = [all_clean_data; clean_day_data];
    end

    % Moneyness_M = ln(Strike/Forward)/(sqrt(Maturity)*IV_atm)
    all_clean_data.Moneyness_M = log(all_clean_data.Strike./all_clean_data.Forward)./(sqrt(all_clean_data.Maturity/252).*all_clean_data.IV_atm);

    % remove IV_atm column to save space
    all_clean_data = removevars(all_clean_data,{'IV_atm'});

    writetable(all_clean_data,strcat(odir,ticker,"_2009_2022.csv"));
end
