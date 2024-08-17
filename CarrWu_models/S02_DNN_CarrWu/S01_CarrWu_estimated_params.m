clc;
clear;

cd '/hpctmp/e0823043/SPX/S03_CarrWu_2016/S02_DNN_on_CarrWu_residuals/'
addpath ../helper_funcs/

%%
% keep only call (PC_flag == 1) or put (PC_flag == -1) options
all_option_type = ["Call" "Put"];

%%
% keep options in years 2009-2021
dat = readtable('/hpctmp/e0823043/SPX/DailyOptions/S03_moneyness/SPX_2009_2021.csv');
dat = dat(dat.Year>=2009,:);
unique(dat.Year)

dat = renamevars(dat,'Moneyness_M','M');

dat(dat.M < -2,:) = [];
dat(dat.M > 2,:) = [];
dat(dat.Maturity < 5,:) = [];
dat(dat.Maturity > 252,:) = [];


for k = 1:length(all_option_type)
    option_type = all_option_type(k);

    %% keep only the selected type of options
    if (option_type == "Call")
        dat1 = dat(dat.PC_flag == 1,:);
    else
        dat1 = dat(dat.PC_flag == -1,:);
    end

    %%
    % all dates to be fitted
    dates = unique(dat1.Date);
    num_days = length(dates);

    % create a table for day indexing
    day_indexing = table(dates,'VariableNames',{'Date'});
    day_indexing.converted_Date = datetime(num2str(day_indexing.Date, '%d'),'InputFormat', 'yyyyMMdd','Format','yyyy-MM-dd');
    day_indexing = sortrows(day_indexing,'converted_Date','ascend');
    day_indexing.Day = (1:1:num_days)';

    dat1 = innerjoin(dat1,day_indexing,'keys','Date');

    % change to the correct date formatting
    dat1 = removevars(dat1,{'Date'});
    dat1 = renamevars(dat1,'converted_Date','Date');

    %%
    % initial values of the parameters to be estimated
    x0  = [0.04, 0.1, 0.5, 0.3,-0.8];

    % upper and lower bounds of the parameters
    lb = [0.001,-Inf,0.001,0.001,-.999];
    ub = [Inf,Inf,Inf,Inf,.999];

    for i = 1:num_days
        % i = 1;
        dat_t = dat1(dat1.Day == i,:);

        % Finding current SP index value:
        Spot_t = unique(dat_t.Spot);

        %% CW train
        K_T_IV_matrix = [dat_t.Strike, dat_t.Maturity/252, dat_t.IV];

        CW = @(params) CW_fit(params, Spot_t, K_T_IV_matrix);

        [CW_params,resnorm] = lsqnonlin(CW,x0,lb,ub);

        if i==1
            all_est_params = CW_params;
        else
            all_est_params = [all_est_params; CW_params];
        end

    end

    all_est_params = array2table(all_est_params,'VariableNames',{'v','m','w','n','rho'});
    all_est_params.Date = day_indexing.converted_Date;

    if ~exist(strcat("S01_CarrWu_estimated_params/"), 'dir')
        mkdir(strcat("S01_CarrWu_estimated_params/"))
    end
    odir = "S01_CarrWu_estimated_params/";
    writetable(all_est_params,strcat(odir,option_type,"_CW_est_params.csv"))

end

