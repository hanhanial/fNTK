clc;
clear;

cd '/hpctmp/e0823043/SPX/S03_CarrWu_2016/'
addpath helper_funcs/

main_model_dir = '/hpctmp/e0823043/SPX/S04_FAR/PCA_2009_2021_evenly_spaced/M_Andersen/';

%%
% keep only call (PC_flag == 1) or put (PC_flag == -1) options
all_option_type = ["Call" "Put"];

% number of steps ahead
all_steps_ahead = [1 5 10 20];

combs = combvec(1:numel(all_option_type), 1:numel(all_steps_ahead));
combs = [all_option_type(combs(1,:)); all_steps_ahead(combs(2,:))]';

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


for k = 1:size(combs,1)
    option_type = combs(k,1);
    steps_ahead = str2num(combs(k,2));

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

    %% keep only test dates and test day ahead dates that we actually use in our main models, e.g. Laplacian kernel
    tmp = readtable(strcat(main_model_dir,"S04_KRR_predict_actual_IV/KRR_laplacian/steps",num2str(steps_ahead),"ahead/",option_type,"_pred_test_IV.csv"));

    dates_to_be_kept = unique(tmp.test_date); % need to convert since test_date is not in the date time format
    dates_to_be_kept = datetime(num2str(dates_to_be_kept, '%d'),'InputFormat', 'yyyyMMdd','Format','yyyy-MM-dd');
    dates_to_be_kept = unique([dates_to_be_kept; tmp.test_day_ahead_date]);

    dat1 = dat1(ismember(dat1.Date,dates_to_be_kept),:);
    selected_day_indx = sort(unique(dat1.Day),'ascend');
    num_selected_days = length(selected_day_indx);

    %%
    % initial values of the parameters to be estimated
    x0  = [0.04, 0.1, 0.5, 0.3,-0.8];

    % upper and lower bounds of the parameters
    lb = [0.001,-Inf,0.001,0.001,-.999];
    ub = [Inf,Inf,Inf,Inf,.999];

    for i = 1:(num_selected_days-steps_ahead)
        % i = 1;
        dat_t = dat1(dat1.Day == selected_day_indx(i),:);
        dat_th = dat1(dat1.Day == selected_day_indx(i + steps_ahead),:);

        % Finding current SP index value:
        Spot_t = unique(dat_t.Spot);
        Spot_th = unique(dat_th.Spot);

        %% CW train
        K_T_IV_matrix = [dat_t.Strike, dat_t.Maturity/252, dat_t.IV];

        CW = @(params) CW_fit(params, Spot_t, K_T_IV_matrix);

        [CW_params,resnorm] = lsqnonlin(CW,x0,lb,ub);


        %% CW test
        CW_predict = CW_ivol_matrix([dat_th.Strike, dat_th.Maturity/252], CW_params, Spot_th);
        dat_th.fcst_IV = CW_predict';

        %% get test date and test day ahead date
        dat_th = renamevars(dat_th,'Date','test_day_ahead_date');
        dat_th.test_date = repmat(unique(dat_t.Date),size(dat_th,1),1);

        %% append output
        if i==1
            all_pred_res = dat_th;
        else
            all_pred_res = [all_pred_res;dat_th];
        end
    end

    all_pred_res.errors = all_pred_res.IV - all_pred_res.fcst_IV;

    %% prediction accuracy at actually observed IV values
    % overall accuracy
    overall_RMSE = sqrt(mean(all_pred_res.errors.^2));
    overall_MAE = mean(abs(all_pred_res.errors));
    overall_MAPE = mean(abs(all_pred_res.errors)./all_pred_res.IV);
    overall_accuracy = table("overall",overall_RMSE,overall_MAE,overall_MAPE,'VariableNames', ...
        ["period","RMSE","MAE","MAPE"]);
    accuracy_tab = overall_accuracy;

    % before covid accuracy
    bf_covid_actual_pred_IV = all_pred_res(all_pred_res.test_day_ahead_date < "2020-01-01",:);
    if size(bf_covid_actual_pred_IV,1) ~= 0
        bf_covid_RMSE = sqrt(mean(bf_covid_actual_pred_IV.errors.^2));
        bf_covid_MAE = mean(abs(bf_covid_actual_pred_IV.errors));
        bf_covid_MAPE = mean(abs(bf_covid_actual_pred_IV.errors)./bf_covid_actual_pred_IV.IV);
        bf_covid_accuracy = table("before_covid",bf_covid_RMSE,bf_covid_MAE,bf_covid_MAPE,'VariableNames', ...
            ["period","RMSE","MAE","MAPE"]);
        accuracy_tab = [accuracy_tab;bf_covid_accuracy];
    end

    % after covid accuracy
    af_covid_actual_pred_IV = all_pred_res(all_pred_res.test_day_ahead_date >= "2020-01-01",:);
    if size(af_covid_actual_pred_IV,1) ~= 0
        af_covid_RMSE = sqrt(mean(af_covid_actual_pred_IV.errors.^2));
        af_covid_MAE = mean(abs(af_covid_actual_pred_IV.errors));
        af_covid_MAPE = mean(abs(af_covid_actual_pred_IV.errors)./af_covid_actual_pred_IV.IV);
        af_covid_accuracy = table("after_covid",af_covid_RMSE,af_covid_MAE,af_covid_MAPE,'VariableNames', ...
            ["period","RMSE","MAE","MAPE"]);
        accuracy_tab = [accuracy_tab;af_covid_accuracy];
    end


    % remove unnecessary variables
    all_pred_res = removevars(all_pred_res,{'Day','errors','IV_atm','min_K_F_dist','min_K_F_ratio','Year'});

    if ~exist(strcat("S01_CarrWu_pred/"), 'dir')
        mkdir(strcat("S01_CarrWu_pred/"))
    end
    if ~exist(strcat("S01_CarrWu_pred/steps",num2str(steps_ahead),"ahead/"), 'dir')
        mkdir(strcat("S01_CarrWu_pred/steps",num2str(steps_ahead),"ahead/"))
    end

    odir = strcat("S01_CarrWu_pred/steps",num2str(steps_ahead),"ahead/");

    writetable(all_pred_res,strcat(odir,option_type,"_pred_test_IV.csv"))
    writetable(accuracy_tab,strcat(odir,option_type,"_pred_test_IV_accuracy.csv"))

end