%%% CW

x0  = [0.04, 0.1, 0.5, 0.3,-0.8];
lb = [0.001,-Inf,0.001,0.001,-.999];
ub = [Inf,Inf,Inf,Inf,.999];

CW_results = NaN(1,9);

for i=1:length(PossibleCurrentDates)-1
    
    % Finding current loop dates:
    Aux = find(DateOptionsCurrent == PossibleCurrentDates(i));
    Aux_ahead = find(DateOptionsCurrent == PossibleCurrentDates(i+1));
    
    % Finding current SP index value:
    CurrentPrice = SPDaily(find(DateSP == PossibleCurrentDates(i)));
    CurrentPrice_ahead = SPDaily(find(DateSP == PossibleCurrentDates(i+1)));
    
    % Saving options data for the current date:
    OptionInfo = [OptionPrice(Aux,:) time2exp(Aux,:) strike_price(Aux,:) dummy_put(Aux) timeIR(Aux) OPimpvol(Aux) Moneyness(Aux)];
    OptionInfo = sortrows(OptionInfo,[2 3]); 
    
    OptionInfo_ahead = [OptionPrice(Aux_ahead,:) time2exp(Aux_ahead,:) strike_price(Aux_ahead,:) dummy_put(Aux_ahead) timeIR(Aux_ahead) OPimpvol(Aux_ahead) Moneyness(Aux_ahead)];
    OptionInfo_ahead = sortrows(OptionInfo_ahead,[2 3]); 
            
    % CW train
    
    K_T_IV_matrix = [OptionInfo(:,3),OptionInfo(:,2)/252,OptionInfo(:,6)];
    
    CW = @(params) CW_fit(params,CurrentPrice,K_T_IV_matrix);
    
    [CW_params(i,:),resnorm] = lsqnonlin(CW,x0,lb,ub);   %,[],[],options
    
    % CW test
    
    CW_predict = CW_ivol_matrix([OptionInfo_ahead(:,3),OptionInfo_ahead(:,2)/252],CW_params(i,:),CurrentPrice_ahead);
     
    CW_results_aux = [ones(size(OptionInfo_ahead,1),1)*PossibleCurrentDates(i), OptionInfo_ahead,CW_predict'];
      
    CW_results = [CW_results; CW_results_aux];
    
    clear Aux OptionInfo train_index test_index OptionInfo_train OptionInfo_test X CW_predict CW_results_aux
    
    fprintf('Date %7.0f of %7.0f.\n',i,length(PossibleCurrentDates));
end

CW_results = CW_results(2:end,:);