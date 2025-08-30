function [err_list,success_list, err_list_f] = multipletrial(m,r,kappa,lambda,params)
    % Extract parameters from the struct 'params'

  
    trial_num = params.trial_num;

    if isfield(params, 'T')
        T = params.T;
    else
        T = 200;
    end
   
    onetrial_alg = params.alg; 

    err_list = zeros(T,1);
    success_list = zeros(T,1); 
    err_list_f = zeros(T,1);
    %output PR transition graph
    parfor i = 1:trial_num
        [output_list, output_listf] = onetrial_alg(m,r,kappa,lambda,params); 
        err_list = err_list + output_list;
        err_list_f = err_list_f + output_listf;
        success_list = success_list + (output_list < lambda*10); % Count successes
    end

    err_list = err_list / trial_num; % Average error over trials
    success_list = success_list / trial_num; % Average success rate over trials
    err_list_f = err_list_f / trial_num; % Average error over trials
end

