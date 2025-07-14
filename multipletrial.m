function err_list = multipletrial(m,d1,d2,r,kappa,trial_num,verbose,Xstar,init_flag,T)
if nargin < 7
    verbose = 0; % Set default value for 'verbose' if not provided
end
if nargin < 8
    Xstar = groundtruth(d1,d2,r,kappa);
end
if nargin < 9
    init_flag = 1;
end
if nargin < 10
    T = 200;
end

err_list = zeros(T,1); 
%output PR transition graph
parfor i = 1:trial_num
    err_list = err_list + onetrial_GD(m,d1,d2,r,Xstar,verbose,0.2,init_flag,T);
end

err_list = err_list / trial_num; % Average error over trials
%err_avg = mean(err_list);
end

