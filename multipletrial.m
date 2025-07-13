function err_avg = multipletrial(m,d1,d2,r,kappa,trial_num,verbose,Xstar)
if nargin < 7
    verbose = 0; % Set default value for 'verbose' if not provided
end
if nargin < 8
    Xstar = groundtruth(d1,d2,r,kappa);
end


err_list = zeros(trial_num); 
%output PR transition graph
parfor i = 1:trial_num
    err_list(i) =  onetrial_GD(m,d1,d2,r,Xstar,Initialization,verbose);
end
err_avg = mean(err_list);
end

