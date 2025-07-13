function success_rate = multipletrial(m,d1,d2,r,kappa,trial_num,verbose,Xstar)
if nargin < 7
    verbose = 0; % Set default value for 'verbose' if not provided
end
if nargin < 8
    Xstar = groundtruth(d1,d2,r,kappa);
end


success_list = zeros(trial_num); 
%output PR transition graph
parfor i = 1:trial_num
    success_list(i) =  onetrial(m,d1,d2,r,Xstar,verbose);
end
success_rate = sum(success_list,"all") / trial_num;
   


end

