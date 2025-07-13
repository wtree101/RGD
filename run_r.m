% Script to run PhaseTransition, collect output w.r.t r, and save data

% Parameters for PhaseTransition (varying r)

d1 = 60; % Fixed parameter
d2 = 80; % Fixed parameter
trials = 20; % Fixed parameter
flag = 1; % Fixed parameter
kappa = 2; % Fixed parameter
parpool(16); % Start a parallel pool with 16 workers
r_values = 1:30; % Values of r to iterate over

% Preallocate storage for results
results = struct(); % Use a structure to store results

% Loop through each value of r
for i = 1:length(r_values) %not enough threads, so not parfor here
    r = r_values(i); % Current value of r
    
    % Display the parameter set being executed
    disp(['Running PhaseTransition with  r=', num2str(r), ', d1=', num2str(d1), ...
          ', d2=', num2str(d2), ', trials=', num2str(trials), ...
          ', kappa=', num2str(kappa), ...
          ', flag=', num2str(flag)]);
    
    % Start timing
    tic;
    
    % Run the PhaseTransition function and collect output
    output = PhaseTransition(d1, d2, r, kappa, trials, flag); % Assuming the function returns some output
    
    % Stop timing
    t = toc;
    disp(['Execution time: ', num2str(t), ' seconds']);
    
    % Store results in the structure
    results(i).r = r; % Store the current value of r
    results(i).output = output; % Store the output of PhaseTransition
    results(i).execution_time = t; % Store the execution time
end

% Save the results to a .mat file
output_file = sprintf('d1_%d_d2_%d_kappa_%d_trial_%d.mat', d1, d2, kappa,trials);
save(['results/',output_file], 'results');
disp(['Results saved to ', output_file]);

delete(gcp('nocreate'));