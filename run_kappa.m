% Script to run PhaseTransition, collect output w.r.t r, and save data

% Parameters for PhaseTransition (varying r)

d1 = 20; % Fixed parameter
d2 = 15; % Fixed parameter
trials = 50; % Fixed parameter
flag = 1; % Fixed parameter
r = 3;
parpool(8);
kappa_values = 1:5:40; % Values of r to iterate over

% Preallocate storage for results
results = struct(); % Use a structure to store results

% Loop through each value of ka[[a
for i = 1:length(kappa_values)
    kappa = kappa_values(i); % Current value of r
    
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
    results(i).kappa = kappa; % Store the current value of r
    results(i).output = output; % Store the output of PhaseTransition
    results(i).execution_time = t; % Store the execution time
end

% Save the results to a .mat file
output_file = sprintf('d1_%d_d2_%d_r_%d_trial_%d.mat', d1, d2, r,trials);
save(['results/',output_file], 'results');
disp(['Results saved to ', output_file]);
delete(gcp('nocreate'));