%%%%%%%%%%
d1 = 60; d2 = 80;
kappa = 2;
trial_num = 20; verbose = 1;
add_flag = 0;

% Parameters
%Max_scale = 14;

r_max = 20; % Adjust r range as needed (e.g., r in [0, 1])
approx = (d1 + d2)*r_max*6;

Max_scale = round(log2(approx));

scale_num = 4; %3 8 points 4 17 points
m_max = 2^Max_scale; % 16384
data_file = sprintf('success_rate_data_d1_%d_d2_%d_rmax_%d_kappa_%d', d1, d2, r_max, kappa);
full_path = fullfile('data2', data_file);
if ~exist(full_path, 'dir')
    mkdir(full_path);
end
dist = ['data2/',data_file,'/2/'];
if ~exist(dist, 'dir')
    mkdir(dist);
end

%%%%%%%%%%%%%%%%%%%%%%

% Grid levels (coarse to fine)
levels = Max_scale-1:-1:Max_scale-scale_num;
%levels = [Max_scale-1, Max_scale-2, Max_scale-3, Max_scale-4]; % Start at 2^13, go down to 2^10 (adjust as needed)
m_grids = cell(length(levels), 1);


for i = 1:length(levels)
    n = levels(i);
    scale_gap = 2^n; % e.g., 8193, 4097, 2049, 1025
    m_grids{i} = 0:scale_gap:m_max;
  
end
m_all = [];

for i = 1:length(levels)
    m_current = m_grids{i};    
    % Exclude points already computed (from higher levels)
    if i == 1
        m_all = [];
    else
        % Check for uniqueness (points not in previous levels)
        
        new_coords = setdiff(m_current, m_all);
        m_all = [m_all, new_coords];
      
    end
end

% Total unique m
m_all = m_all(m_all ~= 0);
points_num = length(m_all);
disp(['Total unique points to compute: ', num2str(points_num)]);

save([full_path,'/mgrid.mat'],"m_all")

parpool(16);
%%%%%%%%%%%%%%%%%%%%%%

for r = 1:r_max
    points_r = cell(points_num, 1);
     % Display the parameters being used for debugging
    disp(['Running PhaseTransition with ', ...
          'r = ', num2str(r), ...
          ', d1 = ', num2str(d1), ...
          ', d2 = ', num2str(d2), ...
          ', trials = ', num2str(trial_num), ...
          ', kappa = ', num2str(kappa)]);
    tic;
    
    parfor i = 1:points_num
        m = m_all(i);
        %Xstar = groundtruth(d1,d2,r,kappa);
        p = multipletrial(m,d1,d2,r,kappa,trial_num,0);
        points = struct();
        points.r = r; points.m = m; points.p = p;
        %points.Xstar = Xstar; 
        points.trial_num = trial_num;
        
        points_r{i} = points;
        %save the data for graph
         
    end
     t = toc;
        disp(['Execution time: ', num2str(t), ' seconds']);
    for i = 1:points_num
        point = points_r{i};
        point_name = sprintf('r_%d_m_%d_t_%d.mat', point.r, point.m, point.trial_num);
        if add_flag==0
            save([dist, point_name], "point");
             disp(['Save ', point_name])
        else
                   % Load existing point data
            old_data = load([dist, point_name], 'point');
            data_old = old_data.point;
            
            % Update probability: weighted average
            p_new = (data_old.p * data_old.trial_num + point.p * point.trial_num) / ...
                    (data_old.trial_num + point.trial_num);
            
            % Update the point structure
            point.p = p_new;
            point.trial_num = data_old.trial_num + point.trial_num; % Combine trial counts
            
            % Save updated point back to the same file
            save([dist, point_name], 'point');
            disp(['Updated ', point_name, ' with p = ', num2str(p_new)]);
        end
       
    end

end

delete(gcp('nocreate'));

%%%%%%%%%%%%%%%%%




