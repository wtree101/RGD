% Load data
dist = 'data2/success_rate_data_d1_60_d2_80_rmax_20_kappa_2/';
data_dir = [dist,'2']; % Replace with your path
m_all = load([dist,'mgrid.mat']);
m_all = sort(m_all.m_all);
r_max = 20;
r_grid = 1:r_max;

file_list = dir(fullfile(data_dir, '*.mat'));
P = zeros(r_max,length(m_all));
m_map = containers.Map(m_all, 1:length(m_all));


for i = 1:length(file_list)
    filename = file_list(i).name;
    tokens = regexp(filename, 'r_(\d+)_m_(\d+)_t_\d+\.mat', 'tokens');
    if ~isempty(tokens)
        
        data = load(fullfile(data_dir, filename));
        p = data.point.p;
        r = data.point.r;
        m = data.point.m;
        if r <= r_max && isKey(m_map, m)
            m_idx = m_map(m);
            P(r, m_idx) = p;
        end
    end
end

% Plot heat-map
P = P';

% Create an invisible figure
fig = figure('Visible', 'off');
imagesc(r_grid, m_all, P);
set(gca, 'YDir', 'normal');
colorbar;
colormap('gray'); % Black (0) to white (1)
xlabel('r');
ylabel('m');
title('Heat-map of Success Probability');

% Save the figure quietly
saveas(fig, [dist,'/success_probability_heatmap.png']);

% Close the figure
close(fig);