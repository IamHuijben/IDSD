clear all;
close all;

baseline = 'vmd' % one of 'ssd' or 'vmd'

if strcmp(baseline, 'ssd');
    addpath("baselines/matlab_code-SSD/");
end

addpath('baselines//npy-matlab-master');
savepath;

% Load signals
x  = readNPY(fullfile('data','Tsunami_10_14_March_measurement.npy'));
x = x - mean(x);

y1 = readNPY(fullfile('data','Tsunami_10_14_March_forecasting.npy'));
y1 = y1 - mean(y1);
y2 = x - y1;

fs = 10/3600;  % sampling frequency: 10 samples per hour

% 
if strcmp(baseline, 'ssd');
    max_nr_comp = 2;
    u = SSD(x, fs, 0.01, max_nr_comp);  
elseif strcmp(baseline, 'vmd');
    % Set alpha for VMD
    alpha = 10000 %, 0.1 1, 10, 100, 1000, 10000
    [u,~] = vmd(x,NumIMFs=2, PenaltyFactor=alpha);
    u = u';
end;

% Reorder modes to best match the ground truth
ground_truth = [transpose(y1); transpose(y2)];     % 2 × N
predictions  = u;                                  % 2 × N

[reordered_modes, ground_truth, order] = reorder_signals_matlab(ground_truth, predictions);
num_modes = size(reordered_modes,1); 

% Compute normalized L2 error
normL2errors = nan(num_modes,1);
for k = 1:num_modes;
    normL2errors(k) = norm(reordered_modes(k,:) - ground_truth(k,:),2) / norm(ground_truth(k,:),2);
end

% Plot
N = numel(x);
t = linspace(10,15, N);

figure('Position',[200 200 600 1200])

% Full signal
subplot(num_modes+1,1, 1)
plot(t, x,'g','DisplayName','True')
hold on
plot(t, sum(u,1),'k--','LineWidth',1,'DisplayName','Prediction')
title(sprintf('Full signal\nnorm L2 err = %.4f',norm(sum(u,1)-transpose(x),2)/norm(transpose(x),2)))
legend('Location','best')
xticks(10:15);
hold off

% Individual modes
for k = 1:num_modes
    subplot(num_modes+1,1, k+1)
    hold on
    plot(t, ground_truth(k,:), 'g','LineWidth',1)
    normerr = normL2errors(k);
    title(sprintf('Mode %d\nnorm L2 error=%.4f', k, normerr))
    plot(t, reordered_modes(k,:), 'k--','LineWidth',1)
    xticks(10:15);
    hold off
end
if k == num_modes; xlabel('Day in March 2011'); end;

sgtitle('Mode decomposition of tsunami data from 10 to 14 March')

currentFile   = matlab.desktop.editor.getActiveFilename;
save_path = fileparts(currentFile) + "/" + baseline;

if strcmp(baseline, 'ssd');
    save(strcat(save_path,'/tsunami_2011_pred_modes.mat'),'reordered_modes');
elseif strcmp(baseline, 'vmd');
    save(strcat(save_path, '/tsunami_2011_pred_modes_alpha', int2str(alpha),'.mat'),'reordered_modes');
end;