clear all;
close all;

baseline = 'ssd' %Choose from emd or ssd

if strcmp(baseline, 'ssd');
    addpath("baselines/matlab_code-SSD/");
end

% Read the synthetic test sets with K=2 or K=5 components and run through
% SSD or EMD.
% The normalized L2 errors are stored in this folder.
for K=[2,5];
    currentFile   = matlab.desktop.editor.getActiveFilename;
    code_folder = fileparts(fileparts(currentFile));
    dataFile = fullfile(code_folder, 'data', sprintf('synthetic_test_set_K%d.mat',K))
    data = load(dataFile);
    fs = 1024;
    num_signals = size(data.x,1);

    all_normL2errors = nan(num_signals, K,1);

    if strcmp(baseline, 'ssd');
        cfg = mex.getCompilerConfigurations('C++', 'Selected');
        if isempty(cfg);
            error('No C++ compiler configured. Please run "mex -setup cpp" once before using this function.');
        end
    end

    for i=1:num_signals;
        x = squeeze(data.x(i,1,:));

        if strcmp(baseline, 'ssd');
            predictions = SSD(x, fs, [], K);  % K × T    
        elseif strcmp(baseline, 'emd');
            [predictions,~] = emd(x,'MaxNumIMF',K);
            predictions = predictions';
        end;

        % Assemble ground truth and predictions into (K × N) arrays
        ground_truth = squeeze(data.y(i,1:end-1,:));        % K × N. Remove the noise component from GT.

        [reordered_modes, ground_truth, order] = reorder_signals_matlab(ground_truth, predictions);
        num_modes = size(reordered_modes,1); 

        % Compute errors
        normL2errors = nan(num_modes,1);
        for k = 1:num_modes;
            l2error = norm(reordered_modes(k,:) - ground_truth(k,:),2);
            normL2errors(k) = l2error / norm(ground_truth(k,:),2);
        end

        all_normL2errors(i,:,:) = normL2errors;

    end
    folderPath = fullfile('baselines', baseline);  
    mkdir(folderPath)
    [~, stem, ~] = fileparts(dataFile);
    save_path = folderPath + "/" + stem;
    save(save_path + "_normL2errors.mat", "all_normL2errors");

end;



