clear all;
close all;

baseline = 'svmd' % Choose from vmd or svmd

if strcmp(baseline, 'svmd');
    addpath("baselines/matlab_code-SSD/SVMD");
end

% Read the synthetic test sets with K=2 or K=3 components and run through
% VMD or SVMD with a range of alphas (for VMD) or max alphas (for SVMD)
% The normalized L2 errors are stored in this folder.
for K=[2,5];
    currentFile   = matlab.desktop.editor.getActiveFilename;
    code_folder = fileparts(fileparts(currentFile));
    dataFile = fullfile(code_folder, 'data', sprintf('synthetic_test_set_K%d.mat',K))
    data = load(dataFile);
    fs = 1024;
    num_signals = size(data.x,1);
    
    if strcmp(baseline, 'vmd');
        alpha_values = [0.1, 1, 10, 100, 1000, 10000];
    elseif strcmp(baseline, 'svmd');
        alpha_values = [100, 1000, 10000];
    end


    all_normL2errors = nan(num_signals, length(alpha_values), K,1);

    for a=1:length(alpha_values);
        alpha = alpha_values(a);

        for i=1:num_signals;
            x = squeeze(data.x(i,1,:));

            % Assemble ground truth and predictions into (K × N) arrays
            ground_truth = squeeze(data.y(i,1:end-1,:));        % K × N. Remove the noise component from GT.
            assert(size(ground_truth,1) == K);

            if strcmp(baseline, 'vmd');
                [imf,~] = vmd(x,NumIMFs=K, PenaltyFactor=alpha);
            elseif strcmp(baseline, 'svmd');
                maxAlpha=alpha; 
                tau=0;       
                tol =1e-6;
                nrIMFs=K;
                [imf,~]=svmd(double(x'),maxAlpha, tau,tol,[],nrIMFs);
                imf = imf';
            end

            [reordered_modes, ground_truth, order] = reorder_signals_matlab(ground_truth, imf');

            % Compute errors
            normL2errors = nan(K,1);
            for k = 1:K
                l2error = norm(reordered_modes(k,:) - ground_truth(k,:),2);
                normL2errors(k) = l2error / norm(ground_truth(k,:),2);
            end

            all_normL2errors(i,a,:,:) = normL2errors;

        end

    end
    folderPath = fullfile('baselines', baseline);  
    mkdir(folderPath)
    [~, stem, ~] = fileparts(dataFile);
    save_path = folderPath + "/" + stem;
    save(save_path + "_normL2errors.mat", "all_normL2errors");
    save(save_path + "_alpha_values.mat", "alpha_values")
    
end;
