function [reordered_predictions, ground_truth, order] = reorder_signals_matlab(ground_truth, predictions)
    % Reorder predicted signals to match ground-truth signals based on L2 norm in time domain.
    %
    % Inputs:
    %   ground_truth: [batch, K, N] array of ground-truth signals
    %   predictions:  [batch, K, N] array of predicted signals
    %
    % Outputs:
    %   reordered_predictions: [batch, K, N] reordered predicted signals
    %   ground_truth_out: potentially modified ground truth
    %   order: [batch, K] array of index mappings
    
    arguments
        ground_truth (:,:,:) double
        predictions  (:,:,:) double
    end

    [K, N] = size(ground_truth);
    assert(size(predictions,1) <= K);
    if size(predictions,1) < K;
        num_missing = K - size(predictions, 1);
        zero_pad = zeros(num_missing, N);  % Create zeros of shape [missing, N]
        predictions = [predictions; zero_pad];  % Concatenate along rows
    end
    
    cost_matrix = zeros(K, K);

    for i = 1:K;
        for j = 1:K;
            diff = ground_truth(j, :) - predictions(i, :);
            cost_matrix(i, j) = norm(diff, 2);
        end
    end

    % Use Hungarian algorithm to solve assignment 
    ordering = matchpairs(cost_matrix, 1e6);
    col_ind = ordering(:,1);

    order = col_ind;
    reordered_predictions = predictions(col_ind, :);


