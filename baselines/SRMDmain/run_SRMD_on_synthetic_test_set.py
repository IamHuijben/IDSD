import numpy as np
import matplotlib.pyplot as plt
from srmdpy import SRMD
from pathlib import Path
import scipy.io
import tqdm
import csv
from scipy.optimize import linear_sum_assignment

"""
Code to run SRMD on the synthetic test set and save the normalized L2 errors.
The SRMD model is published as:

Nicholas Richardson, Hayden Schaeffer, and Giang
Tran, “SRMD: Sparse Random Mode Decomposition,”
Communications on Applied Mathematics and Computation, vol. 6, no. 2, pp. 879–906, 6 2024.

The implementation belongs to the authors of the paper and can be found at: https://github.com/GiangTTran/SparseRandomModeDecomposition

"""
def normalized_l2_error(x,y):
    # x: estimated signal of shape: (*ndim, T)
    # y: ground truth signal of shape: (*ndim, T)
    l2error = np.linalg.norm(x - y, ord=2, axis=-1)
    return l2error / np.linalg.norm(y, ord=2, axis=-1) 


def reorder_signals(gt, pred):
    """
    Reorder predicted signals to match ground-truth signals based L2 distance.

    gt: (batch, K) np array of ground-truth signals
    pred: (batch, N, T) np array of predicted signals
    
    Returns:
        reordered_predictions: (batch, N, T) array of reordered predicted signals
    """
    batch, N, _ = gt.shape

    reordered_predictions = np.zeros_like(gt)
    order = np.zeros((batch, N), dtype=int)

    for i in range(batch):
        cost_matrix = np.linalg.norm(gt[i,:,None,:] - pred[i,None,:,:], axis=-1, ord=2)
        
        # Solve assignment problem to minimize total cost
        _, col_ind = linear_sum_assignment(cost_matrix)
        order[i] = col_ind

        # Now col_ind gives the best matching of predictions to gt
        reordered_predictions[i] = pred[i,col_ind]

    return reordered_predictions


if __name__ == "__main__":
    seed = 314
    K = 2
    test_path = Path.cwd() / f"data/synthetic_test_set_K{K}.mat"
    save_dir = Path.cwd() / "baselines" / "SRMDmain" / f"output_seed{seed}" / f"{test_path.stem}" 

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    data = scipy.io.loadmat(test_path)

    x = data['x'] # [nr_examples, 1, T] 
    y = data['y'] # [nr_examples, nr_modes, T]
    y = y[:, :K]  # Remove the noise ground-truth component
    settings = data['data_settings']  
    fs, T = int(settings['fs']), int(settings['T'])
    t = np.linspace(0, T, int(fs*T), endpoint=False)

    kwargs = {'eps':1, 'frq_scale':1, 'seed':seed, 'n_modes':K}
    norm_l2_errors = np.empty((x.shape[0], K))
    for idx in tqdm.tqdm(range(x.shape[0])):
        
        modes = SRMD(x[idx,0], t, **kwargs)
        modes = modes.transpose()
        
        modes = reorder_signals(y[idx:idx+1], modes[np.newaxis,:,:])
        norm_error = normalized_l2_error(modes, y[idx:idx+1])
        norm_l2_errors[idx] = norm_error[0]

    assert not np.isnan(norm_l2_errors).any()

    # # Store results
    np.save(save_dir / 'norm_l2_errors.npy', norm_l2_errors)

    error_stats = {
        'norm_l2_errorsK_mean': np.nanmean(norm_l2_errors),
        'norm_l2_errorsK_median': np.nanmedian(norm_l2_errors),
        'norm_l2_errorsK_std': np.nanstd(norm_l2_errors),
    }
    with open(str(save_dir / 'errors.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])
        for key, value in error_stats.items():
            writer.writerow([key, value])

    print('Done, results saved to:', save_dir)