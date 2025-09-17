import numpy as np
from scipy.io import savemat
import torch
from data.signal_generator import SignalDecompositionGenerator
from torch.utils.data import DataLoader
from utils.helper_fncs import fix_random_seed
from datetime import date
import pickle
from pathlib import Path

if __name__ == "__main__":

    seed = 1
    fix_random_seed(seed)

    for set_idx, signal_types in enumerate([['self.generate_FM_sinusoids_broadband']]):
        g = torch.Generator().manual_seed(seed)
        fs = 1024
        K = 5
        data_settings = {'min_freq': 1., 
                        'max_freq': fs/2-1, 
                        'fs': fs, 
                        'a_min':0.1, 
                        'a_max':1., 
                        'T': 1., 
                        'nr_components': K,
                        'noise_sigma': 0.1, 
                        'signal_types':signal_types}  
        
        data_set = SignalDecompositionGenerator(**data_settings)
        batch_size = 1000
        train_loader = DataLoader(data_set, batch_size=batch_size, drop_last=True, pin_memory=True,num_workers= 0, generator=g)

        max_samples = 10000
        i = 0
        while i * batch_size < max_samples:  
            for j, (x, y) in enumerate(train_loader):

                x = x.numpy()
                y = y.numpy()
                if i == 0 and j == 0:
                    X = x
                    Y = y
                else:
                    X = np.concatenate((X, x), axis=0)
                    Y = np.concatenate((Y, y), axis=0)
            i += 1
        
        data = {
            'x': X,
            'y': Y,
            'data_settings': data_settings,
        }

        save_path = Path.cwd() / 'data'
        save_name = f'synthetic_test_set_K{K}'

        savemat(str(save_path / f'{save_name}.mat'), data)

        with open(str(save_path / f'{save_name}.pkl'), 'wb') as f:
            pickle.dump(data, f)
