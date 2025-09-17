import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date

from utils.helper_fncs import get_git_branch, get_git_commit_hash, make_unique_path
from utils.helper_fncs import prepare_dict_for_yaml
from utils.config import Config

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from baselines.IRCNNplus_pytorch_implementation.model import IRCNN_plus
from torch.utils.data import DataLoader
from data.signal_generator import SignalDecompositionGenerator
from baselines.IRCNNplus_pytorch_implementation.their_data_generator import IRCNN_datagen
from utils.helper_fncs import fix_random_seed, set_device
from utils.helper_fncs import tensor2array
from callbacks.training import SaveModel,EarlyStopper
from utils.config import Config, load_config_from_yaml
from experiments.inference_fncs import run_inference_synthetic_testset
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse 
import csv
"""
Train IRCNN+ model 
"""
def train_fnc(config, model, train_loader, val_loader, device=None, callbacks=[]):
    epochs = int(config.training.n_epochs)
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)

    scheduler = None
    if config.training.get('scheduler',{'type':None}).get('type') == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold_mode='abs', **config.training.scheduler.kwargs)   
        
    setattr(model, 'metrics_dict', {})  
    model.metrics_dict['loss'] = {'train': np.zeros((epochs,))*np.nan,'val': np.zeros((epochs,))*np.nan}    

    stop_training = False

    loss_dict = {'train':np.zeros(epochs), 'val':np.zeros(epochs)}
    for epoch in range(epochs):
        if not stop_training:

            model.train()
            for x, y in train_loader:
                x = x.to(device, non_blocking=False)
                y = y.to(device) 
                optimizer.zero_grad()

                modes = model(x) 

                train_loss = model.compute_loss(y, modes)
                train_loss.backward()
                optimizer.step()
                loss_dict['train'][epoch] += train_loss.item()

            loss_dict['train'][epoch] /= len(train_loader)

            ### Validation loop ###
            model.eval()

            for x,y in val_loader:
                x = x.to(device, non_blocking=False)
                y = y.to(device)

                with torch.no_grad():
                    modes = model(x)

                    val_loss = model.compute_loss(y, modes)
                    loss_dict['val'][epoch] += val_loss.item()
            loss_dict['val'][epoch] /= len(train_loader)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss_dict['train'][epoch]:.4f}, Validation Loss: {loss_dict['val'][epoch]:.4f}")

            
            # Step the scheduler
            if scheduler is not None:
                prev_lr = scheduler._last_lr[0]
                scheduler.step(metrics=loss_dict['val'][epoch])
                curr_lr = scheduler._last_lr[0]

                if curr_lr < prev_lr: print(f"[INFO] Learning rate reduced from {prev_lr:.4e} to {curr_lr:.4e}")

            for cb in callbacks:
                if cb.__class__.__name__ == 'SaveModel':            
                    cb.on_epoch_end(state_dict=model.state_dict(), 
                                loss_dict=loss_dict, 
                                epoch=epoch, 
                                optimizer_state_dict=optimizer.state_dict(), 
                                scheduler_state_dict=scheduler.state_dict() if scheduler else None)
                else:
                    stop_training = cb.on_epoch_end(loss_dict=loss_dict, epoch=epoch)
 
        else:
            print(f"Early stopping at epoch {epoch+1}")
            break
    return model

if __name__ == "__main__":

    ### Settings ###
    seed = 1
    fix_random_seed(seed)
    device = set_device() 
    
    # Model and training settings
    K = 2  #nr of components (the original code was only made for K=2 components)
    ks1 = 32  #Half kernel size for the first part of the model
    ks2 = 32  #Half kernel size for the second part of the model

    data_gen_type = 'SignalDecompositionGenerator' #IRCNN_datagen or SignalDecompositionGenerator

    "===============  CREATE DATALOADER  ==============="

    # Train models with different values for S: the number of conv layers in each part of the model
    for S in [3,4,5,6]: 

        if data_gen_type == 'SignalDecompositionGenerator':
            fs = 1024
            T =  1
            data_gen_settings = {'min_freq': 1, 
                                'max_freq': fs/2 - 1, 
                                'padL':0,  # the padL parameter here is only used to determine the signal length, the actual padding is done in the model.
                                'fs': fs, 
                                'a_min':0.1, 
                                'a_max':1, 
                                'T': T, 
                                'nr_components': 'max2',  #This is different from my experiments, but IRCNN+ can only deal with 2 components 
                                'noise_sigma': 'max0.2', 
                                'signal_types':['self.generate_FM_sinusoids_broadband'],
                                'data_gen_type': data_gen_type} 
            #Use the same settings as used for training IDSD.
            batch_size = 256
            n_epochs = 500
            lrscheduler = {
                    'type': 'ReduceLROnPlateau',      
                    'kwargs': {
                        'factor': 0.1,
                        'patience': 20, #Set lower than the early stopping patience
                        'threshold': 1e-4, #Relative threshold for measuring the new optimum, to only focus on significant changes.
                    }
                }  
        else:
            fs = 400 
            T =  6
            data_gen_settings = {
                'fs':fs,
                'T':T,
                'data_gen_type': data_gen_type
            }
            #Use their settings
            batch_size = 8 
            n_epochs = 60
            lrscheduler = {'type': None}

        g = torch.Generator().manual_seed(seed)
        if data_gen_settings['data_gen_type'] == 'IRCNN_datagen':
            train_set = IRCNN_datagen(data_fold='train')
            val_set = IRCNN_datagen(data_fold='test')
        elif data_gen_settings['data_gen_type'] == 'SignalDecompositionGenerator':
            train_set = SignalDecompositionGenerator(**data_gen_settings)
            val_set = SignalDecompositionGenerator(**data_gen_settings)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,pin_memory=True,num_workers= 0, generator=g) 
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True,pin_memory=True,num_workers= 0)

    
        "=============== DEFINE MODEL & SETTINGS =============== "


        run_name = f"K{K}_ks1_{ks1}_ks2_{ks2}_S{S}_bs{batch_size}_{data_gen_type}_sortYtrue_s{seed}" #This is the name of the run, which will be used for saving the model.

        today = date.today()
        save_dir = make_unique_path(Path(__file__).resolve().parent / "outputs" / "IRCNNplus" / (f"{str(today.year)[-2:]}{today.month:02d}{today.day:02d}_" +run_name))
        
        run_name = save_dir.stem

        config = {
            'data_gen_settings': data_gen_settings,
            'model': {
                'model': 'IRCNNplus',
                'nr_modes': K,
                'half_kernel_size_part1': ks1, #Half kernel size for the first part of the model
                'half_kernel_size_part2': ks2, #Half kernel size for the second part of the model
                'S': S, #12 #Number of layers in each part of the model
            },
            'training': {
                'seed': seed,
                'batch_size': batch_size,
                'n_epochs': n_epochs, 
                'lr': 0.001, 
                'scheduler': lrscheduler,
            },
            'commit_hash': f"{get_git_branch()} - {get_git_commit_hash()}",
            'logging': str(save_dir),
            'callbacks': {
                'SaveModelCallback': {'every_n_epochs': 0, 'save_last':True, 'log_dir': str(save_dir)},
                'PlotMetrics': {'every_n_epochs':1}, 
                'EarlyStopper': { 
                    'patience': 30, 
                    'stop_at_minimum': True, # Whether to track a minimum or a maximum, dependent on the metric that is being tracked.
                    'min_delta': 1e-4, # Minimum abs. change in the monitored quantity to qualify as an improvement.
                },
            }
        }
        config = Config(prepare_dict_for_yaml(config))
        save_dir = Path(config['logging']).resolve()

        if save_dir is not None:
            config.save_to_yaml(save_dir / 'config.yml')

    
        model = IRCNN_plus(length=int(T*fs),fs=fs, T=T, half_kernel_size_part1=ks1, half_kernel_size_part2=ks2,S=int(config.model.get('S', 12)))
        model = model.to(device)

        "=============== CALLBACKS  ==============="
        save_model_cb = SaveModel(nr_epochs = config.training.n_epochs,
                            data_class = train_set.__dict__, 
                            every_n_epochs = config.callbacks.SaveModelCallback.every_n_epochs,
                            save_last= config.callbacks.SaveModelCallback.save_last,
                            log_dir=save_dir)
        
        early_stopper_cb = EarlyStopper(
                        data_class = train_set.__dict__,
                        nr_epochs = config.training.n_epochs, 
                        patience=config.callbacks.EarlyStopper.patience, 
                        min_delta=config.callbacks.EarlyStopper.get('min_delta', 0),
                        stop_at_minimum=config.callbacks.EarlyStopper.stop_at_minimum) 
        callbacks = [save_model_cb, early_stopper_cb]
        
        print('Callbacks initialized!')

        "=============== TRAINING ==============="

        model = train_fnc(config, model, train_loader=train_loader, val_loader=val_loader, device=device, callbacks=callbacks)

        "=============== INFERENCE ==============="
        model_path = save_dir / 'checkpoints/best_model.pt'
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        config = load_config_from_yaml(model_path.parent.parent / "config.yml")
        fs,T = int(config.data_gen_settings.fs), int(config.data_gen_settings.T)
        model = IRCNN_plus(length=int(T*fs),fs=fs, T=T, half_kernel_size_part1=ks1, half_kernel_size_part2=ks2,S=int(config.model.get('S', 12)))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        save_dir = model_path.parent.parent / f"inference" 
        (save_dir).mkdir(exist_ok=True, parents=True)

        if data_gen_settings['data_gen_type'] == 'IRCNN_datagen':
            ### Run inference on their test set (=val_loader) to benchmark results ###
            with torch.no_grad():
                for i, (x,y) in enumerate(val_loader):
                    x = x.to(device)
                    outputs = model(x)

            pred = tensor2array(outputs.reshape(outputs.shape[0], 2, -1))
            if data_gen_settings['data_gen_type'] == 'IRCNN_datagen':
                x = tensor2array(x[...,1200:-1200]) #unpad
            else:
                x = tensor2array(x)

            N = x.shape[0]
            # print(x.shape, y.shape, pred.shape)
            
            pred = model.reorder_signals(y, pred)

            yr = y.reshape((N,-1))
            predr = pred.reshape((N,-1))
            rmse_testset = np.sqrt(mse(yr,predr))
            mae_testset = mae(yr, predr)

            print(f'rmse: {rmse_testset:.4f}, mae: {mae_testset:.4f}')

            with open(str(save_dir / 'test_set_errors.csv'), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Key', 'Value'])
                writer.writerow(['rmse', rmse_testset])
                writer.writerow(['mae', mae_testset])


            fig, axs = plt.subplots(8,3, figsize=(10,10), sharex=True)
            axs[0,0].set_title('Full signal');axs[0,1].set_title('Mode 1');axs[0,2].set_title('Mode 2')
            for idx in range(8):
                axs[idx,0].plot(tensor2array(x)[idx,0], c='k')
                axs[idx,1].plot(y[idx,0], label='y1');axs[idx,1].plot(pred[idx,0],label='pred 1',alpha=0.7);axs[7,1].legend() 
                axs[idx,2].plot(y[idx,1],label='y2');axs[idx,2].plot(pred[idx,1],label='pred 2',alpha=0.7);axs[7,2].legend()
                axs[idx,0].set_ylabel(f'Signal {idx}')
            plt.savefig(save_dir / 'inference_test_set_examples.png')

        else:
            run_inference_synthetic_testset(model, save_dir=save_dir, K=2)

        print('Inference done')

