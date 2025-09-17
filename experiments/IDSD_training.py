import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data.signal_generator import SignalDecompositionGenerator
from datetime import date

from utils.helper_fncs import get_git_branch, get_git_commit_hash, fix_random_seed, set_device, make_unique_path
from utils.helper_fncs import prepare_dict_for_yaml
from utils.config import Config, load_config_from_yaml
from callbacks.training import SaveModel,EarlyStopper
import yaml
from experiments.model import IDSD
import copy
from experiments.inference_fncs import run_inference_synthetic_testset, run_inference_tsunami_data

"=============== TRAINING LOOP =============== "
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

                xF_split = model.preprocessing(x)
                yF_split = model.preprocessing(y)
                modes, _ = model(xF_split, epoch) # modes has shape[batch_size, 2, nr_modes, N/2]

                train_loss = model.compute_loss(yF_split, modes)
                train_loss.backward()
                optimizer.step()
                loss_dict['train'][epoch] += train_loss.item()

            loss_dict['train'][epoch] /= len(train_loader)

            ### Validation loop ###
            model.eval()

            for x,y in val_loader:
                x = x.to(device, non_blocking=False)
                y = y.to(device)
                xF_split = model.preprocessing(x)
                yF_split = model.preprocessing(y)

                with torch.no_grad():
                    modes, _ = model(xF_split, epoch)

                    val_loss = model.compute_loss(yF_split, modes)
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
    for seed in [1,2,3]:

        "===============  CREATE DATALOADER ==============="
        fix_random_seed(seed)
        device = set_device()

        batch_size = 256
        fs = 1024  # sampling frequency in Hz
        T =  1 # length of signal in seconds

        g = torch.Generator().manual_seed(seed)
        data_gen_settings = {'min_freq': 1, 
                                'max_freq': fs/2 - 1, 
                                'padL':0,  # the padL parameter here is only used to determine the signal length, the actual padding is done in the model with 2*padL (padding left and right of signal)
                                'fs': fs, 
                                'a_min':0.1, 
                                'a_max':1, 
                                'T': T, 
                                'nr_components': 'max5', 
                                'noise_sigma': 'max0.2',  
                                'signal_types':['self.generate_FM_sinusoids_broadband']} 
        train_set = SignalDecompositionGenerator(**data_gen_settings)
        val_set = SignalDecompositionGenerator(**data_gen_settings)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,pin_memory=True,num_workers= 0, generator=g)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True,pin_memory=True,num_workers= 0)

        "=============== DEFINE MODEL & SETTINGS =============== "
        start_nr_chs = 128  # Number of channels in the first layer of the u-net.
        maxch = 128         # Maximum number of channels in the u-net.
        kernel_size = 15    # Kernel size of convolutions in the u-net.

        run_name = f"base_model_bs{batch_size}_s{seed}" #This is the name of the run, which will be used for saving the model.

        today = date.today()
        save_dir = make_unique_path(Path(__file__).resolve().parent / "trained_models" / "IDSD" / (f"{str(today.year)[-2:]}{today.month:02d}{today.day:02d}_" +run_name))

        run_name = save_dir.stem
        
        config = {
            'data_gen_settings': data_gen_settings,
            'model': {
                'model': 'IDSD',
                'kernel_size': kernel_size,
                'padL':data_gen_settings['padL'],
                'start_nr_chs':start_nr_chs,
                'maxch': maxch,
                'bias':False,
                'sigmoid_temperature': {'start':1.0, 'end':0.5, 'nr_iters':50},
            },
            'training': {
                'seed': seed,
                'batch_size': batch_size,
                'n_epochs': 500,
                'lr': 0.0001, 
                'scheduler': {
                    'type': 'ReduceLROnPlateau',      
                    'kwargs': {
                        'factor': 0.1,
                        'patience': 20, #Set lower than the early stopping patience
                        'threshold': 1e-4, #Relative threshold for measuring the new optimum, to only focus on significant changes.
                    }
                }  
            },
            'commit_hash': f"{get_git_branch()} - {get_git_commit_hash()}",
            'logging': str(save_dir),
            'callbacks': {
                'SaveModelCallback': {'every_n_epochs': 0, 'save_last':True, 'log_dir': str(save_dir)},
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


        model = eval(config.model.model)(fs=fs, T=T, **config.model)
        model = model.to(device)


        "=============== CALLBACKS  ==============="
        save_model_cb = SaveModel(nr_epochs = config.training.n_epochs,
                            every_n_epochs = config.callbacks.SaveModelCallback.every_n_epochs,
                            save_last= config.callbacks.SaveModelCallback.save_last,
                            log_dir=save_dir)
        
        early_stopper_cb = EarlyStopper(
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
        model = eval(config.model.model)(fs=fs, T=T, **config.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        save_dir = model_path.parent.parent / f"inference" 
        (save_dir).mkdir(exist_ok=True, parents=True)
        model.eval()

        run_inference_synthetic_testset(model, save_dir=save_dir, K=2)
        run_inference_synthetic_testset(model, save_dir=save_dir, K=3)
        run_inference_tsunami_data(copy.deepcopy(model), max_nr_epochs_finetuning=100, lr_finetuning=1e-5, save_dir=save_dir)

        print(f"All done! Results are saved in {save_dir}")