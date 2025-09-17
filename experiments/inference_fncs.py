import pickle
import numpy as np
import tqdm
import torch
from utils.helper_fncs import tensor2array
import csv
from matplotlib import pyplot as plt
from pathlib import Path
from torch import optim
from torch import nn
import copy 

def normalized_l2_error(x,y):
    # x: estimated signal of shape: (*ndim, N)
    # y: ground truth signal of shape: (*ndim, N)
    l2error = np.linalg.norm(x - y, ord=2, axis=-1)
    return l2error / np.linalg.norm(y, ord=2, axis=-1) 


def run_inference_synthetic_testset(model, save_dir=None, K=2):
    
    with open(str(Path.cwd() / f"data/synthetic_test_set_K{K}.pkl"), 'rb') as f:
        data = pickle.load(f)

    x = data['x']  # [nr_examples, 1, N] 
    y = data['y']  # [nr_examples, nr_modes, N]
    fs, T = model.fs, model.T

    save_dir = save_dir / f"synthetic_test_set_K{K}_inference_best_model"
    (save_dir).mkdir(exist_ok=True, parents=False)

    nr_batches = np.ceil(x.shape[0] / 1000)
    batches_x = np.array_split(x, nr_batches)
    batches_y = np.array_split(y, nr_batches)
    norm_l2_errors = []

    for x, y in tqdm.tqdm(zip(batches_x, batches_y)):
        modes, masks = model.predict(x, y, K=K)
        norm_errors_batch = normalized_l2_error(modes,y)
        norm_l2_errors.append(norm_errors_batch)

    norm_l2_errors = np.concatenate(norm_l2_errors, axis=0)
    np.save(save_dir / 'norm_l2_errors.npy', norm_l2_errors)

    # From last batch plot some representative examples 
    average_idxs = np.argsort(np.nanmean(norm_errors_batch[:,:-1],-1))[550:555] #Don't take into account the noise mode

    if masks is None: masks = [None]*x.shape[0] # The IRCNN model does not output masks
    for idx in average_idxs:
        make_plots(x[idx], y[idx], masks[idx], modes[idx], idx, fs, T, save_dir=save_dir)
    
    # Assuming here that the last mode is always the noise mode
    error_stats = {
        'norm_l2_errors_mean': np.mean(np.nanmean(norm_l2_errors[:,:-1],-1)),
        'norm_l2_errors_median': np.median(np.nanmean(norm_l2_errors[:,:-1],-1)),
        'norm_l2_errors_std': np.std(np.nanmean(norm_l2_errors[:,:-1],-1)),
    }
    with open(str(save_dir / 'errors.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Key', 'Value'])
        for key, value in error_stats.items():
            writer.writerow([key, value])


def run_inference_tsunami_data(model, max_nr_epochs_finetuning, lr_finetuning=1e-5, save_dir=None):
    """
    Run inference on tsunami data.
    """
    
    fs = 10/3600
    T = int(5*24*3600) # 5 days in seconds
    nr_samples = fs*T  # N 
    assert nr_samples % 2 == 0, "It has not yet been implemented to deal with an odd number of samples."
   
    # Determine the padding needed, such that the positive frequency spectrum is divisible by 2**4 (because of 4 pooling layers in u-net)
    nr_samples_with_pad = np.ceil(nr_samples / 2 / 2**4) * 2**4 * 2
    model.padL = (nr_samples_with_pad - nr_samples) / 2 # Divide by 2, as padding is done on both sides
    
    save_dir = save_dir / f"Tsunami_finetuned_{max_nr_epochs_finetuning}epochs_lr{lr_finetuning}_bs1024"
    (save_dir).mkdir(exist_ok=True, parents=False)

    # Load the test case tsunami data of 10 March 2011
    x = np.expand_dims(np.load(Path.cwd() / "data/Tsunami_10_14_March_measurement.npy"),0)
    x = x - np.mean(x,axis=-1, keepdims=True)

    y1 = np.expand_dims(np.load(Path.cwd() / "data/Tsunami_10_14_March_forecasting.npy"),0)
    y1 = y1 - np.mean(y1,axis=-1, keepdims=True)

    y2 = x - y1
    y = np.concatenate([y1, y2], axis=0)  # [nr_components, N]

    # Load finetuning data of 9 Feb - 9 March 2011
    data_file = Path.cwd() / 'data' / f'Tsunami_9_Feb_9_March_measurement.npy'

    x_fine = np.load(data_file)[np.newaxis,np.newaxis,:] #[bs, 1, N]
    x1_fine = np.load(data_file.parent / (data_file.stem[:-12] + '_forecasting.npy'))[np.newaxis,np.newaxis,:] #[bs,1,N] 

    x_fine = x_fine - np.mean(x_fine,axis=-1, keepdims=True) 
    x1_fine = x1_fine - np.mean(x1_fine,axis=-1, keepdims=True)

    x2_fine = x_fine - x1_fine
    y_fine = np.concatenate([x1_fine, x2_fine], axis=1)  # [bs, nr_components, N]

    model = finetune_model(model, x_fine, y_fine, lr=lr_finetuning, nr_epochs=max_nr_epochs_finetuning, Tsunami=True)
    modes, masks = model.predict(x, np.expand_dims(y,0) , K=1) # Set K=1 , as it will extract 1 mode + noise mode, where the 'noise mode' in this case is the Tsunami component.

    np.save(save_dir / 'pred_modes.npy', modes)
    np.save(save_dir / 'gt.npy', y)

    # Plot the results
    idx = 0  # only one signal
    make_plots(x, y, masks[idx], modes[idx], idx, fs, T, save_dir=save_dir)


def finetune_model(model, x, y, lr=1e-5, nr_epochs=100, Tsunami=False):
    model.train()
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

    prev_loss_ema = [np.inf]*5
    lowest_loss = np.inf
    for epoch in range(nr_epochs):
        x = x.to(device, non_blocking=False)
        y = y.to(device) 

        if Tsunami:
            bs = 1024
            # randomly sample windows of 1200 timestamps
            starts = np.random.uniform(0, x.shape[-1]-1200, size=bs).astype(int)
            x_batch = torch.zeros((bs, x.shape[1], 1200), device=device)
            y_batch = torch.zeros((bs, y.shape[1], 1200), device=device)

            for i in range(bs):
                x_batch[i] = x[...,starts[i]:int(starts[i]+1200)] 
                y_batch[i] = y[...,starts[i]:(starts[i]+1200)]

            x_batch = x_batch + torch.randn_like(x_batch) # Add some AWGN(0,1) to the input signal, which acts as a second component during training
            y_batch[:,0] = y_batch[:,0] - torch.mean(y_batch[:,0], dim=-1, keepdim=True) 
            y_batch[:,1] = x_batch[:,0] - y_batch[:,0]  
        else:
            x_batch = x
            y_batch = y

        optimizer.zero_grad()
        xF_split =  model.preprocessing(x_batch)
        yF_split = model.preprocessing(y_batch)
        modes, _ = model(xF_split, epoch=100) # Use epoch 100 to use the low sigmoid temperature (0.5) as used at the end of training
      
        total_loss = model.compute_loss(yF_split, modes)
        total_loss.backward()

        print(f"Epoch {epoch+1}/{nr_epochs}, Loss: {total_loss.item()}")
        optimizer.step()
        if total_loss.item() < lowest_loss:
            lowest_loss = total_loss.item()
            best_model = copy.deepcopy(model)

        prev_loss_ema.append(total_loss.item())
        prev_loss_ema = prev_loss_ema[-5:]  # Keep only the last 5 losses
        if abs(total_loss.item() - np.nanmean(prev_loss_ema)) / (np.nanmean(prev_loss_ema) + 1e-8) < 1e-3:
            if epoch > 10:
                print(f"Early stopping at epoch {epoch+1} due to convergence.")
                break

    best_model.eval()
    return best_model 


def make_plots(x, y, masks, pred_modes, signal_idx, fs, T, save_dir=None):
    """
    x: input signal of shape (1,T)
    y: ground truth modes of shape (nr_modes, T)
    masks: predicted masks of shape (nr_modes, T)
    pred_modes: predicted modes of shape (nr_modes, T)
    signal_idx: index of the signal in the dataset
    fs: sampling frequency
    T: duration of the signal
    """

    """ Plot the signal, and the predicted and ground truth modes """
    nr_modes = y.shape[0]
    fig, axs = plt.subplots(nr_modes+1, 1, figsize=(10,(nr_modes+1)*1.8), sharex=True) #, sharey=True)
    reconstruction = pred_modes.sum(0)
    
    axs[0].plot(np.linspace(0,T,int(T*fs)), x[0], label="Original",c='g', linewidth=1, linestyle='--')
    axs[0].plot(np.linspace(0,T,int(T*fs)), reconstruction, label=f"Reconstruction", c='k', linewidth=1,linestyle='-.')
    axs[0].legend()
    axs[0].set_title(f"Full signal - Normalized L2 error: {normalized_l2_error(x[0], reconstruction):.4f}", fontsize=8)

    for mode_nr in range(nr_modes):
        axs[mode_nr+1].plot(np.linspace(0,T, int(T*fs)), y[mode_nr], label=f"Mode {mode_nr+1}",c='g', linewidth=1, linestyle='-')
        axs[mode_nr+1].plot(np.linspace(0,T, int(T*fs)), pred_modes[mode_nr], label=f"Reconstruction", c='k', linewidth=1,linestyle='-.')
        axs[mode_nr+1].set_title(f"Mode {mode_nr+1} - Normalized L2 error: {normalized_l2_error(pred_modes[mode_nr], y[mode_nr]):.4f}", fontsize=8)

    if save_dir is not None:
        save_dir = save_dir / f"mixed_signal_and_modes_idx{signal_idx}.png"
        plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    

    """ Plot the masks and modes in frequency domain """
    fig, axs = plt.subplots(4, 1, figsize=(7,8))
    plt.subplots_adjust(hspace=0.6)

    # Plot the input signal 
    i = 0
    axs[i].plot(x[0],c='k');axs[i].set_xlabel('Time [s]');axs[i].grid(True);axs[i].set_title('Mixed signal');axs[i].set_xticklabels([])

    # Plot the spectrum of the ground truth modes
    i = 1
    spectrum = np.abs(np.fft.fft(x[0], axis=-1))
    spectrum_y = np.abs(np.fft.fft(y, axis=-1))
    ylabel = 'Magnitude [-]'
    half_spectrum = spectrum[...,:spectrum.shape[-1]//2]
    half_faxis = np.fft.fftfreq(int(fs*T),d=1/fs)[:spectrum.shape[-1]//2]
    
    axs[i].plot(half_faxis,half_spectrum, c='k', linestyle='dashed');axs[i].set_xlabel('frequency [Hz]');axs[i].grid(True);axs[i].set_ylabel(ylabel)
    
    for mode_idx in range(y.shape[0]):
        axs[i].plot(half_faxis, spectrum_y[mode_idx][...,:spectrum.shape[-1]//2], label=f'Ground truth mode {mode_idx+1}', alpha=0.8, linestyle='dotted') 
    axs[i].legend(loc='upper right', fontsize=8)

    # Plot the predicted masks
    if masks is not None:
        i = 2
        for mode_idx in range(masks.shape[0]):
            axs[i].plot(masks[mode_idx], alpha=0.8)
        axs[i].set_xticklabels([])
        axs[i].set_ylabel('Predicted masks');axs[i].set_ylim([-0.05,1.05]),axs[i].set_xlabel('frequency [Hz]');axs[i].grid(True)

    # Plot spectrum of predicted modes
    i = 3
    modes_pred_spectrum = np.abs(np.fft.fft(pred_modes, axis=-1))

    for mode_idx in range(pred_modes.shape[0]):
        axs[i].plot(half_faxis, modes_pred_spectrum[mode_idx,:spectrum.shape[-1]//2], alpha=0.8)
    axs[i].set_xlabel('frequency [Hz]');axs[i].grid(True);axs[i].set_ylabel('Magnitude [-]')

    if save_dir is not None:
        save_dir = save_dir.parent / f"masks_and_spectra_idx{signal_idx}.png"
        plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
