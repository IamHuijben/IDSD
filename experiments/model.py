import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from experiments.Unet import UNet
from utils.helper_fncs import tensor2array
from scipy.optimize import linear_sum_assignment


"=============== MODEL DEFINITION =============== "

class Temp_Sigmoid(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer('temperature', torch.tensor(1.0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        return self.sigmoid(x / self.temperature)
    
class IDSD(nn.Module):
    def __init__(self, fs=256, T=1, **kwargs):
        super(IDSD, self).__init__()

        self.fs = int(fs) #sampling frequency in Hz
        self.T = int(T) #signal duration in seconds
        self.nr_samples = int(fs*T) #nr of samples in the signal N
        self.padL = int(kwargs.get('padL',0)) #nr of samples to pad on each side of the signal
        self.sigmoid_temperature = kwargs.get('sigmoid_temperature', {'start':1.0, 'end':0.5, 'nr_iters':50})


        self.predictor = nn.Sequential(
            UNet(in_channels=1, 
                 out_channels=1, 
                 kernel_size=int(kwargs.get('kernel_size', 15)), 
                 start_nr_chs=int(kwargs.get('start_nr_chs',64)),
                 maxch=int(kwargs.get('maxch',1024)),
                 padding_mode='zeros', 
                 bias=kwargs.get('bias', True)),   
            Temp_Sigmoid(),
        )
        
        self.init_sigmoid_temperature()
        self.criterion = nn.MSELoss(reduction='none')

    def init_sigmoid_temperature(self):
        self.decay_factor_temp = torch.as_tensor(self.sigmoid_temperature['start']-self.sigmoid_temperature['end'])/(self.sigmoid_temperature['nr_iters'])
        self.predictor[1].temperature = torch.tensor(self.sigmoid_temperature['start'])

    def update_sigmoid_temperature(self, epoch):
        self.predictor[1].temperature = torch.maximum(torch.as_tensor(self.sigmoid_temperature['end']),self.sigmoid_temperature['start']-self.decay_factor_temp*epoch)

    def set_sigmoid_temperature(self, value):
        self.predictor[1].temperature = torch.tensor(value, dtype=torch.float32)

    def find_component_with_highest_peak(self, yF_split):
        # yF_split shape: [batch_size, 2, K, N/2]
        # Find the component with the highest peak in the Fourier domain

        maxes, _ = torch.max(yF_split[:,0]**2 + yF_split[:,1]**2, axis=-1) # [batch_size, K]
        max_components = torch.argmax(maxes, axis=-1) #[bs]
        return yF_split[np.arange(yF_split.shape[0]),:,max_components,:] # [batch_size, 2, N/2]

    def compute_loss(self, yF_split, modes):
        """
        yF_split: (batch_size, 2, K, F). Axes 1 is the complex axis (real/imag), axis 2 are the modes.
        modes: (batch_size, 2, 2, F). Axes 1 is the complex axis (real/imag), axis 2 are the predicted mode and the remainder mode.
        """

        largest_comp = self.find_component_with_highest_peak(yF_split) #[bs, 2, N/2]

        # Penalize the mean-squared-error of both the real and imaginary part of the predicted mode 0 and the ground-truth mode with the highest peak in frequency domain.
        value = self.criterion(largest_comp[:,0] , modes[:,0,0]).mean(-1)  + self.criterion(largest_comp[:,1], modes[:,1,0]).mean(-1)
        loss = value.mean() # Average over the batch
        return loss


    def preprocessing(self, x):
        # x with expected shape: [batch_size, 1, N]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)

        x_pad = F.pad(x, (self.padL,self.padL), mode='reflect')
        xF = torch.fft.fft(x_pad, dim=-1)[...,:(x_pad.shape[-1]//2)] #Only use the positive frequencies

        if xF.ndim == 1:
            xF = xF.unsqueeze(0) #Add batch dimension
        if x.ndim == 2:
            xF = xF.unsqueeze(1) #Add channels
        xF_split = torch.stack([torch.real(xF), torch.imag(xF)], dim=1) # [batch_size, 2, channels, N/2]
        return xF_split

    def post_processing(self, x):
        """Post-process the output of the model to get the final predicted components.
        x is the right-side of the F-domain, with real and imaginary split. Shape: [bs, 2 x nr_components x N/2]
        This function returns the real time-domain signal by mirorring the positive frequencies to the negative frequencies and applying the inverse FFT.
        """
        remove_ch_axis  = False
        if x.ndim == 3:
            x = np.expand_dims(x,2) #Add the channel dimension
            remove_ch_axis = True

        bs,_,K,F = x.shape
        T = 2*F
        idxs = np.flip(np.arange(1,F+1),axis = 0)

        u_hat = np.zeros([bs,K,T],dtype = complex)
        u_hat[:,:,F:] = x[:,0] + 1j*x[:,1]
        u_hat[:,:,idxs] = x[:,0] - 1j*x[:,1] #Filling up the negative-frequency spectrum with the conjugate of the found positive spectrum.
        u_hat[:,:,0] = np.conj(u_hat[:,:,-1])    

        u = np.real(np.fft.ifft(np.fft.ifftshift(u_hat,axes=-1), axis=-1)) #[bs , K , N]

        # Remove the padded part of the signal
        if self.padL > 0:
            u = u[...,self.padL:-self.padL] 

        if remove_ch_axis:
            u = u[:,0]
        return u
    
    def post_processing_mask(self, pred_masks, x):
        """Post-process the predicted mask for the dominant mode.
        It will only keep the contiguous cluster in freq domain that contains the highest value in the data among them.
        
        pred_masks is in the right-side of the F-domain. Shape: [bs, 1, N/2]
        x is in the right-side of the F-domain, with real and imaginary split. Shape: [bs, 2, N/2]
        """
        high_idxs = (pred_masks[:,0] > 1e-3).to(torch.int32)  #boolean of size [bs, N/2]
        diffs = torch.diff(high_idxs, axis=-1)

        # Get indices of start and end of clusters, where a cluster is defined as a contiguous set of frequency bins where the mask is above 1e-3.
        cluster_start_indices = [torch.cat([torch.tensor([0], device=pred_masks.device), (torch.where(diffs[b] == 1)[0] + 1).to(torch.int32)]) for b in range(x.shape[0])] 
        cluster_end_indices = [torch.cat([(torch.where(diffs[b] == -1)[0]).to(torch.int32), torch.tensor([pred_masks.shape[-1]-1], device=pred_masks.device)]) for b in range(x.shape[0])] 
    
        sel_cluster = np.zeros(x.shape[0], dtype=np.int32)
        for b in range(x.shape[0]):
            max_values_mask = []
            max_values = []

            for start,end in zip(cluster_start_indices[b], cluster_end_indices[b]):
                max_values_mask.append(tensor2array(torch.max(pred_masks[b,0,start:(end+1)])))

            max_mask = np.max(max_values_mask)
            optional_clusters = np.where(max_mask - max_values_mask < 1e-3)[0] #Find all clusters that have their max mask value close to the overal max mask value.

            for cluster_idx in optional_clusters: 
                start, end = cluster_start_indices[b][cluster_idx], cluster_end_indices[b][cluster_idx]                    
                # Find the maximum value of the input spectrum in this cluster
                max_values.append(tensor2array(torch.max(torch.sqrt(x[b,0,start:(end+1)]**2 + x[b,1,start:(end+1)]**2))))
            
            # Select the cluster that contains the highest value in the input spectrum
            sel_cluster[b] = optional_clusters[int(np.argmax(np.array(max_values)))] 

        # Set the frequency bins to zero which are not part of the final selected cluster
        to_zero = torch.ones(pred_masks.shape[0],1, pred_masks.shape[-1], dtype=torch.bool)
        for b in range(x.shape[0]):
            to_zero[b,0,cluster_start_indices[b][sel_cluster[b]]:(cluster_end_indices[b][sel_cluster[b]]+1)] = False 
        pred_masks[to_zero] = 0.0
        
        return pred_masks
        
    def forward(self, x, epoch, inference=False):
        """
        x is of shape[batch_size, 2, 1, N].
        """
        if epoch is not None:
            self.update_sigmoid_temperature(epoch)
        else:
            assert inference, "If epoch is None, inference should be True"
            self.set_sigmoid_temperature(0.1)

        x = x.squeeze(2) #Remove the channel dimension as it is 1 anyway.
        modes = []

        x_mag = torch.sqrt(x[:,0]**2 + x[:,1]**2).unsqueeze(1) # [batch_size, 1, N/2] --> Use magnitude of the signal as input to the predictor.     
        pred_masks = self.predictor(x_mag) # [batch_size, 1, N/2]

        if inference: 
            pred_masks = self.post_processing_mask(pred_masks, x)

        # Compute the remainder mask (that takes the rest of the signal)
        maskr = torch.ones_like(pred_masks, device=pred_masks.device) - pred_masks # [batch_size, 1, N/2]
        masks = torch.concat([pred_masks, maskr], dim=1) # # [batch_size, 2, N/2]
        modes = masks.unsqueeze(1) * x.unsqueeze(2) #[batch_size, 1, 2,  N/2] x [batch_size, 2, 1, N/2]  --> [batch_size, 2, nr_modes,  N/2]. nr_modes is here always 2 as we predict one mode and the remainder mode.

        return modes, masks # Shape: (batch_size, 2, nr_modes, N/2), (batch_size, nr_modes, N/2)

    def predict(self, x, y, K):
        """
        Predict K+1 modes (K modes and a remaining noise mode) from the input signal x.
        x is of shape [batch_size, 1, N]
        y is of shape [batch_size, K+1, N], to be used for reordering the predictions.

        Returns:
        modes_pred: Numpy array of size: (batch_size, K+1, N)
        masks_pred: Numpy array of size: (batch_size, K+1, N/2) 
        """

        # Preprocess the data
        xF_split = self.preprocessing(x) 
        
        residual = xF_split
        all_modes, all_masks = [], [] 
        self.eval()
        for iteration in range(K): 
            # Run the model
            modes, masks = self.forward(residual, epoch=None, inference=True)  
            residual = residual - modes[:,:,0:1]
            
            all_modes.append(modes[:,:,0])
            all_masks.append(masks[:,0])

        all_modes.append(modes[:,:,1])
        all_masks.append(masks[:,1])
            
        all_modes = torch.stack(all_modes, dim=2) 
        all_masks = torch.stack(all_masks, dim=1) 

        # Post-process the output
        modes_pred = self.post_processing(tensor2array(all_modes))
        masks_pred = tensor2array(all_masks)

        # Reorder the predictions to best match the ground truth, and reorder the predicted masks accordingly        
        modes, masks, _ = self.reorder_signals(y, modes_pred, masks_pred)
        return modes, masks
    
    def reorder_signals(self, gt, pred, masks):
        """
        Reorder predicted signals to match ground-truth signals based L2 distance.

        gt: (batch, K, N) np array of ground-truth signals
        pred: (batch, K, N) np array of predicted signals
        masks: (batch, K, N/2) np array of predicted masks in Frequency domain (positive axis only)
        
        Returns:
            reordered_predictions: (batch, K, N) array of reordered predicted signals
            reordered_masks: (batch, K, N/2) array of reordered predicted masks
            order: (batch, K) array of indices indicating the reordering
        """
        batch, K, _ = gt.shape

        reordered_predictions = np.zeros_like(gt)
        order = np.zeros((batch, K), dtype=int)

        for i in range(batch):
            cost_matrix = np.linalg.norm(gt[i,:,None,:] - pred[i,None,:,:], axis=-1, ord=2)
            
            # Solve assignment problem to minimize total cost
            _, col_ind = linear_sum_assignment(cost_matrix)
            order[i] = col_ind

            # Now col_ind gives the best matching of predictions to gt
            reordered_predictions[i] = pred[i,col_ind]

        reordered_masks = masks[np.arange(masks.shape[0])[:, None],order]
        return reordered_predictions, reordered_masks, order
