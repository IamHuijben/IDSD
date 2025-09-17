from baselines.IRCNNplus_pytorch_implementation.modules_torch import *
import torch
import torch.nn as nn
from utils.helper_fncs import tensor2array
from scipy.optimize import linear_sum_assignment
import numpy as np

"=============== TORCH IMPLEMENTATION OF IRRCNN+ MODEL =============== "


class IRCNN_plus(nn.Module):
    def __init__(self, length, fs, T, half_kernel_size_part1, half_kernel_size_part2,S=12):
        """
        :param length: nr of samples in signal
        :param fs: sampling frequency in Hz
        :param T: signal duration in seconds
        :param half_kernel_size_part1: int, half kernel size for part 1 of the model
        :param half_kernel_size_part2: int, half kernel size for part 2 and 3 of the model
        :param S: int, number of layers in each part of the model
        """
        super(IRCNN_plus, self).__init__()
        self.fs = fs
        self.T = T
                                
        # Cell 1 — part 1 
        self.part1_layers = nn.Sequential(
            *[RPConv1DBlock_attention(half_kernel_size_part1) for _ in range(S)]
        )

        # Cell 1 — part 2 
        self.part2_layers = nn.Sequential(
            *[RPConv1DBlock_attention(half_kernel_size_part2) for _ in range(S)]
        )
        self.tvd1 = TVD_IC(0.04, 5)
        self.ext1 = ExtMidSig(2*length, length)

        # Cell 2 — part 3
        self.part3_layers = nn.Sequential(
            *[RPConv1DBlock_attention(half_kernel_size_part2) for _ in range(S)]
        )
        self.tvd2 = TVD_IC(0.08, 6)
        self.ext2 = ExtMidSig(2*length, length)
        self.S = S
        self.padL = length // 2
        self.length = length

    def forward(self, x):
        # For IRCNN_plus, we need to return the data in a specific format. 
        # x is: [bs,1,T] or [bs, 1,KT]
        # final_out is of shape: [bs, 1, KT]
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
    
        if x.shape[-1] == (2*self.length):
            # Input is already padded in their dataloader
            pass
        else: # My dataloader did not pad the input data yet. 
            assert x.shape[-1] == self.length
            x = torch.nn.functional.pad(x, (self.padL,self.padL), mode='reflect') # [bs,1,KT]
        # x = x.moveaxis(1,2) # [bs,KT,1]

        # Cell 1: part 1
        out = self.part1_layers(x)
        inputs2 = x - out  # residual

        # Cell 1: part 2
        out = self.part2_layers(inputs2)
        out = self.tvd1(out)
        out1 = self.ext1(out)

        # Cell 2
        inputs3 = inputs2 - out
        out = self.part3_layers(inputs3)
        out = self.tvd2(out)
        out2 = self.ext2(out)

        # Concatenate on time dimension (make this the last dimension, to have channel-first format for pytorch)
        final_out = torch.cat([out1, out2], dim=-1)
        return final_out

    def sort_on_amplitude(self, y_true):
        # y_true shape: [batch_size, K, T]
        # Find the component with the highest amplitude in time domain

        maxes, _ = torch.max(y_true, axis=-1) # [batch_size, K]
        order = torch.argsort(maxes, descending=True, dim=-1) #[bs, K]

        # Expand order to match [batch_size, K, T]
        order_exp = order.unsqueeze(-1).expand(-1, -1, y_true.shape[-1])

        # Gather along dim=1
        y_sorted = torch.gather(y_true, dim=1, index=order_exp)

        return y_sorted # [batch_size, K, T]
    
    def compute_loss(self, y_true, y_pred):
        """
        y_true  is expected to be of shape [batch_size, K, T]
        y_pred is expected to be of shape [batch_size, 1, KT]
        """
        if (y_true.shape[1]*y_true.shape[-1]) > y_pred.shape[-1]:
            # Remove the last y_true component which is the noise component. This is triggered when using the SignalDecompositionGenerator with K modes, which adds a noise component as (K+1)-th mode.
            y_true = y_true[:,:-1,:]
            assert (y_true.shape[1]*y_true.shape[-1]) == y_pred.shape[-1]

            # Given that the SignalDecompositionGenerator generates the modes in random order, we y_true based on amplitude to give consistent targets to the model during training.
            y_true = self.sort_on_amplitude(y_true) # I have added this functionality to the model, as 

        y_true = y_true.reshape(y_true.shape[0], -1).unsqueeze(1) # [bs,1, KT]
        
        assert y_true.shape == y_pred.shape, f"y_true shape {y_true.shape} does not match y_pred shape {y_pred.shape}"
        assert y_true.ndim == 3
        assert y_true.shape[1] == 1, f"y_true shape {y_true.shape} does not match expected shape [batch_size, T, 1]"

        error = y_true - y_pred
        mse = torch.mean(error ** 2)

        length = y_pred.shape[-1]
        y_pred1 = y_pred[..., :(length // 2)]
        y_pred2 = y_pred[..., (length // 2):]

        # plt.figure();plt.plot(y_pred1[0,0].cpu().detach().numpy());plt.plot(y_pred2[0,0].cpu().detach().numpy())
        # First-order total variation
        tv_pred11 = torch.mean(torch.abs(y_pred1[..., :-1] - y_pred1[..., 1:]))
        tv_pred21 = torch.mean(torch.abs(y_pred2[..., :-1] - y_pred2[..., 1:]))


        # Third-order total variation
        tv_pred13 = torch.mean(torch.abs(
            y_pred1[..., :-3] 
            - 3 * y_pred1[..., 1:-2] 
            + 3 * y_pred1[..., 2:-1] 
            - y_pred1[..., 3:]
        ))
        tv_pred23 = torch.mean(torch.abs(
            y_pred2[..., :-3] 
            - 3 * y_pred2[..., 1:-2] 
            + 3 * y_pred2[..., 2:-1] 
            - y_pred2[..., 3:]
        ))

        # Optional: Orthogonal component (commented out, but kept for reference)
        # dot_product = (y_pred1 * y_pred2).pow(2)
        # norm_product = torch.norm(y_pred1) * torch.norm(y_pred2)
        # orthogonal = dot_product.sum() / (norm_product + 1e-8)  # +eps to avoid division by zero

        l1_penalty = 0
        l2_penalty = 0
        for n,p in self.named_parameters():
            if 'C1D_act' in n and p.requires_grad:
                l2_penalty += torch.norm(p, 2)
            elif 'kernel2' in n or 'kernel3' in n:
                l1_penalty += torch.norm(p, 1)
        loss = mse + 0.01 * (tv_pred11 + 2 * tv_pred13 + tv_pred21 + 2 * tv_pred23) + 1e-4 * l1_penalty + 1e-4 * l2_penalty
        return loss
    
    def predict(self, x, y, K):
        """ 
        Predict 2 modes from the input signal x.
        x is of shape [batch_size, 1, T]
        y is of shape [batch_size, K+1, T], to be used for reordering the predictions.

        Returns:
        modes_pred: Numpy array of size: (batch_size, K+1, N)
        """
        
        modes_pred = self.forward(x)   #[bs, 1, KN]
        modes_pred = modes_pred.reshape(modes_pred.shape[0], K, -1)  #[bs, K, N]
        modes_pred = tensor2array(modes_pred)

        add_last_comp = False
        if (y.shape[1] - modes_pred.shape[1]) == 1:
            # Remove the last y component which is the noise component. This is part of y in case the SignalDecompositionGenerator is used.
            y = y[:,:-1,:]
            assert y.shape == modes_pred.shape
            add_last_comp = True

        # Reorder the predictions to best match the ground truth, and reorder the predicted masks accordingly        
        modes = self.reorder_signals(y, modes_pred)
        
        if add_last_comp:
            # Add a predicted zero mode to match the noise component in the ground-truth.
            modes = np.concatenate([modes, np.zeros((modes.shape[0],1,modes.shape[2]))], axis=1)

        return modes, None
    

    def reorder_signals(self, gt, pred):
        """
        Reorder predicted signals to match ground-truth signals based L2 distance.

        gt: (batch, N, T) np array of ground-truth signals
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
