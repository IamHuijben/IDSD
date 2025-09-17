import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ReflectionPadding1D(nn.Module):
    def __init__(self, padding=(1, 1)):
        super(ReflectionPadding1D, self).__init__()
        self.padding = padding

    def forward(self, x):
        left, right = self.padding
        return F.pad(x, (left, right), mode='reflect') #In the github they use 'symmetric' from Keras, which does not exist in Pytorch.


class KerasStyleAttention(nn.Module):
    """ 
    implement Keras-style attention mechanism which is basically single-head, without scaling before softmax and without trainable linear projection.
    """
    def __init__(self):
        super(KerasStyleAttention, self).__init__()

    def forward(self, query, value=None, key=None, mask=None):
        # Keras-style default: self-attention if only one input
        if value is None: value = query
        if key is None: key = query

        # query, key, value: (batch, time, features)
        # Attention scores: (batch, time, time)
        scores = torch.matmul(query, key.transpose(1, 2))  # no scaling

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (batch, time, time)
        output = torch.matmul(attn_weights, value)  # (batch, time, features)
        return output
    
class RPConv1DBlock_attention(nn.Module):
    def __init__(self, half_kernel_size=7):
        super(RPConv1DBlock_attention, self).__init__()
        self.half_kernel_size = half_kernel_size
        self.half_kernel_size_hf = half_kernel_size // 2
        self.half_kernel_size_hhf = half_kernel_size // 4

        self.RP = ReflectionPadding1D((half_kernel_size, half_kernel_size))
        self.RP_hf = ReflectionPadding1D((self.half_kernel_size_hf, self.half_kernel_size_hf))
        self.RP_hhf = ReflectionPadding1D((self.half_kernel_size_hhf, self.half_kernel_size_hhf))

        self.C1D_act = nn.Conv1d(1, 1, kernel_size=2*half_kernel_size+1, bias=True)
        self.C1D_act_hf = nn.Conv1d(1, 1, kernel_size=2*self.half_kernel_size_hf+1, bias=True)
        self.C1D_act_hhf = nn.Conv1d(1, 1, kernel_size=2*self.half_kernel_size_hhf+1, bias=True)

        # Use glorot uniform / Xavier initialization (=same) since that is the standard in keras
        init.xavier_uniform_(self.C1D_act.weight)
        init.xavier_uniform_(self.C1D_act_hf.weight)
        init.xavier_uniform_(self.C1D_act_hhf.weight)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # self.point_conv_relu = nn.Conv1d(1, 1, kernel_size=1)
        # self.point_conv_sigmoid = nn.Conv1d(1, 1, kernel_size=1)

        # Parameters (kernels)
        self.kernel2 = nn.Parameter(torch.empty(1, 5, 2*half_kernel_size+1))
        self.kernel3 = nn.Parameter(torch.empty(1, 1, 2*half_kernel_size+1))

        # nn.init.uniform_(self.kernel1, a=-0.05, b=0.05)
        nn.init.uniform_(self.kernel2, a=-0.05, b=0.05)
        nn.init.uniform_(self.kernel3, a=-0.05, b=0.05)

        self.attention = KerasStyleAttention()

    def forward(self, x):
        # Input shape: (batch, channels=1, time)
        B, C, T = x.shape
        assert C == 1, "Input should have 1 channel"

        x_pad = self.RP(x)

        xx = self.attention(x, x)
        x_f = self.relu(self.C1D_act(x_pad))
        x_f = self.dropout(x_f)
        x_hf = self.relu(self.C1D_act_hf(x_pad))
        x_hf = self.dropout(x_hf)
        x_hhf = self.relu(self.C1D_act_hhf(x_pad))
        x_hhf = self.dropout(x_hhf)

        # Concatenate
        x_concat = torch.cat([
            x_pad,
            self.RP(x_f),
            self.RP_hf(x_hf),
            self.RP_hhf(x_hhf),
            self.RP(xx)
        ], dim=1)  # concat along channel dim
        
        
        # Conv1d: kernel2 shape = (out_channels=1, in_channels=5, kernel_size)
        x_conv = F.conv1d(x_concat, self.kernel2, stride=1, padding=0)
        x_conv = torch.tanh(x_conv)
        x_conv_pad = self.RP(x_conv)

        # Normalized filter3
        filt3 = self.kernel3 ** 2
        filt3 = filt3 / filt3.sum()
        x_out = F.conv1d(x_conv_pad, filt3, stride=1, padding=0)

        # Residual subtraction
        x_final = x - x_out
        return x_final



class TVD_IC(nn.Module):
    def __init__(self, lam=1.0, Num=2):
        super(TVD_IC, self).__init__()
        self.lam = lam
        self.Num = Num

    def forward(self, inputs):
        # inputs shape: (B, 1, T) → squeeze last dim
        assert inputs.dim() == 3, "Input should be 3D tensor (B, T, 1)"
        assert inputs.shape[1] == 1, "2nd dimension should be 1 for single channel input"
        xx = inputs.squeeze(1)  # shape: (B, T)
        
        z = torch.zeros_like(xx[:, :-1])  # shape: (B, T-1)
        
        for _ in range(self.Num):
            left = -z[:, 0:1]  # shape: (B, 1)
            mid = z[:, :-1] - z[:, 1:]      # shape: (B, T-2)
            right = z[:, -1:]               # shape: (B, 1)
            y = torch.cat([left, mid, right], dim=1)  # shape: (B, T)
            x = xx - y                      # residual
            z = z + (1.0 / 3.0) * (x[:, 1:] - x[:, :-1])
            z = torch.clamp(z, min=-self.lam / 2, max=self.lam / 2)

        # restore shape: (B,1, T)
        return x.unsqueeze(1)

    def extra_repr(self):
        return f"lam={self.lam}, Num={self.Num}"


class ExtMidSig(nn.Module):
    def __init__(self, x_length, y_length):
        super(ExtMidSig, self).__init__()
        self.x_length = x_length
        self.y_length = y_length
        self.exd_length = int((self.x_length - self.y_length) / 2)

    def forward(self, x):
        # x is expected to be (batch_size, channels, time_steps)
        return x[..., self.exd_length : -self.exd_length]
