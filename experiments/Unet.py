import torch
import torch.nn as nn
from torch.nn.functional import relu

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=15, start_nr_chs=64, maxch=1024, padding_mode='zeros', bias=True):
        super().__init__()
        # Encoder
        # In the encoder, convolutional layers with the Conv1d function are used to extract features from the input signal. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.

        # The number of channels in the convolutional layers doubles after each block, starting from start_nr_chs and capped at maxch.
        # The kernel size for the convolutional layers is specified by the kernel_size parameter, and padding is applied to maintain the same length of the output as the input.
        # The padding_mode parameter determines how the padding is applied (e.g., 'zeros', 'reflect', etc.), and the bias parameter indicates whether to include a bias term in the convolutional layers.
        # -------
        
        self.e11 = nn.Conv1d(in_channels, min(maxch,start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)
        self.e12 = nn.Conv1d(min(maxch,start_nr_chs), min(maxch,start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias) 
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.e21 = nn.Conv1d(min(maxch,start_nr_chs), min(maxch,2*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias) 
        self.e22 = nn.Conv1d(min(maxch,2*start_nr_chs), min(maxch,2*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias) 
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.e31 = nn.Conv1d(min(maxch,2*start_nr_chs), min(maxch,4*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias) 
        self.e32 = nn.Conv1d(min(maxch,4*start_nr_chs), min(maxch,4*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias) 
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) 

        self.e41 = nn.Conv1d(min(maxch,4*start_nr_chs), min(maxch,8*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)
        self.e42 = nn.Conv1d(min(maxch,8*start_nr_chs), min(maxch,8*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias) 
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) 

        self.e51 = nn.Conv1d(min(maxch,8*start_nr_chs), min(maxch,16*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias) 
        self.e52 = nn.Conv1d(min(maxch,16*start_nr_chs), min(maxch,16*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias) 


        # Decoder
        self.upconv1 = nn.ConvTranspose1d(min(maxch,16*start_nr_chs), min(maxch,8*start_nr_chs), kernel_size=2, stride=2)
        self.d11 = nn.Conv1d(min(2*maxch,16*start_nr_chs), min(maxch,8*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)
        self.d12 = nn.Conv1d(min(maxch,8*start_nr_chs), min(maxch,8*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)

        self.upconv2 = nn.ConvTranspose1d(min(maxch,8*start_nr_chs), min(maxch,4*start_nr_chs), kernel_size=2, stride=2)
        self.d21 = nn.Conv1d(min(2*maxch,8*start_nr_chs), min(maxch,4*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)
        self.d22 = nn.Conv1d(min(maxch,4*start_nr_chs), min(maxch,4*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)

        self.upconv3 = nn.ConvTranspose1d(min(maxch,4*start_nr_chs), min(maxch,2*start_nr_chs), kernel_size=2, stride=2)
        self.d31 = nn.Conv1d(min(2*maxch,4*start_nr_chs), min(maxch,2*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)
        self.d32 = nn.Conv1d(min(maxch,2*start_nr_chs), min(maxch,2*start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)

        self.upconv4 = nn.ConvTranspose1d(min(maxch,2*start_nr_chs), min(maxch,start_nr_chs), kernel_size=2, stride=2)
        self.d41 = nn.Conv1d(min(2*maxch,2*start_nr_chs), min(maxch,start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)
        self.d42 = nn.Conv1d(min(maxch,start_nr_chs), min(maxch,start_nr_chs), kernel_size=kernel_size, padding='same', padding_mode=padding_mode, bias=bias)

        # Output layer
        self.outconv = nn.Conv1d(start_nr_chs, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out