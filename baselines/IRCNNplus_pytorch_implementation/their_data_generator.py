import numpy as np
import pywt
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import torch

class IRCNN_datagen(Dataset):
    def __init__(self,data_fold='train'):
        """
        Args:
        data_fold: 'train' or 'test'. Whether to load the train or test part of the dataset.
        """
        self.x_train, self.y_train, self.x_test, self.y_test = self.gen_data()
        self.data_fold = data_fold
        if self.data_fold=='train':
            self.x = self.x_train #[532, 4800, 1]
            self.y = self.y_train #[532, 4800, 1]
        else:
            self.x = self.x_test #[228, 4800, 1]
            self.y = self.y_test #[228, 4800, 1]

        # Reshape data to be compatible with model input: [bs, 1, KT] for x 
        # make y of shape [bs, K, T] as that is what my loss function expects.
        self.x = self.x.transpose(0,2,1)
        self.y = self.y.reshape(self.y.shape[0], 2, -1)
        self.dataset = 'IRCNN_datagen'

    def __len__(self):
        return len(self.x)
    
    # Generate data
    def gen_data(self,):
        # This function is taken from: https://github.com/zhoudafa08/RRCNN_plus/blob/main/rrcnn_att.py
        # add Gaussian white noise
        def wgn(x, snr):
            snr = 10**(snr/10.0)
            xpower = np.sum(x**2)/len(x)
            npower = xpower / snr
            noise = np.random.randn(len(x)) * np.sqrt(npower)
            return noise
        length = 2400
        snr = 25
        exd_length = int(0.5*length)
        mode = 'symmetric'
        x_train = np.empty([length+2*exd_length, 1])
        y_train = np.empty([2*length, 1])
        t = np.linspace(0, 6, length)
        
        for j in range(5, 15):
            x1 = np.cos(j *np.pi * t)
            x2 = np.cos((j+1.5)*np.pi*t)
            tmp_x = pywt.pad(x1 + x2, exd_length, mode)
            tmp_noise = wgn(tmp_x.reshape(-1), snr)
            x_train = np.c_[x_train, tmp_x+tmp_noise]
            y_train = np.c_[y_train, np.r_[x2, x1]]
            tmp_x = pywt.pad(x2, exd_length, mode)
            tmp_noise = wgn(tmp_x.reshape(-1), snr)
            x_train = np.c_[x_train, tmp_x+tmp_noise]
            y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
            x2 = np.cos((j+1.5)*np.pi*t + t * t + np.cos(t))
            tmp_x = pywt.pad(x1 + x2, exd_length, mode)
            tmp_noise = wgn(tmp_x.reshape(-1), snr)
            x_train = np.c_[x_train, tmp_x+tmp_noise]
            y_train = np.c_[y_train, np.r_[x2, x1]]
            tmp_x = pywt.pad(x2, exd_length, mode)
            tmp_noise = wgn(tmp_x.reshape(-1), snr)
            x_train = np.c_[x_train, tmp_x+tmp_noise]
            y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
            for i in range(2, 20):
                x2 = np.cos(j*i*np.pi*t)
                tmp_x = pywt.pad(x1 + x2, exd_length, mode)
                tmp_noise = wgn(tmp_x.reshape(-1), snr)
                x_train = np.c_[x_train, tmp_x+tmp_noise]
                y_train = np.c_[y_train, np.r_[x2, x1]]
                tmp_x = pywt.pad(x2, exd_length, mode)
                tmp_noise = wgn(tmp_x.reshape(-1), snr)
                x_train = np.c_[x_train, tmp_x+tmp_noise]
                y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
                x2 = np.cos(j*i*np.pi*t + t * t + np.cos(t))
                tmp_x = pywt.pad(x1 + x2, exd_length, mode)
                tmp_noise = wgn(tmp_x.reshape(-1), snr)
                x_train = np.c_[x_train, tmp_x+tmp_noise]
                y_train = np.c_[y_train, np.r_[x2, x1]]
                tmp_x = pywt.pad(x2, exd_length, mode)
                tmp_noise = wgn(tmp_x.reshape(-1), snr)
                x_train = np.c_[x_train, tmp_x+tmp_noise]
                y_train = np.c_[y_train, np.r_[x2, 0.0*t]]
        

        # print(x_train.shape, y_train.shape)  
        x_train = np.delete(x_train, 0, axis=1)
        y_train = np.delete(y_train, 0, axis=1)
        indices = np.arange(x_train.shape[1])
        np.random.seed(0)
        np.random.shuffle(indices)
        x_sample = x_train[:, indices]
        y_sample = y_train[:, indices]
        train_num = int(0.7*x_sample.shape[1])
        x_train = x_sample[:, :train_num]
        x_test = x_sample[:, train_num:]
        y_train = y_sample[:, :train_num]
        y_test = y_sample[:, train_num:]
        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        
        x_train = x_train.transpose().reshape(-1, length+2*exd_length, 1)
        y_train = y_train.transpose().reshape(-1, 2*length, 1)
        x_test = x_test.transpose().reshape(-1, length+2*exd_length, 1)
        y_test = y_test.transpose().reshape(-1, 2*length, 1)
        # print(x_train.shape, y_train.shape)
        return x_train, y_train, x_test, y_test

    def __getitem__(self, idx):
        # shape of x: [1, KT] (already padded)
        # shape of y: [K, T]
        return torch.tensor(self.x[idx],dtype=torch.float32), torch.tensor(self.y[idx],dtype=torch.float32)
    
if __name__ == "__main__":
    dataloader = IRCNN_datagen(data_fold='test')
    # x_train, y_train, x_test, y_test = data()

    # idx = 1
    # fig,axs = plt.subplots(2,1, figsize=(10,6))
    # axs[0].plot(x_train[idx,:2400,0]);axs[0].set_ylabel('y0')
    # axs[1].plot(x_train[idx,2400:,0]);axs[1].set_ylabel('y1')
    # plt.show()
    print('done')
