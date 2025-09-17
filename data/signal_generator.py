import torch
from torch.utils.data import Dataset
import numpy as np


class SignalDecompositionGenerator(Dataset):
    def __init__(self, **kwargs):
        """
        Args:
            min_freq (float): Minimum frequency for the generated signals.
            max_freq (float): Maximum frequency for the generated signals.
            fs (int): Sampling frequency.
            a_min (float): Minimum amplitude for the generated signals.
            a_max (float): Maximum amplitude for the generated signals.
            padL (int): Padding length for the signals as used in the model, to use here to determine the order of GT based on highest peak in padded spectrum.
            T (float): Duration of the signal in seconds.
            nr_components (int or string): Number of components in the signal.
            noise_sigma (float): Standard deviation of Gaussian noise to add to the signals.
        """
        self.f_min = kwargs.get('min_freq')
        self.f_max = kwargs.get('max_freq')
        self.fs = kwargs.get('fs')
        assert self.f_min > 0, "min_freq must be greater than 0"
        assert self.f_max <= self.fs/2, "max_freq must be less than fs/2"
        self.a_min, self.a_max = kwargs.get('a_min', 0.1), kwargs.get('a_max', 1)
        self.T = kwargs.get('T')
        self.samples = self.fs*self.T
        self.t = np.arange(0, self.T, 1/self.fs)
        self.samples = int(self.fs*self.T)
        self.padL = kwargs.get('padL', 64)
        if isinstance(kwargs.get('nr_components'), str) and 'max' in kwargs.get('nr_components'):   
            self.max_nr_components = int(kwargs.get('nr_components')[3:])
            self.K = -1
            assert 'If nr_components is a string, it must be of the form "maxN" where N is an integer > 0.'
        else:
            self.K = kwargs.get('nr_components')
            self.max_nr_components = self.K
        
        if isinstance(kwargs.get('noise_sigma'), str) and 'max' in kwargs.get('noise_sigma'):
            self.max_noise_sigma = float(kwargs.get('noise_sigma')[3:])
            self.noise_sigma = - 1 
        else:
            self.noise_sigma = kwargs.get('noise_sigma', 0)
            self.max_noise_sigma = self.noise_sigma

        self.signal_types = kwargs.get('signal_types',['self.generate_sum_of_sinusoids','self.generate_AM_sinusoids','self.generate_intermittent_sinusoids'])
        
        self.dataset = 'SignalDecompositionGenerator'


    def __len__(self):
        return 1024
    

    def generate_sum_of_sinusoids(self, batch_size, K, frequency_sampling='random',**kwargs):
        """
        Returns:
        - x: Sum of sinusoids of size [batch_size, 1, N]
        - y: Components of size [batch_size, K, N]
        - frequencies: Frequencies of the components of size [batch_size, K]
        """
        if K == 1:
            frequency_sampling = 'random'

        if frequency_sampling == 'random':
            frequencies = np.random.uniform(self.f_min, self.f_max, (batch_size,K,1))
        elif frequency_sampling == 'nearby':
            f1_and_rest = np.random.uniform(self.f_min, self.f_max, (batch_size, K-1,1))
            f2 = np.random.uniform(np.max([f1_and_rest[:,0,0]-20,np.ones((batch_size))*self.f_min], axis=0), np.min([f1_and_rest[:,0,0]+20, np.ones((batch_size))*self.f_max],axis=0), (batch_size)) #sample second freq within 20Hz of first component
            frequencies = np.concat([f1_and_rest, f2[:, np.newaxis,np.newaxis]], axis=1)
        else:
            raise NotImplementedError
        
        amplitudes = np.random.uniform(self.a_min, self.a_max, (batch_size,K,1))  # Random amplitudes between 0.1 and 10
        amplitudes = np.sort(amplitudes,axis=1)[:,::-1] # Sort comopnents from highest to lowest amplitude
        phases = np.random.uniform(0, 2 * np.pi, (batch_size,K,1))  # Random phase shifts
        y = amplitudes * np.cos(2*np.pi * frequencies * self.t[np.newaxis,np.newaxis,:] + phases)
        x = y.sum(1,keepdims=True)
        return x,y,frequencies


    def generate_AM_sinusoids(self, batch_size, K,**kwargs):

        x,y,_ = self.generate_sum_of_sinusoids(batch_size, K, frequency_sampling='random')
        apply_on_0_or_1_comp = np.random.choice([0,min(K-1,1)], batch_size) 

        # Linear amplitude modulation between am_min and am_max, half of the time it goes up, half of the time it goes down
        am_start =  np.random.uniform(0.1, 1., (batch_size,1)) 
        am_end = np.random.uniform(0.1, 1., (batch_size,1)) 
        envelope = ((am_end - am_start) / self.T) * self.t[np.newaxis,:] + am_start

        y[np.arange(batch_size),apply_on_0_or_1_comp] = envelope*y[np.arange(batch_size),apply_on_0_or_1_comp]
        x = y.sum(1, keepdims=True)

        return x,y

    def generate_intermittent_sinusoids(self, batch_size, K, intermittent_type='on_off',**kwargs):
        """
        intermittent_type: 'on_off' or 'gaussian_burst'
        """
        x,y,_ = self.generate_sum_of_sinusoids(batch_size, K, frequency_sampling='random')

        envelope = np.zeros((batch_size, self.samples))
        num_bursts = np.random.randint(1, 5, batch_size)  
        apply_on_0_or_1_comp = np.random.choice([0,min(K-1,1)], batch_size) 

        for i in range(batch_size):
            burst_indices = np.random.randint(0., self.samples, size=num_bursts[i])
            width_of_bursts = np.random.uniform(0.01*self.samples, 0.1*self.samples, size=num_bursts[i]) # the sigma is between 1% and 10% of the total duration of the signal
            for burst_point, wb in zip(burst_indices, width_of_bursts):
                if intermittent_type == 'on_off':
                    start_of_burst = int(np.clip(burst_point - (wb//2), 0, self.samples-1))
                    end_of_burst = int(np.clip(burst_point + (wb//2), 0, self.samples-1))
                    envelope[i,start_of_burst:end_of_burst] += 1
                elif intermittent_type == 'gaussian_burst':
                    # Create a Gaussian burst centered at burst_point with width wb
                    gaussian_burst = np.exp(-((self.t - self.T*(burst_point/self.samples))**2) / (2 * self.T*(wb/self.samples)**2))
                    envelope[i] += gaussian_burst
                else:
                    raise NotImplementedError(f"Intermittent type '{intermittent_type}' is not implemented.")
        envelope = np.clip(envelope,0,1) #prevent double bursts at the same time location
        y[np.arange(batch_size),apply_on_0_or_1_comp] = envelope*y[np.arange(batch_size),apply_on_0_or_1_comp]

        x = y.sum(1, keepdims=True)
        return x,y
    
    
    def generate_various_sinusoid_types(self, batch_size, K, intermittent_type, frequency_sampling):
        data_gen_fnc = eval(np.random.choice(['self.generate_sum_of_sinusoids','self.generate_AM_sinusoids','self.generate_intermittent_sinusoids'], 1)[0]) 

        gen_data = data_gen_fnc(batch_size=batch_size, K=K,
                    frequency_sampling=frequency_sampling,
                    intermittent_type=intermittent_type) 
        if len(gen_data) == 3:
            _, y, _ = gen_data
        else:
            _, y = gen_data

        x = y.sum(1, keepdims=True)
        return x,y
    
    def generate_FM_sinusoids_broadband(self, batch_size, K,intermittent_type, frequency_sampling, **kwargs):
        """
        Generate always at least 1 FM signal, and the rest are sinusoids which are either pure, AM, or intermittent.
        """
        if K > 1:
            _,y = self.generate_various_sinusoid_types(batch_size, K-1, intermittent_type=intermittent_type, frequency_sampling=frequency_sampling)

        # Generate 1 FM signal
        amplitudes = np.random.uniform(self.a_min, self.a_max, (batch_size,1))
        phases = np.random.uniform(0, 2 * np.pi, (batch_size,1))  
        frequency = np.random.uniform(self.f_min, self.f_max, (batch_size,1)) 
        delta_freq =  np.random.uniform(-self.f_max*0.1,self.f_max*0.1,(batch_size,1))/self.T

        modulated_freq = np.clip((frequency + delta_freq*self.t[np.newaxis,:]), self.f_min, self.f_max)  # Ensure frequency stays within f_min and f_max
        y_FM = amplitudes * np.cos(2*np.pi * modulated_freq  * self.t[np.newaxis,:] + phases)

        if K > 1:
            y = np.concatenate([y, y_FM[:,np.newaxis,:]], axis=1)  # Add the FM signall 
        else:
            y = y_FM[:,np.newaxis,:]  
        x = y.sum(1,keepdims=True)
        
        return x,y           

    
    def add_noise(self, x, y):
        if self.noise_sigma > 0:
            noise = torch.normal(mean=0, std=self.noise_sigma*torch.ones_like(x))
            return x +  noise, torch.concat([y, noise], axis=1) 
        elif self.noise_sigma == -1:  # If noise_sigma is set to -1, sample noise_sigma from a uniform distribution between 0 and max_noise_sigma
            assert x.shape[0] == 1
            noise_sigma = np.random.uniform(0, self.max_noise_sigma)
            noise = torch.normal(mean=0, std=noise_sigma*torch.ones_like(x))
            return x + noise,  torch.concat([y, noise], axis=1) 
        else:
            noise = torch.zeros_like(x)
            return x, torch.concat([y, noise], axis=1)  # Add the zero-noise component to y, to make sure that noise is always the last one in y.

    def __getitem__(self, idx):
        if self.K < 0: # Randomly sample K between 1 and max_nr_components
            K = np.random.randint(1, self.max_nr_components+1)
        else:
            K = self.K

        data_gen_fnc = eval(np.random.choice(self.signal_types, 1)[0])
        if K > 1:
            frequency_sampling_opts = ['random', 'nearby']
        else:
            frequency_sampling_opts = ['random']

        gen_data = data_gen_fnc(batch_size=1, K=K,
                            frequency_sampling=np.random.choice(frequency_sampling_opts, 1)[0],
                            intermittent_type=np.random.choice(['on_off', 'gaussian_burst'], 1)[0]) 
        if len(gen_data) == 3:
            x, y, _ = gen_data
        else:
            x, y = gen_data
            
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        x,y = self.add_noise(x, y)

        # In case y is not having the max nr of components, pad it with zeros
        if y.shape[1] < (self.max_nr_components + 1): #Add 1 to account for the noise component that is added to y always
            padding = torch.zeros_like(y[:,0:1,:]).repeat(1,int(self.max_nr_components + 1 - y.shape[1]),1) # Create a padding with the same shape as the first component
            y = torch.cat([y, padding], dim=1)

        # Remove the batch size dimension here, as the dataloader automatically stacks that dimension
        return x[0], y[0] #[1, N], [K, N] 


