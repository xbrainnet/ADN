import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
from torch.utils.data.dataset import Dataset

def denoise_channel(ts, bandpass, signal_freq, bound):
    """
    reference: https://github.com/ycq091044/ContraWR
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)

    return np.array(ts_out)

def noise_channel(ts, mode, degree, bound):
    """
    reference: https://github.com/ycq091044/ContraWR
    Add noise to ts
    mode: high, low, both
    degree: degree of noise, compared with range of ts    
    
    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)
        
    """
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
    low_length = np.random.randint(len_ts//20, len_ts//2)

    # high frequency noise
    if mode == 'high':
        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
        out_ts = ts + noise
        
    # low frequency noise
    elif mode == 'low':
        noise = degree * num_range * (2*np.random.rand(low_length)-1)
        x_old = np.linspace(0, 1, num=low_length, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise
        
    # both high frequency noise and low frequency noise
    elif mode == 'both':
        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
        noise2 = degree * num_range * (2*np.random.rand(low_length)-1)
        x_old = np.linspace(0, 1, num=low_length, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts
        
    return out_ts

class MIDataset(Dataset):
    def __init__(self, path_data, path_label, 
                       contra, bound, 
                       bandpass1, bandpass2, 
                       signal_freq):
        super(MIDataset, self).__init__()
        self.contra = contra
        self.n_channels = 64
        self.bound = bound
        self.bandpass1 = bandpass1
        self.bandpass2 = bandpass2
        self.signal_freq = signal_freq
        self.path_data = path_data
        self.path_label = path_label
        self.X = np.load(self.path_data)

        self.y = np.load(self.path_label)
        self.y = torch.from_numpy(self.y).flatten()

    def jittering(self, x, ratio, deg):
        """
        Add noise to multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=deg, bound=self.bound)
        return x
    
    def bandpas_filtering(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_length, n_channel)
        Output: 
            x: (n_length, n_channel)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            else:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
        return x
    
    def augment(self, x):
        t = np.random.rand()

        if t > 0.5:
            x = self.jittering(x, ratio=0.5, deg=0.05)
        else:
            x = self.bandpas_filtering(x, ratio=0.5)
        return x

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        
        if self.contra == 0:
            return torch.FloatTensor(X), y
        else:
            return torch.FloatTensor(X.copy()), torch.FloatTensor(self.augment(X.copy())), y

    def __len__(self):
        return len(self.y)
    