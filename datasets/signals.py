from scipy.signal import butter,lfilter
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


def butter_bandpass(lowcut=0.1,highcut=5,fs=125,order=5):
    nyq=0.5*fs
    low=lowcut/nyq
    high=highcut/nyq
    return butter(order,[low,high],btype='band')

def butter_filter(data,lowcut=0.1,highcut=5,fs=128,order=5):
    b,a=butter_bandpass(lowcut,highcut,fs,order=order)
    y=lfilter(b,a,data)
    return y



def stft(sig,fs,nperseg,noverlap,spec_only=False):
    f, t, Zxx = signal.stft(sig, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, boundary=None)
    Zxx=2*np.abs(Zxx)/np.sum(np.hanning(nperseg))
    # Zxx = np.log(Zxx+1e-8)
    Zxx=Zxx[np.where(np.logical_and(f>=0.0 , f <=5))[0],:]
    f=f[np.where(np.logical_and(f>=0.0 , f <=5))]
    if spec_only:
        return Zxx

    return f,t,Zxx

def rand_sfft(sig,fs,output_shape=(30,15)):
    slice_sec=np.random.uniform(2,3,1)[0]
    slide_sec = np.random.uniform(0.1,0.3,1)[0]
    nperseg = int(slice_sec * fs)
    step = int(slide_sec * fs)
    noverlap = nperseg - step
    f, t, Zxx = signal.stft(sig, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, boundary=None)
    Zxx = 2 * np.abs(Zxx) / np.sum(np.hanning(nperseg))
    h,w=Zxx.shape
    if h < output_shape[0]:
        Zxx=np.pad(Zxx,((output_shape[0]-h,0),(0,0)),mode='constant',constant_values=0)
    else:
        Zxx=Zxx[h-output_shape[0]:,:]

    if w < output_shape[1]:
        Zxx=np.pad(Zxx,((0,0),(0,output_shape[1]-w)),mode="constant",constant_values=0)
    else:
        Zxx=Zxx[:,0:output_shape[1]]

    return Zxx


def resample(x,fs_in,fs_out):
    n_out=int(len(x)*fs_out/fs_in)
    sig_out=signal.resample(x,n_out)
    return sig_out


def gaus_noise(x1,x2=None,min_sd=0.00001,max_sd=0.01,p=0.5):
    if np.random.rand()<p:
        if x2 is not None:
            return x1,x2
        return x1
    sds=np.logspace(np.log10(min_sd),np.log10(max_sd),num=1000)
    sd=np.random.choice(sds,size=1)
    if x2 is None:
        return x1+np.random.normal(0,sd,len(x1))
    else:
        return x1+np.random.normal(0,sd,len(x1)),x2+np.random.normal(0,sd,len(x2))

def permute(x1,x2=None,n_segments=5,p=0.5):
    assert len(x1) % n_segments == 0
    if np.random.rand()<p:
        if x2 is not None:
            return x1,x2
        return x1
    l=len(x1)

    l_segment=l//n_segments
    i_segments=[i*l_segment for i in range(n_segments)]
    order_segments = np.random.permutation(range(n_segments))
    x1_segments=[x1[i:i+l_segment] for i in i_segments]
    x1_new = [x1_segments[i] for i in order_segments]
    x1_new=np.concatenate(x1_new)
    if x2 is not None:
        x2_segments=[x2[i:i+l_segment] for i in i_segments]
        x2_new = [x2_segments[i] for i in order_segments]
        x2_new = np.concatenate(x2_new)
        return x1_new,x2_new
    return x1_new



