import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from scipy import signal
import torch
from torchaudio.transforms import Spectrogram
import librosa
import librosa.display


segments=io.loadmat("triage.mat")

id=6
fs=segments['Fs'][0][0]

sig=segments['red'][id,:]

fig,ax=plt.subplots(2,1)
ax[0].plot(segments['red'][6,:])
ax[0].set_title("Red")
ax[1].plot(segments['infrared'][6,:])
ax[1].set_title("Infrared")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()





slice_sec=3.0
slide_sec=1.5

n_fft=int(slice_sec*fs)
step=int(slide_sec*fs)
overlap=n_fft-step

rect = patches.Rectangle((0.0,0.1),n_fft,0.01,linewidth=1,edgecolor='r',facecolor='none',linestyle="--")
arrow=patches.Arrow(202,0.102,100,0,width=0.0001,color='r')
fig,ax=plt.subplots(1)
ax.plot(sig)
ax.add_patch(rect)
ax.add_patch(arrow)
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()



f, t, Zxx=signal.stft(sig,fs=fs,window='hann',nperseg=n_fft,noverlap=overlap,boundary=None,padded=False)
Zxx.shape

power=np.log(np.abs(Zxx))

# rect = patches.Rectangle((0.0,0.0),1.5,40,linewidth=1.1,edgecolor='r',facecolor='none',linestyle="--")
fig,ax=plt.subplots(1)
ax.pcolormesh(t, f, power)
# ax.add_patch(rect)
plt.ylim(0,5)
#plt.colorbar()
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()

spec=torch.stft(torch.tensor(sig),n_fft=n_fft,hop_length=step,win_length=20,center=False)
spec.shape

spec2=Spectrogram(n_fft=n_fft,hop_length=step,normalized=True,pad=0)(torch.tensor(sig))
spec2.shape
plt.imshow(spec2,norm=colors.LogNorm())
plt.colorbar()
plt.show()


spec_librosa=librosa.core.stft(sig,n_fft=n_fft,hop_length=step,center=False,win_length=20)
spec_librosa.shape
librosa.display.specshow(librosa.amplitude_to_db(spec_librosa, ref=np.max),sr=fs, y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

plt.plot(sig)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#data augmentation

fig,ax=plt.subplots(2,1)
ax[0].plot(sig)
ax[0].set_title("Original signal")
ax[0].set_ylabel("Amplitude")
ax[1].plot(sig+np.random.randn(len(sig))*0.001)
ax[1].set_title("Signal corrupted with gaussian noise")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()


f, t, Zxx=signal.stft(sig,fs=fs,window='hann',nperseg=n_fft,noverlap=overlap,boundary=None,padded=False)
Zxx.shape
power=np.log(np.abs(Zxx))
power2=power.copy()
power2[np.where(np.logical_and(f>1,f<2))[0],:]=np.nan
fig,ax=plt.subplots(2,1)
ax[0].pcolormesh(t, f, power)
ax[0].set_ylim(0,5)
ax[0].set_title("Original spectrogram")
ax[0].set_ylabel("Frequency")
ax[1].pcolormesh(t, f, power2)
ax[1].set_ylim(0,5)
ax[1].set_title("Spectrogram with frequency masking")

plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()