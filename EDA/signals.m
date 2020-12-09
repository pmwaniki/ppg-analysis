close all;
clear all;
clc;

triage=load('triage.mat');
id=10
data=triage.infrared(id,:);

clf
plot(data)


sf = triage.Fs;
Nfft=length(data);
f= (-Nfft/2:Nfft/2-1)*sf/Nfft;


fft_red=fft(data,length(f));

figure
plot(f,20*log10(abs(fftshift(fft_red))))



#[b, a] = butter (n, [Wl, Wh])
#filtered = filter(b,a,data);


# short time fft
pkg load signal
figure

slice=500
step=5
overlap=slice-step
specgram(data,slice,sf,hanning(slice),overlap)
ylim([0 5]);


[S, f2, t]=specgram(data,slice,sf,hanning(slice),overlap);
S = abs(S); S = S/max(S(:));
imagesc (t, f2, flipud(log(S)));
imagesc(t, fliplr(f), flipud(log(S(idx,:))));

