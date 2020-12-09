import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio
from settings import data_dir
import os
from models.cnn.networks import groupNorm,resnet1d,Net,wideresnet50
from models.selfsupervised.losses import NTXentLoss
from datasets.signals import stft, gaus_noise
from functools import partial

from datasets.loaders import TriagePairs,TriageDataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from settings import Fs
import joblib
from settings import weights_dir

display = os.environ.get('DISPLAY', None) is not None

lr_=0.001
l2_=5e-3
representation_size=64
output_size=128
temp=0.01
batch_size=64

weights_file=os.path.join(weights_dir,f"simclr_lr{lr_}_l2{l2_}_z{representation_size}_x{output_size}_temp{temp}.pt")


fs = Fs
slice_sec = 2.5
slide_sec = 0.2

nperseg = int(slice_sec * fs)
step = int(slide_sec * fs)
noverlap = nperseg - step

triage_segments = pd.read_csv(os.path.join(data_dir, 'triage/segments.csv'))


train_ids, test_ids = joblib.load(os.path.join(data_dir, "triage/ids.joblib"))
np.random.seed(123)
train_encoder_ids=np.random.choice(train_ids,size=660,replace=False)

train = triage_segments[triage_segments['id'].isin(train_ids)]
non_repeated_ids = [k for k, v in train['id'].value_counts().items() if v <= 15]
train = train[~train['id'].isin(non_repeated_ids)]  # remove non repeating ids
train_encoder = train[train['id'].isin(train_encoder_ids)]
train_classifier = train[~train['id'].isin(train_encoder_ids)]
test = triage_segments[triage_segments['id'].isin(test_ids)]


sfft_fun = partial(stft, fs=fs, nperseg=nperseg, noverlap=noverlap, spec_only=True)

train_transformation = nn.Sequential(
    torchaudio.transforms.TimeMasking(time_mask_param=5),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=3),
)

aug_gausian = partial(gaus_noise, min_sd=1e-5, max_sd=1e-1)

encoder_train_dataset = TriagePairs(train_encoder, id_var="id", stft_fun=sfft_fun,
                            transforms=train_transformation, aug_raw=aug_gausian)
encoder_train_loader = DataLoader(encoder_train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)

encoder_test_dataset = TriagePairs(train_classifier, id_var="id", stft_fun=sfft_fun, aug_raw=None)
encoder_test_loader = DataLoader(encoder_test_dataset, batch_size=64, shuffle=False, num_workers=5)

classifier_train_dataset=TriageDataset(train_encoder,stft_fun=sfft_fun)
classifier_train_loader=DataLoader(classifier_train_dataset,batch_size=64,shuffle=False)

classifier_test_dataset=TriageDataset(test,stft_fun=sfft_fun)
classifier_test_loader=DataLoader(classifier_test_dataset,batch_size=64,shuffle=False)

iter_ = iter(encoder_train_loader)
x1_raw, x1_stft, x2_raw, x2_stft = iter_.next()
print("Input shape", x1_raw.shape, " and ", x1_stft.shape)

if display:
    fig, axs = plt.subplots(4, 3, figsize=(12, 5))
    for r in range(2):
        for c in range(3):
            axs[r * 2 + 1, c].imshow(x1_stft.numpy()[c, r, :, :],
                                     # vmin=batch_x.numpy()[:, r, :, :].min(),
                                     # vmax=batch_x.numpy()[:, r, :, :].max(),
                                     norm=colors.LogNorm()
                                     )
            axs[r * 2 + 1, c].invert_yaxis()
    for r in range(2):
        for c in range(3):
            axs[r * 2, c].plot(x1_raw.numpy()[c, r, :],
                               # vmin=batch_x.numpy()[:, r, :, :].min(),
                               # vmax=batch_x.numpy()[:, r, :, :].max(),
                               # norm=colors.LogNorm()
                               )

    plt.show()


device="cuda" if torch.cuda.is_available() else "cpu"
# model=Classifier(in_features=2,hid_dim=64,z_dim=64)

model_raw=resnet1d(num_classes=1,norm_layer=groupNorm(8))
model_stft=wideresnet50(num_classes=1,norm_layer=groupNorm(8))
model=Net(raw_model=model_raw,stft_model=model_stft,
          representation_size=representation_size,num_classes=output_size).to(device)
# optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001 ,weight_decay=1e-4)
# optimizer=torch.optim.Adam(params=model.parameters(),lr=lr_ ,weight_decay=l2_)
optimizer=torch.optim.SGD(params=model.parameters(),lr=lr_ ,weight_decay=l2_,momentum=0.99)
criterion=NTXentLoss(device=device, temperature=temp, use_cosine_similarity=True).to(device)
epochs=500
scheduler=ReduceLROnPlateau(optimizer,factor=0.5,patience=10,verbose=True,min_lr=1e-5)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-1,epochs=epochs,steps_per_epoch=len(train_loader))

epoch=0
best_model={'loss':np.Inf,'params':None}
early_stop_counter=0
losses=[]

