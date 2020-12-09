import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from settings import data_dir
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from functools import partial
from datasets.signals import stft
from datasets.loaders import TriageDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from settings import Fs
import joblib

fs=Fs

slice_sec=2.5
slide_sec=0.2
dropout_=0.5
l2_=5e-4
lr_=0.01
pos_weight=7.0
representation_size=128
batch_size=64

nperseg=int(slice_sec*fs)
step=int(slide_sec*fs)
noverlap=nperseg-step



triage_segments = pd.read_csv(os.path.join(data_dir, 'triage/segments.csv'))
non_repeated_ids = [k for k, v in triage_segments['id'].value_counts().items() if v == 1]
triage_segments = triage_segments[~triage_segments['id'].isin(non_repeated_ids)]  # remove non repeating ids

train_ids, test_ids = joblib.load(os.path.join(data_dir, "triage/ids.joblib"))
np.random.seed(123)
train_encoder_ids=np.random.choice(train_ids,size=660,replace=False)

train = triage_segments[triage_segments['id'].isin(train_ids)]
train_encoder = train[train['id'].isin(train_encoder_ids)]
train_classifier = train[~train['id'].isin(train_encoder_ids)]
test = triage_segments[triage_segments['id'].isin(test_ids)]



sfft_fun = partial(stft, fs=fs, nperseg=nperseg, noverlap=noverlap, spec_only=True)



classifier_test_dataset=TriageDataset(test,stft_fun=sfft_fun)
classifier_test_loader=DataLoader(classifier_test_dataset,batch_size=64,shuffle=False)






test_embedding=[]
with torch.no_grad():
    for x1_raw, x1_stft in tqdm(classifier_test_loader):

        test_embedding.append(x1_raw.numpy())

test_embedding=np.concatenate(test_embedding)
test_embedding=test_embedding.reshape((test_embedding.shape[0],-1))

scl=StandardScaler()
scl_test_embedding=scl.fit_transform(test_embedding)

pca=PCA(n_components=15)
test_pca=pca.fit_transform(scl_test_embedding)


test_admitted=test['admitted']
for a in [0, 1]:
    plt.scatter(test_pca[test_admitted == a, 0], test_pca[test_admitted == a, 1],
                marker="o", label="Yes" if a else "No",
                alpha=0.5)
plt.legend()
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
