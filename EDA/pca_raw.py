import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from settings import data_dir
import os
from sklearn.decomposition import PCA
import sklearn.manifold as manifold
from sklearn.preprocessing import StandardScaler
from functools import partial
from datasets.signals import stft
from datasets.loaders import TriageDataset
import matplotlib.pyplot as plt
from matplotlib import cm
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

dataset=TriageDataset(triage_segments)
dataset_loader=DataLoader(dataset,batch_size=64,shuffle=False)

classifier_test_dataset=TriageDataset(test,stft_fun=sfft_fun)
classifier_test_loader=DataLoader(classifier_test_dataset,batch_size=64,shuffle=False)






test_embedding=[]
with torch.no_grad():
    for x1_raw in tqdm(dataset_loader):

        test_embedding.append(x1_raw.numpy())

test_embedding=np.concatenate(test_embedding)
test_embedding=test_embedding.reshape((test_embedding.shape[0],-1))

scl=StandardScaler()
scl_test_embedding=scl.fit_transform(test_embedding)

pca=PCA(n_components=32)
test_pca=pca.fit_transform(scl_test_embedding)



test_identifier=triage_segments['id']
subject_embeddings=[]
subject_ids=[]
subject_admitted=[]
subject_died=[]
subject_resp_rate=[]
subject_spo2=[]
subject_hb=[]
subject_hr=[]
for id in test_identifier.unique():
    temp_=test_pca[test_identifier==id]
    subject_embeddings.append(temp_.mean(axis=0))
    subject_ids.append(id)
    subject_admitted.append(triage_segments.loc[triage_segments['id']==id,'admitted'].iloc[0])
    subject_died.append(triage_segments.loc[triage_segments['id'] == id, 'died'].iloc[0])
    subject_resp_rate.append(triage_segments.loc[triage_segments['id'] == id, 'resp_rate'].iloc[0])
    subject_spo2.append(triage_segments.loc[triage_segments['id'] == id, 'spo2'].iloc[0])
    subject_hb.append(triage_segments.loc[triage_segments['id'] == id, 'hb'].iloc[0])
    subject_hr.append(triage_segments.loc[triage_segments['id'] == id, 'hr'].iloc[0])


subject_embeddings=np.stack(subject_embeddings)
scl=StandardScaler()
subject_scl=scl.fit_transform(subject_embeddings)

subject_admitted=np.array(subject_admitted)
subject_died=np.array(subject_died)
subject_resp_rate=np.array(subject_resp_rate)

subject_spo2=np.array(subject_spo2); subject_spo2[subject_spo2<65]=np.nan
subject_hb=np.array(subject_hb)
subject_hr=np.array(subject_hr)


def plot_embedding(X,y=None,categorical=False, title=None,ax=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    X = X[~np.isnan(y), :]
    y = y[~np.isnan(y)]

    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1],
                    color="red" if y[i]==1.0 else "blue",
                    alpha=0.5,)

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    ax.set_xticks([]), ax.set_yticks([])
    # # plt.legend()
    if title is not None:
        ax.set_title(title)

def plot_embedding2(X,y=None, title=None,cmap=cm.hot,ax=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    X=X[~np.isnan(y),:]
    y=y[~np.isnan(y)]

    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1],
                    color=cmap((y[i]-np.nanmin(y))/(np.nanmax(y)-np.nanmin(y))),
                    # color=cm.hot(y[i]),
                    alpha=0.5,)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    ax.set_xticks([]), ax.set_yticks([])
    # # plt.legend()
    if title is not None:
        ax.set_title(title)


tsne = manifold.TSNE(n_components=3, init='pca', random_state=0,perplexity=100,learning_rate=100)
X_tsne = tsne.fit_transform(subject_scl)

fig, axs=plt.subplots(1,3,figsize=(12,4))

# plot_embedding(X_tsne,subject_admitted,title="Admission",ax=axs[0][0])
# # plt.show()

# plot_embedding(X_tsne,subject_died,title="Death")
# plt.show()

plot_embedding2(X_tsne,subject_resp_rate,title="Respiratory rate",ax=axs[0])
# plt.show()

plot_embedding2(X_tsne,subject_hr,title="Heart rate",ax=axs[1])
# plt.show()

plot_embedding2(X_tsne,subject_spo2,title="SPO2",ax=axs[2])
plt.show()



test_admitted=triage_segments['admitted']
for a in [0, 1]:
    plt.scatter(test_pca[test_admitted == a, 0], test_pca[test_admitted == a, 1],
                marker="o", label="Yes" if a else "No",
                alpha=0.5)
plt.legend()
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
