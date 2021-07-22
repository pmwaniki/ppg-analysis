import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from settings import data_dir,output_dir
import os
from sklearn.decomposition import PCA
import sklearn.manifold as manifold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from functools import partial
from datasets.loaders import TriageDataset
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import joblib
import gc

representation_size=32
batch_size=64
include_sepsis=False

experiment=f"PCA-32"
if include_sepsis:
    experiment=experiment+"-sepsis"
    sepsis_segments1 = pd.read_csv(os.path.join(data_dir, "segments-sepsis.csv"))
    sepsis_segments2 = pd.read_csv(os.path.join(data_dir, "segments-sepsis_0m.csv"))
    sepsis = pd.concat([sepsis_segments1, sepsis_segments2])
    del sepsis_segments1
    del sepsis_segments2
    sepsis['id'] = sepsis['id'] + "-" + sepsis['episode']




triage_segments = pd.read_csv(os.path.join(data_dir, 'triage/segments.csv'))

train_ids, test_ids = joblib.load(os.path.join(data_dir, "triage/ids.joblib"))


train = triage_segments[triage_segments['id'].isin(train_ids)]
test = triage_segments[triage_segments['id'].isin(test_ids)]

if include_sepsis:
    sepsis_dataset=TriageDataset(sepsis)
    sepsis_loader=DataLoader(sepsis_dataset,batch_size=64,shuffle=False)
    sepsis_segments = []
    for x1_raw in tqdm(sepsis_loader):
        sepsis_segments.append(x1_raw.numpy())

    sepsis_segments=np.concatenate(sepsis_segments)
    sepsis_segments = sepsis_segments.reshape((sepsis_segments.shape[0], -1))

train_dataset=TriageDataset(train)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=False)

test_dataset=TriageDataset(test)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)





train_segments=[]
for x1_raw in tqdm(train_loader):
    train_segments.append(x1_raw.numpy())

test_segments=[]
for x1_raw in tqdm(test_loader):
    test_segments.append(x1_raw.numpy())

train_segments=np.concatenate(train_segments)
test_segments=np.concatenate(test_segments)

train_segments=train_segments.reshape((train_segments.shape[0],-1))
test_segments=test_segments.reshape((test_segments.shape[0],-1))


clf=Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=32))])

if include_sepsis:
    all_train_segments=np.concatenate([sepsis_segments,train_segments])
    del sepsis_segments
    gc.collect()
else:
    all_train_segments=train_segments

clf.fit(all_train_segments)

train_embeddings=clf.transform(train_segments)
test_embeddings=clf.transform(test_segments)

embeddings=np.concatenate([train_embeddings,test_embeddings])
all_data=pd.concat([train,test])

test_identifier=all_data['id']
subject_embeddings=[]
subject_ids=[]
subject_admitted=[]
subject_died=[]
subject_resp_rate=[]
subject_spo2=[]
subject_hb=[]
subject_hr=[]
for id in test_identifier.unique():
    temp_=embeddings[test_identifier==id]
    subject_embeddings.append(temp_.mean(axis=0))
    subject_ids.append(id)
    subject_admitted.append(all_data.loc[all_data['id']==id,'admitted'].iloc[0])
    subject_died.append(all_data.loc[all_data['id'] == id, 'died'].iloc[0])
    subject_resp_rate.append(all_data.loc[all_data['id'] == id, 'resp_rate'].iloc[0])
    subject_spo2.append(all_data.loc[all_data['id'] == id, 'spo2'].iloc[0])
    subject_hb.append(all_data.loc[all_data['id'] == id, 'hb'].iloc[0])
    subject_hr.append(all_data.loc[all_data['id'] == id, 'hr'].iloc[0])


subject_embeddings=np.stack(subject_embeddings)
scl=StandardScaler()
subject_scl=scl.fit_transform(subject_embeddings)

subject_admitted=np.array(subject_admitted)
subject_died=np.array(subject_died)
subject_resp_rate=np.array(subject_resp_rate)

subject_spo2=np.array(subject_spo2); subject_spo2[subject_spo2<65]=np.nan
subject_hb=np.array(subject_hb)
subject_hr=np.array(subject_hr)

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


# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=100,learning_rate=100)
# X_tsne = tsne.fit_transform(subject_scl)

fig, axs=plt.subplots(1,3,figsize=(12,4))


plot_embedding2(subject_embeddings,subject_resp_rate,title="Respiratory rate",ax=axs[1])

plot_embedding2(subject_embeddings,subject_hr,title="Heart rate",ax=axs[0])

plot_embedding2(subject_embeddings,subject_spo2,title="SPO2",ax=axs[2])
plt.savefig(os.path.join(output_dir,f"Dimensionality reduction-{experiment}.png"))
plt.show()


joblib.dump((train_embeddings,test_embeddings,train,test),
            os.path.join(data_dir,f"results/{experiment}.joblib"))
