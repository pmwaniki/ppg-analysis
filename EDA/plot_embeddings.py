import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import offsetbox
import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn import manifold
from settings import data_dir
import itertools


experiment="Contrastive-sample-DotProduct32b"
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")

data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
triage_segments=pd.read_csv(os.path.join(data_dir,'triage/segments.csv'))
classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)

embeddings=np.concatenate([classifier_embedding,test_embedding])
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

# pca=PCA(n_components=6)
# subject_pca=pca.fit_transform(subject_scl)






# fig,axs=plt.subplots(3,5,figsize=(15,10))
# for ax,vals in zip(axs.flatten(),itertools.combinations(range(6),2)):
#     r,c=vals
#     ax.scatter(subject_pca[subject_admitted == 0, r], subject_pca[subject_admitted == 0, c],
#                                           marker="o", label="No",
#                                           alpha=0.5)
#     ax.scatter(subject_pca[subject_admitted == 1, r], subject_pca[subject_admitted == 1, c],
#                                           marker="o", label="Yes",
#                                           alpha=0.5)
#     ax.set_xlabel(f"PCA {r + 1}")
#     ax.set_ylabel(f"PCA {c + 1}")
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# fig.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_{os.uname()[1]}_{experiment}.png")
# plt.show(block=False)



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


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=100,learning_rate=100)
X_tsne = tsne.fit_transform(subject_scl)

fig, axs=plt.subplots(1,3,figsize=(12,4))

# plot_embedding(X_tsne,subject_admitted,title="Admission",ax=axs[0][0])
# # plt.show()

# plot_embedding(X_tsne,subject_died,title="Death")
# plt.show()

plot_embedding2(X_tsne,subject_resp_rate,title="Respiratory rate",ax=axs[1])
# plt.show()

plot_embedding2(X_tsne,subject_hr,title="Heart rate",ax=axs[0])
# plt.show()

plot_embedding2(X_tsne,subject_spo2,title="SPO2",ax=axs[2])
plt.show()

# plot_embedding2(X_tsne,subject_hb,title="HB",cmap=cm.summer)
# plt.show()

fig,axs=plt.subplots(1,3,figsize=(15,10))
for ax,vals in zip(axs.flatten(),itertools.combinations(range(3),2)):
    r,c=vals
    ax.scatter(X_tsne[subject_admitted == 0, r], X_tsne[subject_admitted == 0, c],
                                          marker="o", label="No",
                                          alpha=0.7)
    ax.scatter(X_tsne[subject_admitted == 1, r], X_tsne[subject_admitted == 1, c],
                                          marker="o", label="Yes",
                                          alpha=0.7)
    ax.set_xlabel(f"Component {r + 1}")
    ax.set_ylabel(f"Component {c + 1}")
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_tsne_{os.uname()[1]}_{experiment}.png")
plt.show(block=False)

### PCA
