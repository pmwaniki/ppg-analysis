from functools import partial

import numpy as np

import joblib
import pandas as pd
import pyro.contrib.gp as gp
from pyro.contrib.easyguide import easy_guide, EasyGuide
from pyro.distributions import constraints
import pyro.distributions.transforms as T
from pyro.infer import SVI, Trace_ELBO, Predictive, NUTS, MCMC
from pyro.infer.autoguide import AutoGuide, init_to_sample, init_to_uniform, init_to_feasible
from pyro.nn import PyroModule, PyroSample, PyroParam, DenseNN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score,confusion_matrix,plot_confusion_matrix
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,PolynomialFeatures



import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

import pyro
import pyro.distributions as dist
from pyro import poutine

from settings import data_dir,output_dir,clinical_predictors
import os
import matplotlib.pyplot as plt

from utils import admission_confusion_matrix, admission_distplot
pyro.set_rng_seed(123)
device='cuda' if torch.cuda.is_available() else "cpu"
# device='cpu'

# experiment="Contrastive-original-sample-DotProduct32"
experiment="Contrastive-original-sample-DotProduct32-sepsis"
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")
classifier_embedding,test_embedding,seg_train,seg_test=joblib.load(experiment_file)
train_ids=seg_train['id'].unique()
test_ids=seg_test['id'].unique()

classifier_embedding_reduced=np.stack(map(lambda id:classifier_embedding[seg_train['id']==id,:].mean(axis=0) ,train_ids))
test_embedding_reduced=np.stack(map(lambda id:test_embedding[seg_test['id']==id,:].mean(axis=0) ,test_ids))
admitted_train=np.stack(map(lambda id:seg_train.loc[seg_train['id']==id,'admitted'].iat[0],train_ids))
admitted_test=np.stack(map(lambda id:seg_test.loc[seg_test['id']==id,'admitted'].iat[0],test_ids))



data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
# feature engineering
data['spo_transformed'] = 4.314 * np.log10(103.711-data['Oxygen saturation']) - 37.315

train=data.iloc[[np.where(data['Study No']==id)[0][0] for id in train_ids],:].copy()
test=data.iloc[[np.where(data['Study No']==id)[0][0] for id in test_ids],:].copy()





#Imputation
clinical_predictors=['Weight (kgs)','Irritable/restlessness ','MUAC (Mid-upper arm circumference) cm',
            'Can drink / breastfeed?','spo_transformed','Temperature (degrees celsius)',
            'Difficulty breathing','Heart rate(HR) ']

# predictors_oximeter=['Heart rate(HR) ','spo_transformed',]

for p in clinical_predictors:
    if data[p].dtype==np.float:
        median=train[p].median()
        train[p]=train[p].fillna(median)
        test[p] = test[p].fillna(median)
    else:
        categories=pd.unique(train[p].dropna())
        majority=train[p].value_counts().index[0]
        train[p]=train[p].fillna(majority)
        test[p] = test[p].fillna(majority)
        train[p]=pd.Categorical(train[p],categories=categories)
        test[p] = pd.Categorical(test[p], categories=categories)

# train['Can drink / breastfeed?']=train['Can drink / breastfeed?'].map({"Yes":"No","No":"Yes"})

train_x=pd.get_dummies(train[clinical_predictors],drop_first=True)
test_x=pd.get_dummies(test[clinical_predictors],drop_first=True)

preprocess_clinical=Pipeline([
    ('scl',StandardScaler()),
    # ('pca',PCA(n_components=8)),
    # ('poly',PolynomialFeatures(degree=2,include_bias=False,interaction_only=True))
])
preprocess_ppg=Pipeline([
    ('scl',StandardScaler()),
    # ('pca',PCA(n_components=16)),
    ('poly',PolynomialFeatures(degree=2,include_bias=False,interaction_only=True))
])
preprocess_concat=Pipeline([
    ('scl',StandardScaler()),
    # ('pca',PCA(n_components=16)),
    # ('poly',PolynomialFeatures(degree=2,include_bias=False,interaction_only=True))
])
x_data_clinical=torch.tensor(preprocess_clinical.fit_transform(train_x),dtype=torch.float)
x_data_test_clinical=torch.tensor(preprocess_clinical.transform(test_x),dtype=torch.float)
x_data_ppg=torch.tensor(preprocess_ppg.fit_transform(classifier_embedding_reduced),dtype=torch.float)
x_data_test_ppg=torch.tensor(preprocess_ppg.transform(test_embedding_reduced),dtype=torch.float)

fit_ppg=Pipeline([
    ('scl',StandardScaler()),
    # ('clf',LinearDiscriminantAnalysis(n_components=1))
    # ('pca',PCA(n_components=16)),
    # ('poly',PolynomialFeatures(degree=2,include_bias=False,interaction_only=True))
])

ppg_compressed_train=fit_ppg.fit_transform(x_data_ppg.cpu().numpy(),admitted_train)
ppg_compressed_test=fit_ppg.transform(x_data_test_ppg.cpu().numpy())

x_data_concat=torch.tensor(preprocess_concat.fit_transform(np.concatenate([train_x.values,ppg_compressed_train],axis=1)),dtype=torch.float,device=device)
x_data_test_concat=torch.tensor(preprocess_concat.fit_transform(np.concatenate([test_x.values,ppg_compressed_test],axis=1)),dtype=torch.float,device=device)
# x_data=scl.fit_transform(np.concatenate([classifier_embedding_reduced,train_x],axis=1))
# x_data_test=scl.transform(np.concatenate([test_embedding_reduced,test_x],axis=1))
# poly=PolynomialFeatures(degree=1,include_bias=False)
# x_data=poly.fit_transform(x_data)
# x_data_test=poly.transform(x_data_test)

y_data=torch.Tensor(admitted_train).to(device)
x_data_clinical,x_data_ppg,x_data_concat=x_data_clinical.to(device),x_data_ppg.to(device),x_data_concat.to(device)
x_data_test_clinical,x_data_test_ppg,x_data_test_concat=x_data_test_clinical.to(device),x_data_test_ppg.to(device),x_data_test_concat.to(device)

#######################################################################################################################
# CLINICAL
####################################################################################################################

# kernel_clinical = gp.kernels.RBF(input_dim=x_data_clinical.shape[1])
kernel_clinical=gp.kernels.Matern52(input_dim=x_data_clinical.shape[1])
# kernel_clinical.lengthscale=PyroSample(dist.Uniform(torch.tensor(1.0,device=device), torch.tensor(3.0,device=device)))
# kernel_clinical.variance=PyroSample(dist.Uniform(torch.tensor(0.5,device=device), torch.tensor(1.5,device=device)))
pyro.clear_param_store()
likelihood_clinical = gp.likelihoods.Binary()
# Important -- we need to add latent_shape argument here to the number of classes we have in the data
model_clinical = gp.models.VariationalGP(
    x_data_clinical,
    y_data,
    kernel_clinical.to(device),
    likelihood=likelihood_clinical.to(device),
    whiten=True,
    jitter=1e-03,
    # latent_shape=torch.Size([1]),
)
num_steps = 1000
loss = gp.util.train(model_clinical, num_steps=num_steps)
plt.plot(loss)
plt.show()

mean,var=model_clinical(x_data_test_clinical)

samples_clinical=torch.sigmoid(dist.Normal(mean,var).sample([1000,]))
test_pred_clinical=samples_clinical.mean(dim=0).reshape(-1).cpu().numpy()
roc_auc_score(admitted_test,test_pred_clinical)

base_rate=admitted_train.mean()
pred_positive_clinical=(test_pred_clinical>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_clinical)

admission_distplot(samples_clinical.cpu().numpy(),admitted_test,pred_positive_clinical)

#######################################################################################################################
# ppg
####################################################################################################################

kernel_ppg = gp.kernels.Linear(input_dim=x_data_ppg.shape[1])
# kernel_ppg=gp.kernels.RationalQuadratic(input_dim=x_data_ppg.shape[1])
# kernel_ppg.lengthscale=PyroSample(dist.Uniform(torch.tensor(1.0,device=device), torch.tensor(3.0,device=device)))
# kernel_ppg.variance=PyroSample(dist.Uniform(torch.tensor(0.5,device=device), torch.tensor(1.5,device=device)))
# kernel_ppg.autoguide('variance',dist.Normal)
# kernel_ppg.autoguide('lengthscale',dist.Normal)
# kernel_ppg=gp.kernels.Polynomial(input_dim=x_data_ppg.shape[1],degree=1)
pyro.clear_param_store()
likelihood_ppg = gp.likelihoods.Binary()
# Important -- we need to add latent_shape argument here to the number of classes we have in the data
model_ppg = gp.models.VariationalGP(
    x_data_ppg,
    y_data,
    kernel_ppg.to(device),
    likelihood=likelihood_ppg.to(device),
    whiten=True,
    jitter=1e-03,
    # latent_shape=torch.Size([1]),
)
num_steps = 10000
loss = gp.util.train(model_ppg,optimizer=torch.optim.Adam(model_ppg.parameters(),lr=0.001), num_steps=num_steps)
plt.plot(loss)
plt.show()

mean_train,var_train=model_ppg(x_data_ppg)
samples_ppg_train=torch.sigmoid(dist.Normal(mean_train,var_train).sample([1000,]))
train_pred_ppg=samples_ppg_train.mean(dim=0).reshape(-1).cpu().numpy()
roc_auc_score(admitted_train,train_pred_ppg)

samples_ppg_test=list()
for _ in range(1000):
    mean_test,var_test=model_ppg(x_data_test_ppg)
    samples_ppg_test_=torch.sigmoid(dist.Normal(mean_test,var_test).sample([1,]))
    samples_ppg_test.append(samples_ppg_test_.cpu().numpy())
samples_ppg_test=np.stack(samples_ppg_test,axis=0).squeeze()

test_pred_ppg=samples_ppg_test.mean(axis=0)

roc_auc_score(admitted_test,test_pred_ppg)

base_rate=admitted_train.mean()
pred_positive_ppg=(test_pred_ppg>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_ppg)

admission_distplot(samples_ppg_test,admitted_test,pred_positive_ppg)

samples_both=np.concatenate([samples_clinical.cpu().numpy(),samples_ppg_test.cpu().numpy()],axis=0)
test_pred_both=samples_both.mean(axis=0).reshape(-1)
roc_auc_score(admitted_test,test_pred_both)

#######################################################################################################################
# concat
####################################################################################################################
# scl_=StandardScaler()
def logits(x):
    return np.log(x/(1-x))
x_data_concat=torch.cat([x_data_clinical,torch.tensor(logits(train_pred_ppg.reshape(-1,1)),device=device)],dim=1)
x_data_test_concat=torch.cat([x_data_test_clinical,torch.tensor(logits(test_pred_ppg.reshape(-1,1)),device=device)],dim=1)

kernel_concat = gp.kernels.RBF(input_dim=x_data_concat.shape[1])
pyro.clear_param_store()
likelihood_concat = gp.likelihoods.Binary()
# Important -- we need to add latent_shape argument here to the number of classes we have in the data
model_concat = gp.models.VariationalGP(
    x_data_concat,
    y_data,
    kernel_concat.to(device),
    likelihood=likelihood_concat.to(device),
    whiten=True,
    jitter=1e-03,
    # latent_shape=torch.Size([1]),
)
num_steps = 5000
loss = gp.util.train(model_concat,optimizer=torch.optim.Adam(model_concat.parameters(),lr=0.001), num_steps=num_steps)
plt.plot(loss)
plt.show()

mean_train,var_train=model_concat(x_data_concat)

samples_concat_train=torch.sigmoid(dist.Normal(mean_train,var_train).sample([1000,]))
train_pred_concat=samples_concat_train.mean(dim=0).reshape(-1).cpu().numpy()
roc_auc_score(admitted_train,train_pred_concat)

mean,var=model_concat(x_data_test_concat)

samples_concat=torch.sigmoid(dist.Normal(mean,var).sample([1000,]))
test_pred_concat=samples_concat.mean(dim=0).reshape(-1).cpu().numpy()
roc_auc_score(admitted_test,test_pred_concat)

base_rate=admitted_train.mean()
pred_positive_concat=(test_pred_concat>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_concat)

admission_distplot(samples_concat.cpu().numpy(),admitted_test,pred_positive_concat)