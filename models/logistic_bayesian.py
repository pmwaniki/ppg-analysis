import numpy as np

import joblib
import pandas as pd
from pyro.infer import SVI, Trace_ELBO, Predictive, NUTS, MCMC
from pyro.nn import PyroModule, PyroSample
from sklearn.metrics import roc_auc_score,confusion_matrix,plot_confusion_matrix

from sklearn.preprocessing import StandardScaler,PolynomialFeatures



import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist

from settings import data_dir,output_dir
import os
import matplotlib.pyplot as plt

from utils import admission_confusion_matrix, admission_distplot

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
predictors=['Weight (kgs)','Irritable/restlessness ','MUAC (Mid-upper arm circumference) cm',
            'Can drink / breastfeed?','spo_transformed','Temperature (degrees celsius)',
            'Difficulty breathing','Heart rate(HR) ']

predictors_oximeter=['Heart rate(HR) ','spo_transformed',]

for p in predictors:
    if data[p].dtype==np.float:
        median=train[p].median()
        train[p]=train[p].fillna(median)
        test[p] = test[p].fillna(median)
    else:
        majority=train[p].value_counts().index[0]
        train[p]=train[p].fillna(majority)
        test[p] = test[p].fillna(majority)

train['Can drink / breastfeed?']=train['Can drink / breastfeed?'].map({"Yes":"No","No":"Yes"})

train_x=pd.get_dummies(train[predictors],drop_first=True)
test_x=pd.get_dummies(test[predictors],drop_first=True)

scl=StandardScaler()
x_data_clinical=torch.tensor(scl.fit_transform(train_x),dtype=torch.float)
x_data_test_clinical=torch.tensor(scl.transform(test_x),dtype=torch.float)
x_data_ppg=torch.tensor(scl.fit_transform(classifier_embedding_reduced),dtype=torch.float)
x_data_test_ppg=torch.tensor(scl.transform(test_embedding_reduced),dtype=torch.float)
# x_data=scl.fit_transform(np.concatenate([classifier_embedding_reduced,train_x],axis=1))
# x_data_test=scl.transform(np.concatenate([test_embedding_reduced,test_x],axis=1))
# poly=PolynomialFeatures(degree=1,include_bias=False)
# x_data=poly.fit_transform(x_data)
# x_data_test=poly.transform(x_data_test)

y_data=torch.Tensor(admitted_train)

# class LinearModel(nn.Module):
#     def __init__(self,dim_x):
#         super(LinearModel, self).__init__()
#         self.fc=nn.Linear(dim_x,1)
#         self.activation=nn.Sigmoid()
#     def forward(self,x):
#         return self.activation(self.fc(x))



# linear_model=PyroModule[LinearModel](train_x.shape[1])
# def model(obs_x,obs_y=None):
#     p=obs_x.shape[1]
#     alpha=pyro.sample('alpha',dist.Normal(torch.tensor(0.0),1.0))
#     betas=pyro.sample('betas',dist.Normal(torch.zeros(p),torch.ones(p)).to_event(1))
#     # pyro.module('linear_model',linear_model)
#     with pyro.plate("data"):
#         # y_pred = linear_model(obs_x)
#         y_pred=alpha+obs_x@betas.reshape((-1,1))
#         pyro.sample('y', dist.Bernoulli(logits=y_pred).to_event(1), obs=obs_y)
# def mlp_block(infeatures,out_features,dim_hidden,activation=nn.LeakyReLU(0.2)):
#     layers=[]
#     sizes=[infeatures,] + dim_hidden + [out_features]
#     for i,s in enumerate(sizes):
#         if i==0:
#             continue
#         elif i<len(sizes-1):
#             pass

class BayesianMLP(PyroModule):
    def __init__(self, in_features, out_features=1,dim_hidden=128):
        super().__init__()
        self.linear1 = PyroModule[nn.Linear](in_features, dim_hidden)
        self.linear2 = PyroModule[nn.Linear](dim_hidden, out_features)
        self.linear1.weight = PyroSample(dist.Normal(0., 1.).expand([dim_hidden, in_features]).to_event(2))
        self.linear1.bias = PyroSample(dist.Normal(0., 10.).expand([dim_hidden]).to_event(1))
        self.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, dim_hidden]).to_event(2))
        self.linear2.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))
        self.activation=nn.LeakyReLU(0.1)

    def forward(self, x, y=None):
        # sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        out=self.activation(self.linear1(x))
        mean=pyro.deterministic('mean',value=self.linear2(out).sigmoid())
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
        return mean

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.0).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        # sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean=pyro.deterministic('mean',value=self.linear(x).sigmoid())
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
        return mean

class ScaleModel(PyroModule):
    def __init__(self, in_features):
        super().__init__()
        self.linear_alpha = PyroModule[nn.Linear](in_features, 1)
        self.linear_alpha.weight = PyroSample(dist.Normal(0., 1.).expand([1, in_features]).to_event(2))
        self.linear_alpha.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))
        self.linear_beta = PyroModule[nn.Linear](in_features, 1)
        self.linear_beta.weight = PyroSample(dist.Normal(0., 1.).expand([1, in_features]).to_event(2))
        self.linear_beta.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))
        self.activation=nn.ReLU()

    def forward(self, x, y=None):
        # sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        loc = self.linear_alpha(x)
        scale = self.linear_beta(x)

        # mean=pyro.deterministic('mean',value=self.linear(x).sigmoid().squeeze(-1))
        with pyro.plate("data", x.shape[0]):
            # logits = pyro.sample('logits', dist.Normal(loc, torch.exp(scale)).to_event(1))
            # mean = pyro.deterministic('mean', logits.sigmoid())
            mean=pyro.sample('mean',dist.Beta(torch.exp(loc),torch.exp(scale)).to_event(1))
            obs = pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
        return mean

# model_clinical=ScaleModel(in_features=x_data_clinical.shape[1],)
model_clinical=BayesianMLP(in_features=x_data_clinical.shape[1],)
nuts_kernel=NUTS(pyro.poutine.block(model_clinical,hide=['logits','mean']),)
mcmc=MCMC(nuts_kernel,num_samples=5000,warmup_steps=1000,)
mcmc.run(x_data_clinical,y_data.reshape(-1,1))
posterior_samples=mcmc.get_samples()
predictive=Predictive(model_clinical,posterior_samples)
samples_clinical=predictive(x_data_test_clinical,None)
# # linear_guide=pyro.infer.autoguide.AutoDiagonalNormal(linear_model)
# guide_clinical=pyro.infer.autoguide.AutoGaussian(pyro.poutine.block(model_clinical,hide=['obs','mean']))
# # def null_guide(obs_x,obs_y):
# #     pass
#
# adam_clinical = pyro.optim.Adam({"lr": 0.0001})
# svi_clinical = SVI(model_clinical, guide_clinical, adam_clinical, loss=Trace_ELBO())
#
# pyro.clear_param_store()
# for j in range(100000):
#     # calculate the loss and take a gradient step
#     loss = svi_clinical.step(x_data_clinical, y_data.reshape(-1,1))
#     if j % 1000 == 0:
#         print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data_clinical)))
#
#
# predictive_clinical = Predictive(model_clinical, guide=guide_clinical, num_samples=1000,
#                         return_sites=( 'mean',))
# samples_clinical = predictive_clinical(x_data_test_clinical ,)
test_pred_clinical=samples_clinical['mean'].mean(dim=0).reshape(-1).numpy()
roc_auc_score(admitted_test,test_pred_clinical)


base_rate=admitted_train.mean()
pred_positive_clinical=(test_pred_clinical>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_clinical)

admission_distplot(samples_clinical['mean'].squeeze(),admitted_test,pred_positive_clinical)

###############################################################################################################


###########################################################################################################
model_ppg=BayesianMLP(in_features=x_data_ppg.shape[1],)
nuts_kernel=NUTS(pyro.poutine.block(model_ppg,hide=['logits','mean']),)
mcmc=MCMC(nuts_kernel,num_samples=5000,warmup_steps=1000,)
mcmc.run(x_data_ppg,y_data.reshape(-1,1))
posterior_samples=mcmc.get_samples()
predictive=Predictive(model_ppg,posterior_samples)
samples_ppg=predictive(x_data_test_ppg,None)
# linear_guide=pyro.infer.autoguide.AutoDiagonalNormal(linear_model)
# guide_ppg=pyro.infer.autoguide.AutoGaussian(pyro.poutine.block(model_ppg,hide=['obs','mean']))
# # def null_guide(obs_x,obs_y):
# #     pass
#
# adam_ppg = pyro.optim.Adam({"lr": 0.0001})
# svi_ppg = SVI(model_ppg, guide_ppg, adam_ppg, loss=Trace_ELBO())
#
# pyro.clear_param_store()
# for j in range(100000):
#     # calculate the loss and take a gradient step
#     loss = svi_ppg.step(x_data_ppg, y_data.reshape(-1,1))
#     if j % 1000 == 0:
#         print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data_ppg)))
#
#
# predictive_ppg = Predictive(model_ppg, guide=guide_ppg, num_samples=1000,
#                         return_sites=( "obs",'mean'),)
# samples_ppg = predictive_ppg(x_data_test_ppg ,)
test_pred_ppg=samples_ppg['mean'].mean(dim=0).reshape(-1).numpy()
roc_auc_score(admitted_test,test_pred_ppg)


base_rate=admitted_train.mean()
pred_positive_ppg=(test_pred_ppg>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_ppg)

displot_ppg=admission_distplot(samples_ppg['mean'].squeeze(),admitted_test,pred_positive_ppg)

samples_both=np.concatenate([samples_clinical['mean'],samples_ppg['mean']],axis=0)
test_pred_both=samples_both.mean(axis=0).reshape(-1)
roc_auc_score(admitted_test,test_pred_both)


base_rate=admitted_train.mean()
pred_positive_both=(test_pred_both>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_both)

distplot_both=admission_distplot(samples_both.squeeze(),admitted_test,pred_positive_both)