import numpy as np

import joblib
import pandas as pd
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.nn import PyroModule, PyroSample
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler,PolynomialFeatures

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist

from settings import data_dir
import os
import matplotlib.pyplot as plt

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
# x_data=scl.fit_transform(train_x)
# x_data_test=scl.transform(test_x)
# x_data=scl.fit_transform(classifier_embedding_reduced)
# x_data_test=scl.transform(test_embedding_reduced)
x_data=scl.fit_transform(np.concatenate([classifier_embedding_reduced,train_x],axis=1))
x_data_test=scl.transform(np.concatenate([test_embedding_reduced,test_x],axis=1))
poly=PolynomialFeatures(degree=1,include_bias=False)
x_data=poly.fit_transform(x_data)
x_data_test=poly.transform(x_data_test)
x_data,x_data_test=torch.tensor(x_data,dtype=torch.float),torch.tensor(x_data_test,dtype=torch.float)

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

class BayesianMLP(PyroModule):
    def __init__(self, in_features, out_features,dim_hidden=32):
        super().__init__()
        self.linear1 = PyroModule[nn.Linear](in_features, dim_hidden)
        self.linear2 = PyroModule[nn.Linear](dim_hidden, out_features)
        self.linear1.weight = PyroSample(dist.Normal(0., 1.).expand([dim_hidden, in_features]).to_event(2))
        self.linear1.bias = PyroSample(dist.Normal(0., 10.).expand([dim_hidden]).to_event(1))
        self.linear2.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, dim_hidden]).to_event(2))
        self.linear2.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))
        self.activation=nn.Softplus()

    def forward(self, x, y=None):
        # sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        out=self.activation(self.linear1(x))
        mean=pyro.deterministic('mean',value=self.linear2(out).sigmoid().squeeze(-1))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(mean), obs=y)
        return mean

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        # sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean=pyro.deterministic('mean',value=self.linear(x).sigmoid().squeeze(-1))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(mean), obs=y)
        return mean
linear_model=BayesianMLP(in_features=x_data.shape[1],out_features=1)
# linear_guide=pyro.infer.autoguide.AutoDiagonalNormal(linear_model)
linear_guide=pyro.infer.autoguide.AutoMultivariateNormal(linear_model)
# def null_guide(obs_x,obs_y):
#     pass

adam = pyro.optim.Adam({"lr": 0.0001})
svi = SVI(linear_model, linear_guide, adam, loss=Trace_ELBO())

pyro.clear_param_store()
for j in range(100000):
    # calculate the loss and take a gradient step
    loss = svi.step(x_data, y_data)
    if j % 1000 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data)))


predictive = Predictive(linear_model, guide=linear_guide, num_samples=1000,
                        return_sites=( "obs",'mean'),)
samples = predictive(x_data_test ,)
test_pred=samples['mean'].mean(dim=0).reshape(-1).numpy()
roc_auc_score(admitted_test,test_pred)

plt.hist(samples['mean'][100])
plt.show()
# with torch.no_grad():
#     test_pred=linear_model(x_data_test).cpu().numpy().reshape(-1)

