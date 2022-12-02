import matplotlib.pyplot as plt
import os
import json
import sys
import multiprocessing

import joblib
import numpy as np
import pandas as pd
import pyro
import scipy
import torch
# torch.multiprocessing.set_sharing_strategy('file_descriptor')
from pyro.distributions import constraints
import pyro.distributions.transforms as T
from pyro.infer import SVI, Trace_ELBO, Predictive, NUTS, MCMC, autoguide
from pyro.infer.autoguide import AutoGuide, init_to_sample, init_to_uniform, init_to_feasible
from pyro.nn import PyroModule, PyroSample, PyroParam, DenseNN
import pyro.distributions as dist
from pyro import poutine

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,PolynomialFeatures,PowerTransformer
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.linear_model import Ridge,Lasso,LinearRegression,SGDRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression,SelectPercentile,VarianceThreshold
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor
from torch import nn

from settings import data_dir,weights_dir,output_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import save_regression


pyro.set_rng_seed(0)
device='cpu'
num_chains=1
polynomial_degree=2
experiment="Contrastive-original-sample-DotProduct32-sepsis"
# experiment="Contrastive-original-sample-DotProduct32"
# experiment='PCA-32'
# weights_file=os.path.join(weights_dir,f"Contrastive_{experiment}_svm.joblib")
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")

classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)

train_ids=train['id'].unique()
test_ids=test['id'].unique()
classifier_embedding_reduced=np.stack(map(lambda id:classifier_embedding[train['id']==id,:].mean(axis=0) ,train_ids))
test_embedding_reduced=np.stack(map(lambda id:test_embedding[test['id']==id,:].mean(axis=0) ,test_ids))
# admitted_train=np.stack(map(lambda id:train.loc[train['id']==id,'admitted'].iat[0],train_ids))
# admitted_test=np.stack(map(lambda id:test.loc[test['id']==id,'admitted'].iat[0],test_ids))
#
hr_train=np.stack(map(lambda id:train.loc[train['id']==id,'hr'].median(skipna=True),train_ids))
hr_test=np.stack(map(lambda id:test.loc[test['id']==id,'hr'].median(skipna=True),test_ids))
x_train_hr,y_train_hr=classifier_embedding_reduced[hr_train!=0,:],hr_train[hr_train!=0]
x_test_hr,y_test_hr=test_embedding_reduced[hr_test!=0,:],hr_test[hr_test!=0]
preprocess_hr=Pipeline([
    ('poly',PolynomialFeatures(degree=polynomial_degree,include_bias=False)),
    ('scl',StandardScaler()),
    # ('poly',PolynomialFeatures(degree=2,include_bias=False)),
])
preprocess_hr.fit(x_train_hr)
x_train_hr=torch.tensor(preprocess_hr.transform(x_train_hr),dtype=torch.float,device=device)
x_test_hr=torch.tensor(preprocess_hr.transform(x_test_hr),dtype=torch.float,device=device)
y_train_hr=torch.tensor(y_train_hr,device=device,dtype=torch.float)

resp_rate_train=np.stack(map(lambda id:train.loc[train['id']==id,'resp_rate'].median(skipna=True),train_ids))
resp_rate_test=np.stack(map(lambda id:test.loc[test['id']==id,'resp_rate'].median(skipna=True),test_ids))

x_train_resp,y_train_resp=classifier_embedding_reduced[~np.isnan(resp_rate_train),:],resp_rate_train[~np.isnan(resp_rate_train)]
x_test_resp,y_test_resp=test_embedding_reduced[~np.isnan(resp_rate_test),:],resp_rate_test[~np.isnan(resp_rate_test)]
preprocess_resp=Pipeline([
    ('poly',PolynomialFeatures(degree=polynomial_degree,include_bias=False)),
    ('scl',StandardScaler()),
    # ('poly',PolynomialFeatures(degree=2,include_bias=False)),
])
preprocess_resp.fit(x_train_resp)
x_train_resp=torch.tensor(preprocess_resp.transform(x_train_resp),dtype=torch.float,device=device)
x_test_resp=torch.tensor(preprocess_resp.transform(x_test_resp),dtype=torch.float,device=device)
y_train_resp=torch.tensor(y_train_resp,device=device,dtype=torch.float)

spo2_train=np.stack(map(lambda id:train.loc[train['id']==id,'spo2'].median(skipna=True),train_ids))
spo2_test=np.stack(map(lambda id:test.loc[test['id']==id,'spo2'].median(skipna=True),test_ids))

spo2_min=80
x_train_spo2,y_train_spo2=classifier_embedding_reduced[spo2_train>spo2_min,:],spo2_train[spo2_train>spo2_min]
x_test_spo2,y_test_spo2=test_embedding_reduced[spo2_test>spo2_min,:],spo2_test[spo2_test>spo2_min]
preprocess_spo2=Pipeline([
    ('poly',PolynomialFeatures(degree=polynomial_degree,include_bias=False)),
    ('scl',StandardScaler()),
    # ('poly',PolynomialFeatures(degree=2,include_bias=False)),
])
preprocess_spo2.fit(x_train_spo2)
x_train_spo2=torch.tensor(preprocess_spo2.transform(x_train_spo2),dtype=torch.float,device=device)
x_test_spo2=torch.tensor(preprocess_spo2.transform(x_test_spo2),dtype=torch.float,device=device)
y_train_spo2=torch.tensor(y_train_spo2,device=device,dtype=torch.float)

class GammaRegression(PyroModule):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(1.0,device=device)).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(10.,device=device)).expand([out_features]).to_event(1))
        # self.sigma=PyroSample(dist.HalfNormal(scale=10.0))

    def forward(self, x, y=None):
        min_value = torch.finfo(x.dtype).eps
        max_value = torch.finfo(x.dtype).max
        z=pyro.deterministic('z',value=self.linear(x))
        mean=pyro.deterministic('mean',1/z)
        mean2=torch.exp(z).clamp(min=min_value, max=max_value)
        rate=pyro.sample('rate',dist.HalfCauchy(scale=torch.tensor(10.0,device=device))).clamp(min=min_value, max=max_value)
        shape = (mean2 * rate)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Gamma(shape,rate).to_event(1), obs=y)
        return mean

class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(0.1,device=device)).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(10.,device=device)).expand([out_features]).to_event(1))
        # self.sigma=PyroSample(dist.HalfNormal(scale=10.0))

    def forward(self, x, y=None):
        mean=pyro.deterministic('mean',value=self.linear(x))
        sigma=pyro.sample('sigma',dist.HalfNormal(scale=torch.tensor(10.0,device=device)))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean,sigma).to_event(1), obs=y)
        return mean

class RegressionDropout(PyroModule):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.in_features=in_features
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(
            dist.Normal(torch.tensor(0., device=device), torch.tensor(1.0, device=device)).expand(
                [out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(
            dist.Normal(torch.tensor(0., device=device), torch.tensor(10., device=device)).expand(
                [out_features]).to_event(1))
        self.select_prob=PyroSample(dist.Beta(torch.tensor(10.0,device=device),torch.tensor(10.0,device=device)).expand([in_features]).to_event(1))


    def forward(self, x, y=None):
        selection_prior = pyro.sample('selected_vars', dist.Bernoulli(self.select_prob).to_event(1))
        # selection_prior = pyro.deterministic('selected_vars' + self.suffix, (self.select_prob>0.5)*1.0)
        new_x = x * selection_prior
        mean = pyro.deterministic('mean', value=self.linear(new_x))
        sigma = pyro.sample('sigma', dist.HalfNormal(scale=torch.tensor(10.0, device=device)))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
        return mean

def guide_selection(x,y=None):
    P=x.shape[1]
    select_prob_loc=pyro.param('select_prob_loc',torch.ones([P,],device=device)*0.2,constraint=constraints.unit_interval)
    select_prob=pyro.sample('select_prob',dist.Delta(select_prob_loc).to_event(1))
    loc_weight=pyro.param('loc_weight',torch.zeros((1,P),device=device))
    loc_bias=pyro.param('loc_bias',torch.tensor(0.0,device=device).reshape([1,]))
    scale_weight=pyro.param('scale_weight',torch.ones((1,P),device=device),constraint=constraints.positive)
    scale_bias=pyro.param('scale_bias',torch.tensor(1.0,device=device).reshape([1,]),constraint=constraints.positive)
    weight=pyro.sample('linear.weight',dist.Normal(loc_weight,scale_weight).to_event(2))
    bias=pyro.sample('linear.bias',dist.Normal(loc_bias,scale_bias).to_event(1))
    scale_sigma=pyro.param('scale_sigma',torch.tensor(5.0,device=device),constraint=constraints.positive)
    sigma=pyro.sample('sigma',dist.HalfCauchy(scale=scale_sigma))
    return {'weight':weight,'bias':bias,'select_prob':select_prob}

def model_horseshoe(x,y=None):
    P = x.shape[1]

    # sample from horseshoe prior
    lambdas = pyro.sample("lambdas", dist.HalfCauchy(torch.ones(P)))
    tau = pyro.sample("tau", dist.HalfCauchy(torch.ones(1)))


    unscaled_betas = pyro.sample("unscaled_betas", dist.Normal(0.0, torch.ones(P)))
    scaled_betas = pyro.deterministic("betas", tau * lambdas * unscaled_betas)
    bias = pyro.sample('bias', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(10., device=device)))
    mean_function = pyro.deterministic('mean', bias +x @ scaled_betas.reshape(-1,1))


    prec_obs = pyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / torch.sqrt(prec_obs)

    # observe data
    pyro.sample("obs", dist.Normal(mean_function, sigma_obs), obs=y)
    return mean_function
#******************************************************************************************************************

model_hr=model_horseshoe
# model_hr=BayesianRegression(x_train_hr.shape[1])
params=poutine.trace(model_hr).get_trace(x_train_hr).stochastic_nodes
poutine.trace(model_hr).get_trace(x_train_hr).param_nodes

hr_kernel = NUTS(model_hr )
mcmc=MCMC(hr_kernel,num_samples=5000,warmup_steps=1000,num_chains=num_chains)
mcmc.run(x_train_hr,y_train_hr.log().reshape(-1,1))
posterior_samples_hr=mcmc.get_samples().copy()
predictive_hr=Predictive(model_hr,posterior_samples_hr)
samples_hr=predictive_hr(x_test_hr,None)
joblib.dump(posterior_samples_hr,os.path.join(data_dir,f"results/weights/RegressionBayesian_hr_{experiment}.joblib"))


# guide_hr=autoguide.AutoDelta(poutine.block(model_hr,hide=['selected_vars']),)
# # guide_hr=guide_selection
# optim_hr = pyro.optim.ClippedAdam({"lr": 0.001,'weight_decay':0.0001,})
# svi_hr = SVI(model_hr, guide_hr, optim_hr, loss=Trace_ELBO(num_particles=2))
#
# pyro.clear_param_store()
# loss_=[]
# for j in range(30000):
#     # calculate the loss and take a gradient step
#     loss = svi_hr.step(x_train_hr, y_train_hr.log().reshape(-1,1))
#     loss_.append(loss/len(x_train_hr))
#     if j % 1000 == 0:
#         print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_train_hr)))
#
# plt.plot(loss_)
# plt.show()
# list(pyro.get_param_store().items())
# predictive_hr = Predictive(model_hr, guide=guide_hr, num_samples=1000,
#                         return_sites=( 'mean',),)
# samples_hr = predictive_hr(x_test_hr ,)
test_pred_hr=samples_hr['mean'].mean(dim=0).exp().reshape(-1).cpu().numpy()


print("R2: heart rate ",r2_hr:=r2_score(y_test_hr,test_pred_hr))
rmse_hr=mean_squared_error(y_test_hr,test_pred_hr,squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(test_pred_hr,y_test_hr,)
ax2.plot([50,225],[50,225],'r--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Observed")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()

# joblib.dump(hr_clf,os.path.join(data_dir,f"results/weights/Regression_hr_{experiment}.joblib"))

##*******************************************************************************************************************
#resp rate

plt.hist(resp_rate_train[~np.isnan(resp_rate_train)])
plt.show()

pw_transformer=PowerTransformer(method='box-cox')
plt.hist(pw_transformer.fit_transform(resp_rate_train[~np.isnan(resp_rate_train)].reshape(-1,1)))
plt.show()



# resp_rate_clf=regressor(resp=BayesianRegression
# model_resp=BayesianRegression(x_train_resp.shape[1])
model_resp=model_horseshoe
params=poutine.trace(model_resp).get_trace(x_train_resp).stochastic_nodes
poutine.trace(model_resp).get_trace(x_train_resp).param_nodes

resp_kernel = NUTS(model_resp )
mcmc=MCMC(resp_kernel,num_samples=5000,warmup_steps=1000,num_chains=num_chains)
mcmc.run(x_train_resp,y_train_resp.log().reshape(-1,1))
posterior_samples_resp=mcmc.get_samples().copy()
predictive_resp=Predictive(model_resp,posterior_samples_resp)
samples_resp=predictive_resp(x_test_resp,None)
joblib.dump(posterior_samples_resp,os.path.join(data_dir,f"results/weights/RegressionBayesian_resp_{experiment}.joblib"))
# guide_resp=autoguide.AutoDelta(model_resp,)
# optim_resp = pyro.optim.ClippedAdam({"lr": 0.001,'weight_decay':0.0001,})
# svi_resp = SVI(model_resp, guide_resp, optim_resp, loss=Trace_ELBO(num_particles=5))
#
# pyro.clear_param_store()
# loss_=[]
# for j in range(30000):
#     # calculate the loss and take a gradient step
#     loss = svi_resp.step(x_train_resp, y_train_resp.log().reshape(-1,1))
#     loss_.append(loss/len(x_train_resp))
#     if j % 1000 == 0:
#         print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_train_resp)))
#
# plt.plot(loss_)
# plt.show()
# list(pyro.get_param_store().items())
# predictive_resp = Predictive(model_resp, guide=guide_resp, num_samples=1000,
#                         return_sites=( 'mean',),)
# samples_resp = predictive_resp(x_test_resp ,)
test_pred_resp=samples_resp['mean'].mean(dim=0).exp().reshape(-1).cpu().numpy()


r2_rest_rate=r2_score(y_test_resp,test_pred_resp)
rmse_rest_rate=mean_squared_error(y_test_resp,test_pred_resp,squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(test_pred_resp,y_test_resp)
ax2.plot([20,100],[20,100],'r--')
ax2.set_ylabel("Observed")
ax2.set_xlabel("Predicted")
ax2.set_title("Respiratory rate")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()

# joblib.dump(resp_rate_clf,os.path.join(data_dir,f"results/weights/Regression_resp_rate_{experiment}.joblib"))
#spo2*****************************************************************************************************************
plt.hist(spo2_train[spo2_train>70])
plt.show()

pw_transformer=PowerTransformer(method='box-cox',standardize=True)
plt.hist(pw_transformer.fit_transform(spo2_train[spo2_train>70].reshape(-1,1)))
plt.show()

q_transformer=QuantileTransformer(n_quantiles=20)
plt.hist(q_transformer.fit_transform(spo2_train[spo2_train>70].reshape(-1,1)))
plt.show()


# spo2_clf=regressor()
# model_spo2=BayesianRegression(x_train_spo2.shape[1])
model_spo2=model_horseshoe
params=poutine.trace(model_spo2).get_trace(x_train_spo2).stochastic_nodes
poutine.trace(model_spo2).get_trace(x_train_spo2).param_nodes

spo2_kernel = NUTS(model_spo2 )
mcmc=MCMC(spo2_kernel,num_samples=5000,warmup_steps=1000,num_chains=num_chains)
mcmc.run(x_train_spo2,y_train_spo2.log().reshape(-1,1))
posterior_samples_spo2=mcmc.get_samples().copy()
predictive_spo2=Predictive(model_spo2,posterior_samples_spo2)
samples_spo2=predictive_spo2(x_test_spo2,None)
joblib.dump(posterior_samples_spo2,os.path.join(data_dir,f"results/weights/RegressionBayesian_spo2_{experiment}.joblib"))
# guide_spo2=autoguide.AutoDelta(model_spo2,)
# optim_spo2 = pyro.optim.ClippedAdam({"lr": 0.001,'weight_decay':0.0001,})
# svi_spo2 = SVI(model_spo2, guide_spo2, optim_spo2, loss=Trace_ELBO(num_particles=5))
#
# pyro.clear_param_store()
# loss_=[]
# for j in range(30000):
#     # calculate the loss and take a gradient step
#     loss = svi_spo2.step(x_train_spo2, y_train_spo2.log().reshape(-1,1))
#     loss_.append(loss/len(x_train_spo2))
#     if j % 1000 == 0:
#         print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_train_spo2)))
#
# plt.plot(loss_)
# plt.show()
# list(pyro.get_param_store().items())
# predictive_spo2 = Predictive(model_spo2, guide=guide_spo2, num_samples=1000,
#                         return_sites=( 'mean',),)
# samples_spo2 = predictive_spo2(x_test_spo2 ,)
test_pred_spo2=samples_spo2['mean'].mean(dim=0).exp().reshape(-1).cpu().numpy()


r2_spo2=r2_score(y_test_spo2,test_pred_spo2)
rmse_spo2=mean_squared_error(y_test_spo2,test_pred_spo2,squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(test_pred_spo2,y_test_spo2)
ax2.plot([80,100],[80,100],'r--')
ax2.set_ylabel("Observed")
ax2.set_xlabel("Predicted")
ax2.set_title("SPO2")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()

# joblib.dump(spo2_clf,os.path.join(data_dir,f"results/weights/Regression_spo2_{experiment}.joblib"))

#combined plot ******************************************************************************************************************************************

# hr_clf=joblib.load(os.path.join(data_dir,f"results/weights/Regression_hr_{experiment}.joblib"))
# test_pred_hr=hr_clf.predict(test_embedding_reduced)
# r2_hr=r2_score(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ])
# rmse_hr=mean_squared_error(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ],squared=False)
#
# resp_rate_clf=joblib.load(os.path.join(data_dir,f"results/weights/Regression_resp_rate_{experiment}.joblib"))
# test_pred_resp_rate=resp_rate_clf.predict(test_embedding_reduced)
# r2_rest_rate=r2_score(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)])
# rmse_rest_rate=mean_squared_error(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)],squared=False)
#
# spo2_clf=joblib.load(os.path.join(data_dir,f"results/weights/Regression_spo2_{experiment}.joblib"))
# test_pred_spo2=spo2_clf.predict(test_embedding_reduced)
# r2_spo2=r2_score(spo2_test[spo2_test>80],test_pred_spo2[spo2_test>80])
# rmse_spo2=mean_squared_error(spo2_test[spo2_test>80],test_pred_spo2[spo2_test>80],squared=False)


fig,axs=plt.subplots(1,3,figsize=(12,4))

axs[0].scatter(test_pred_hr,y_test_hr,)
axs[0].plot([50,225],[50,225],'r--')
axs[0].text(0.05,0.90,f"r2={r2_hr:.2f}\nrmse={rmse_hr:.1f}",transform=axs[0].transAxes,
            bbox={'boxstyle':"round",'facecolor':"wheat",'alpha':0.5})
axs[0].set_ylabel("Observed")
axs[0].set_xlabel("Predicted")
axs[0].set_title("Heart rate")

axs[1].scatter(test_pred_resp,y_test_resp,)
axs[1].plot([20,100],[20,100],'r--')
axs[1].text(0.05,0.9,f"r2={r2_rest_rate:.2f}\nrmse={rmse_rest_rate:.1f}",transform=axs[1].transAxes,
            bbox={'boxstyle':"round",'facecolor':"wheat",'alpha':0.5})
axs[1].set_ylabel("Observed")
axs[1].set_xlabel("Predicted")
axs[1].set_title("Respiratory rate")

axs[2].scatter(test_pred_spo2,y_test_spo2,)
axs[2].plot([80,100],[80,100],'r--')
axs[2].text(0.05,0.9,f"r2={r2_spo2:.2f}\nrmse={rmse_spo2:.1f}",transform=axs[2].transAxes,
            bbox={'boxstyle':"round",'facecolor':"wheat",'alpha':0.5})
axs[2].set_ylabel("Observed")
axs[2].set_xlabel("Predicted")
axs[2].set_title("SPO2")
plt.savefig(os.path.join(output_dir,f"Regression plots Bayesian - {experiment}.png"))
plt.show()


# save_regression(model=experiment + "-Bayesian",rmse=rmse_hr,r2=r2_hr,details='heart rate',
#                 other=json.dumps({k:v for k,v in hr_clf.best_params_.items() if k != "select__score_func"}))
#
# save_regression(model=experiment + "-Bayesian",rmse=rmse_rest_rate,r2=r2_rest_rate,details='respiratory rate',
#                 other=json.dumps({k:v for k,v in resp_rate_clf.best_params_.items() if k != "select__score_func"}))
#
# save_regression(model=experiment + "-Bayesian",rmse=rmse_spo2,r2=r2_spo2,details='SpO2',
#                 other=json.dumps({k:v for k,v in spo2_clf.best_params_.items() if k != "select__score_func"}))
