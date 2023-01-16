import matplotlib.pyplot as plt
import os


import joblib
import numpy as np
import pandas as pd
import pyro
import torch

from pyro.infer import  Predictive, NUTS, MCMC

import pyro.distributions as dist
from pyro import poutine

from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,PolynomialFeatures,PowerTransformer
from sklearn.metrics import r2_score,mean_squared_error

from sklearn.pipeline import Pipeline


from settings import data_dir,weights_dir,output_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pyro.set_rng_seed(0)
device='cpu'
num_chains=1
polynomial_degree=2
# experiment="Contrastive-original-sample-DotProduct32-sepsis"
experiment="Contrastive-original-sample-DotProduct32"
# experiment='PCA-32'
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")

classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)

train_ids=train['id'].unique()
test_ids=test['id'].unique()
classifier_embedding_reduced=np.stack(map(lambda id:classifier_embedding[train['id']==id,:].mean(axis=0) ,train_ids))
test_embedding_reduced=np.stack(map(lambda id:test_embedding[test['id']==id,:].mean(axis=0) ,test_ids))

#
hr_train=np.stack(map(lambda id:train.loc[train['id']==id,'hr'].median(skipna=True),train_ids))
hr_test=np.stack(map(lambda id:test.loc[test['id']==id,'hr'].median(skipna=True),test_ids))
x_train_hr,y_train_hr=classifier_embedding_reduced[hr_train!=0,:],hr_train[hr_train!=0]
x_test_hr,y_test_hr=test_embedding_reduced[hr_test!=0,:],hr_test[hr_test!=0]
preprocess_hr=Pipeline([
    ('poly',PolynomialFeatures(degree=polynomial_degree,include_bias=False)),
    ('scl',StandardScaler()),
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
])
preprocess_spo2.fit(x_train_spo2)
x_train_spo2=torch.tensor(preprocess_spo2.transform(x_train_spo2),dtype=torch.float,device=device)
x_test_spo2=torch.tensor(preprocess_spo2.transform(x_test_spo2),dtype=torch.float,device=device)
y_train_spo2=torch.tensor(y_train_spo2,device=device,dtype=torch.float)



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
params=poutine.trace(model_hr).get_trace(x_train_hr).stochastic_nodes
poutine.trace(model_hr).get_trace(x_train_hr).param_nodes

hr_kernel = NUTS(model_hr )
mcmc=MCMC(hr_kernel,num_samples=5000,warmup_steps=1000,num_chains=num_chains)
mcmc.run(x_train_hr,y_train_hr.log().reshape(-1,1))
posterior_samples_hr=mcmc.get_samples().copy()
predictive_hr=Predictive(model_hr,posterior_samples_hr)
samples_hr=predictive_hr(x_test_hr,None)
joblib.dump(posterior_samples_hr,os.path.join(data_dir,f"results/weights/RegressionBayesian_hr_{experiment}.joblib"))

test_pred_hr=samples_hr['mean'].mean(dim=0).exp().reshape(-1).cpu().numpy()


print("R2: heart rate ",r2_hr:=r2_score(y_test_hr,test_pred_hr))
rmse_hr=mean_squared_error(y_test_hr,test_pred_hr,squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(test_pred_hr,y_test_hr,)
ax2.plot([50,225],[50,225],'r--')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Observed")
plt.show()


##*******************************************************************************************************************
#resp rate

plt.hist(resp_rate_train[~np.isnan(resp_rate_train)])
plt.show()

pw_transformer=PowerTransformer(method='box-cox')
plt.hist(pw_transformer.fit_transform(resp_rate_train[~np.isnan(resp_rate_train)].reshape(-1,1)))
plt.show()




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

test_pred_resp=samples_resp['mean'].mean(dim=0).exp().reshape(-1).cpu().numpy()


r2_rest_rate=r2_score(y_test_resp,test_pred_resp)
rmse_rest_rate=mean_squared_error(y_test_resp,test_pred_resp,squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(test_pred_resp,y_test_resp)
ax2.plot([20,100],[20,100],'r--')
ax2.set_ylabel("Observed")
ax2.set_xlabel("Predicted")
ax2.set_title("Respiratory rate")
plt.show()

#spo2*****************************************************************************************************************
plt.hist(spo2_train[spo2_train>70])
plt.show()

pw_transformer=PowerTransformer(method='box-cox',standardize=True)
plt.hist(pw_transformer.fit_transform(spo2_train[spo2_train>70].reshape(-1,1)))
plt.show()

q_transformer=QuantileTransformer(n_quantiles=20)
plt.hist(q_transformer.fit_transform(spo2_train[spo2_train>70].reshape(-1,1)))
plt.show()



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

test_pred_spo2=samples_spo2['mean'].mean(dim=0).exp().reshape(-1).cpu().numpy()


r2_spo2=r2_score(y_test_spo2,test_pred_spo2)
rmse_spo2=mean_squared_error(y_test_spo2,test_pred_spo2,squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(test_pred_spo2,y_test_spo2)
ax2.plot([80,100],[80,100],'r--')
ax2.set_ylabel("Observed")
ax2.set_xlabel("Predicted")
ax2.set_title("SPO2")
plt.show()


#combined plot ******************************************************************************************************************************************



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
# axs[1].set_ylabel("Observed")
axs[1].set_xlabel("Predicted")
axs[1].set_title("Respiratory rate")

axs[2].scatter(test_pred_spo2,y_test_spo2,)
axs[2].plot([80,100],[80,100],'r--')
axs[2].text(0.05,0.9,f"r2={r2_spo2:.2f}\nrmse={rmse_spo2:.1f}",transform=axs[2].transAxes,
            bbox={'boxstyle':"round",'facecolor':"wheat",'alpha':0.5})
# axs[2].set_ylabel("Observed")
axs[2].set_xlabel("Predicted")
axs[2].set_title("SPO2")
plt.savefig(os.path.join(output_dir,f"Regression plots Bayesian - {experiment}.png"))
plt.show()

