from functools import partial

import numpy as np

import joblib
import pandas as pd
from pyro.contrib.easyguide import easy_guide, EasyGuide
from pyro.distributions import constraints
import pyro.distributions.transforms as T
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoGuide, init_to_sample, init_to_uniform
from pyro.nn import PyroModule, PyroSample, PyroParam, DenseNN
from sklearn.metrics import roc_auc_score,confusion_matrix,plot_confusion_matrix
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,PolynomialFeatures



import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro import poutine

from settings import data_dir,output_dir,clinical_predictors
import os
import matplotlib.pyplot as plt

from utils import admission_confusion_matrix, admission_distplot
pyro.set_rng_seed(123)
# device='cuda' if torch.cuda.is_available() else "cpu"
device='cpu'

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

preprocess=Pipeline([
    ('scl',StandardScaler()),
    # ('poly',PolynomialFeatures(degree=2,include_bias=False,interaction_only=True))
])
x_data_clinical=torch.tensor(preprocess.fit_transform(train_x),dtype=torch.float)
x_data_test_clinical=torch.tensor(preprocess.transform(test_x),dtype=torch.float)
x_data_ppg=torch.tensor(preprocess.fit_transform(classifier_embedding_reduced),dtype=torch.float)
x_data_test_ppg=torch.tensor(preprocess.transform(test_embedding_reduced),dtype=torch.float)

x_data_concat=torch.concat([x_data_clinical,x_data_ppg],dim=1)
x_data_test_concat=torch.concat([x_data_test_clinical,x_data_test_ppg],dim=1)
# x_data=scl.fit_transform(np.concatenate([classifier_embedding_reduced,train_x],axis=1))
# x_data_test=scl.transform(np.concatenate([test_embedding_reduced,test_x],axis=1))
# poly=PolynomialFeatures(degree=1,include_bias=False)
# x_data=poly.fit_transform(x_data)
# x_data_test=poly.transform(x_data_test)

y_data=torch.Tensor(admitted_train).to(device)
x_data_clinical,x_data_ppg,x_data_concat=x_data_clinical.to(device),x_data_ppg.to(device),x_data_concat.to(device)
x_data_test_clinical,x_data_test_ppg,x_data_test_concat=x_data_test_clinical.to(device),x_data_test_ppg.to(device),x_data_test_concat.to(device)




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

class SpikeSlab(PyroModule):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.in_features=in_features
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.0).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))
        # self.prior_selected=PyroParam(torch.ones([in_features])*0.5,constraint=constraints.unit_interval)
        # self.selected_vars=pyro.sample('selected',dist.Bernoulli(self.prior_selected).to_event(1))
        self.select_prob=PyroSample(dist.Beta(1.0,1.0).expand([in_features]).to_event(1))


    def forward(self, x, y=None):
        selection_prior = pyro.sample('selected_vars', dist.Bernoulli(self.select_prob).to_event(1))
        # gamma = pyro.sample('gamma', dist.Bernoulli(selection_prior).to_event(1))
        new_x = x * selection_prior
        mean=pyro.deterministic('mean',value=self.linear(new_x).sigmoid())
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
        return mean



class SpikeSlabMLP(PyroModule):
    def __init__(self, in_features,dim_layers=[16,8,4,1]):
        super().__init__()
        module_list=[]
        for i,l in enumerate(dim_layers):
            if i==0:
                p=PyroModule[nn.Linear](in_features, l)
                p.weight = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(1.0,device=device)).expand([l, in_features]).to_event(2))
                p.bias = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(10.,device=device)).expand([l]).to_event(1))
                module_list.append(p)
                module_list.append(nn.LeakyReLU(0.1).to(device))
            else:
                p = PyroModule[nn.Linear](dim_layers[i-1], l)
                p.weight = PyroSample(
                    dist.Normal(torch.tensor(0., device=device), torch.tensor(1.0, device=device)).expand(
                        [l, dim_layers[i-1]]).to_event(2))
                p.bias = PyroSample(
                    dist.Normal(torch.tensor(0., device=device), torch.tensor(10., device=device)).expand(
                        [l]).to_event(1))
                module_list.append(p)
                if i==(len(dim_layers)-1): module_list.append(nn.LeakyReLU(0.1).to(device))

        self.net=PyroModule[nn.Sequential](*module_list)

        # self.linear1 = PyroModule[nn.Linear](in_features, dim_hidden)
        # self.linear2 = PyroModule[nn.Linear](dim_hidden, out_features)
        # self.linear1.weight = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(1.,device=device)).expand([dim_hidden, in_features]).to_event(2))
        # self.linear1.bias = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(10.,device=device)).expand([dim_hidden]).to_event(1))
        # self.linear2.weight = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(1.,device=device)).expand([out_features, dim_hidden]).to_event(2))
        # self.linear2.bias = PyroSample(dist.Normal(torch.tensor(0.,device=device), torch.tensor(10.,device=device)).expand([out_features]).to_event(1))
        # self.activation=nn.LeakyReLU(0.1).to(device)
        self.select_prob = PyroSample(dist.Beta(torch.tensor(10.0,device=device), torch.tensor(10.0,device=device)).expand([in_features]).to_event(1))

    def forward(self, x, y=None):
        selection_prior = pyro.sample('selected_vars', dist.Bernoulli(self.select_prob).to_event(1))
        # gamma = pyro.sample('gamma', dist.Bernoulli(selection_prior).to_event(1))
        new_x = x * selection_prior
        # out=self.activation(self.linear1(new_x))
        mean=pyro.deterministic('mean',value=self.net(new_x).sigmoid())
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
        return mean

############
model_ppg=SpikeSlabMLP(in_features=x_data_ppg.shape[1],)
poutine.trace(model_ppg).get_trace(x_data_ppg).stochastic_nodes
poutine.trace(model_ppg).get_trace(x_data_ppg).param_nodes
# guide_ppg=pyro.infer.autoguide.AutoGuideList(model_ppg)
# guide_ppg.append(pyro.infer.autoguide.AutoMultivariateNormal(pyro.poutine.block(model_ppg,hide=['select_prob','selected_vars','obs'])))
# guide_ppg.append(pyro.infer.autoguide.AutoDelta(pyro.poutine.block(model_ppg,expose=['select_prob',])))
# guide_ppg.append(pyro.infer.autoguide.AutoCallable(pyro.poutine.block(model_ppg,expose=['selected_vars',]),guide_spike_slab))
# def null_guide(obs_x,obs_y):
#     pass
transforms=partial(T.iterated, 2, T.block_autoregressive)
guide_ppg=pyro.infer.autoguide.AutoGuideList(model_ppg)
# guide_concat.append(pyro.infer.autoguide.AutoMultivariateNormal(poutine.block(model_concat,hide=['select_prob','selected_vars','obs']),init_loc_fn=init_to_uniform(radius=1.0),init_scale=0.01))
guide_ppg.append(pyro.infer.autoguide.AutoNormalizingFlow(poutine.block(model_ppg,hide=['select_prob','selected_vars','obs']),init_transform_fn=transforms))
guide_ppg.append(pyro.infer.autoguide.AutoDelta(pyro.poutine.block(model_ppg,expose=['select_prob',])))
# guide_ppg.to(device)

adam_ppg = pyro.optim.Adam({"lr": 0.001,'betas':(0.9,0.999)})
svi_ppg = SVI(model_ppg, guide_ppg, adam_ppg, loss=Trace_ELBO(num_particles=5))

pyro.clear_param_store()
loss_=[]
for j in range(30000):
    # calculate the loss and take a gradient step
    loss = svi_ppg.step(x_data_ppg, y_data.reshape(-1,1))
    loss_.append(loss/len(x_data_ppg))
    if j % 1000 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data_ppg)))

plt.plot(loss_)
plt.show()

# for k,v in pyro.get_param_store().items():
#     if "selection_prior" in k:
#         selection_param_name=k
#         selected_param_value=pyro.get_param_store().get_param(k)
#         selected=(selected_param_value>0.5)*1.0

predictive_ppg = Predictive(model_ppg, guide=guide_ppg, num_samples=1000,
                        return_sites=( 'mean','gamma'),)
samples_ppg = predictive_ppg(x_data_test_ppg ,)
test_pred_ppg=samples_ppg['mean'].mean(dim=0).reshape(-1).numpy()
roc_auc_score(admitted_test,test_pred_ppg)


base_rate=admitted_train.mean()
pred_positive_ppg=(test_pred_ppg>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_ppg)

displot_ppg=admission_distplot(samples_ppg['mean'].squeeze(),admitted_test,pred_positive_ppg)


# def guide_spike_slab(x, y=None):
#     P = x.shape[1]
#     prior_p = pyro.param('prior_p', torch.tensor([0.5]).expand([P]), constraint=constraints.unit_interval)
#     # gamma_prior_beta = pyro.param('gamma_prior_beta', torch.ones(P), constraint=constraints.positive)
#     selected_probs = pyro.sample('select_prob', dist.Delta(prior_p).to_event(1))
#     return {'select_prob': selected_probs}

# def guide_spike_slab(x, y=None):
#     P = x.shape[1]
#     alpha=pyro.param('alpha',lambda : torch.tensor([1.0]).expand([P]),constraint=constraints.positive)
#     beta = pyro.param('beta', lambda : torch.tensor([1.0]).expand([P]), constraint=constraints.positive)
#
#     prior_p = pyro.sample('prior_p', dist.Beta(alpha,beta).to_event(1))
#     # gamma_prior_beta = pyro.param('gamma_prior_beta', torch.ones(P), constraint=constraints.positive)
#     selection_prior = pyro.sample('selected_vars', dist.Bernoulli(prior_p).to_event(1))
#     return {'selected_vars': selection_prior}
# model_clinical=ScaleModel(in_features=x_data_clinical.shape[1],)
model_clinical=SpikeSlab(in_features=x_data_clinical.shape[1],)
poutine.trace(model_clinical).get_trace(x_data_clinical).stochastic_nodes
poutine.trace(model_clinical).get_trace(x_data_clinical).param_nodes

# model_clinical=SpikeSlabDispersed(in_features=x_data_clinical.shape[1],)
# guide_clinical=pyro.infer.autoguide.AutoDelta(poutine.block(model_clinical,hide=['select_prob','selected_vars','obs']))
# @easy_guide(model_clinical)
# def guide_clinical(self,x,y=None):
#     self.map_estimate('select_prob')
#     group_weights = self.group(match="linear.weight")
#     group_weights_loc=pyro.param('group_weights_loc',torch.randn(group_weights.event_shape)*0.1)
#     group_weights_scale = pyro.param('group_weights_scale', torch.full(group_weights.event_shape,0.1),constraint=constraints.positive)
#     weights=group_weights.sample('weights',dist.Normal(loc=group_weights_loc,scale=group_weights_scale).to_event(1))
#     group_bias = self.group(match="linear.bias")
#     group_bias_loc = pyro.param('group_bias_loc', torch.randn(group_bias.event_shape) * 0.1)
#     group_bias_scale = pyro.param('group_bias_scale', torch.full(group_bias.event_shape, 0.1),
#                                      constraint=constraints.positive)
#     bias=group_bias.sample('bias', dist.Normal(loc=group_bias_loc, scale=group_bias_scale).to_event(1))
#     return {'weights':weights,'bias':bias}






# # linear_guide=pyro.infer.autoguide.AutoDiagonalNormal(linear_model)
# guide_clinical=pyro.infer.autoguide.AutoGaussian(pyro.poutine.block(model_clinical,hide=['obs','mean']))
guide_clinical=pyro.infer.autoguide.AutoGuideList(model_clinical)
guide_clinical.append(pyro.infer.autoguide.AutoMultivariateNormal(pyro.poutine.block(model_clinical,hide=['select_prob','selected_vars','obs'])))
# # guide_clinical.append(pyro.infer.autoguide.AutoCallable(pyro.poutine.block(model_clinical,expose=['select_prob',]),guide_spike_slab))
guide_clinical.append(pyro.infer.autoguide.AutoDelta(pyro.poutine.block(model_clinical,expose=['select_prob',])))
# def null_guide(obs_x,obs_y):
#     pass

adam_clinical = pyro.optim.Adam({"lr": 0.001})
svi_clinical = SVI(model_clinical, guide_clinical, adam_clinical, loss=Trace_ELBO())

pyro.clear_param_store()
loss_=[]
for j in range(30000):
    # calculate the loss and take a gradient step
    loss = svi_clinical.step(x_data_clinical, y_data.reshape(-1,1))
    loss_.append(loss/len(x_data_clinical))
    if j % 1000 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data_clinical)))

plt.plot(loss_)
plt.show()


# for k,v in pyro.get_param_store().items():
#     if "selection_prior" in k:
#         selection_param_name=k
#         selected_param_value=pyro.get_param_store().get_param(k)
#         selected=(selected_param_value>0.5)*1.0
list(pyro.get_param_store().items())
# selected=(pyro.get_param_store()['AutoGuideList.1.select_prob']>0.5)*1.0
#
# predictive_clinical = Predictive(poutine.condition(model_clinical,data={'selected_vars':selected}), guide=guide_clinical, num_samples=1000,
#                         return_sites=( 'mean','gamma'))
predictive_clinical = Predictive(model_clinical, guide=guide_clinical, num_samples=1000,
                        return_sites=( 'mean','gamma'))
samples_clinical = predictive_clinical(x_data_test_clinical ,)
test_pred_clinical=samples_clinical['mean'].mean(dim=0).reshape(-1).numpy()
roc_auc_score(admitted_test,test_pred_clinical)


base_rate=admitted_train.mean()
pred_positive_clinical=(test_pred_clinical>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_clinical)

admission_distplot(samples_clinical['mean'].squeeze(),admitted_test,pred_positive_clinical)

###############################################################################################################


###########################################################################################################

samples_both=np.concatenate([samples_clinical['mean'],samples_ppg['mean']],axis=0)
test_pred_both=samples_both.mean(axis=0).reshape(-1)
roc_auc_score(admitted_test,test_pred_both)


base_rate=admitted_train.mean()
pred_positive_both=(test_pred_both>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_both)

distplot_both=admission_distplot(samples_both.squeeze(),admitted_test,pred_positive_both)

#######################################################################################################################
#Concatenated
######################################################################################################################
model_concat=SpikeSlabMLP(in_features=x_data_concat.shape[1],)

# nuts_kernel=NUTS(pyro.poutine.block(model_concat,hide=['logits','mean']),)
# mcmc=MCMC(nuts_kernel,num_samples=50,warmup_steps=10,)
# mcmc.run(x_data_concat,y_data.reshape(-1,1))
# posterior_samples=mcmc.get_samples()
# predictive=Predictive(model_concat,posterior_samples)
# samples_concat=predictive(x_data_test_concat,None)
# guide_concat=pyro.infer.autoguide.AutoGaussian(pyro.poutine.block(model_concat,hide=['obs','mean']))

# transforms=T.ComposeTransform([T.spline,T.spline,T.spline])
# input_dim = 50+1
# split_dim = 40
# param_dims = [input_dim-split_dim, input_dim-split_dim]
# hypernet = DenseNN(split_dim, [10*input_dim], param_dims)
# transforms = T.AffineCoupling(split_dim, hypernet)
# flow_dist = dist.TransformedDistribution(dist.Normal(torch.zeros(input_dim),torch.ones(input_dim)), [transforms])
# flow_dist.sample()  # doctest: +SKIP
transforms=partial(T.iterated, 2, T.block_autoregressive)
guide_concat=pyro.infer.autoguide.AutoGuideList(model_concat)
# guide_concat.append(pyro.infer.autoguide.AutoMultivariateNormal(poutine.block(model_concat,hide=['select_prob','selected_vars','obs']),init_loc_fn=init_to_uniform(radius=1.0),init_scale=0.01))
guide_concat.append(pyro.infer.autoguide.AutoNormalizingFlow(poutine.block(model_concat,hide=['select_prob','selected_vars','obs']),init_transform_fn=transforms))
guide_concat.append(pyro.infer.autoguide.AutoDelta(pyro.poutine.block(model_concat,expose=['select_prob',])))
guide_concat.to('cuda')
# guide_concat.append(pyro.infer.autoguide.AutoCallable(pyro.poutine.block(model_concat,expose=['selected_vars',]),guide_spike_slab))
# def null_guide(obs_x,obs_y):
#     pass

num_steps = 15000
initial_lr = 0.001
gamma = 0.01  # final learning rate will be gamma * initial_lr
lrd = gamma ** (1 / num_steps)
adam_concat = pyro.optim.ClippedAdam({"lr": initial_lr,'lrd':lrd})
svi_concat = SVI(model_concat, guide_concat, adam_concat, loss=Trace_ELBO(num_particles=1))

pyro.clear_param_store()
loss_=[]
for j in range(num_steps):
    # calculate the loss and take a gradient step
    loss = svi_concat.step(x_data_concat, y_data.reshape(-1,1))
    loss_.append(loss/len(x_data_concat))
    if j % 1000 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data_concat)))

plt.plot(loss_)
plt.show()


selected_variables=pyro.get_param_store()['AutoGuideList.1.select_prob']

predictive_concat = Predictive(model_concat, guide=guide_concat, num_samples=10,
                        return_sites=( 'mean','gamma'),)
samples_concat = predictive_concat(x_data_test_concat ,)
test_pred_concat=samples_concat['mean'].mean(dim=0).reshape(-1).numpy()
roc_auc_score(admitted_test,test_pred_concat)

base_rate=admitted_train.mean()
pred_positive_concat=(test_pred_concat>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_concat)

displot_concat=admission_distplot(samples_concat['mean'].squeeze(),admitted_test,pred_positive_concat)