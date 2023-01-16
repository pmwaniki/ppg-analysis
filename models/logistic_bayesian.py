import json
from functools import partial

import numpy as np

import joblib
import pandas as pd
from pyro.contrib.easyguide import easy_guide, EasyGuide
from pyro.distributions import constraints
import pyro.distributions.transforms as T
from pyro.distributions.transforms import block_autoregressive
from pyro.infer import SVI, Trace_ELBO, Predictive, NUTS, MCMC, autoguide, config_enumerate, TraceEnum_ELBO, HMC
from pyro.infer.autoguide import AutoGuide, init_to_sample, init_to_uniform, init_to_feasible
from pyro.infer.reparam import NeuTraReparam
from pyro.nn import PyroModule, PyroSample, PyroParam, DenseNN
from pyro.ops.indexing import Vindex
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pyro
import pyro.distributions as dist
from pyro import poutine

from settings import data_dir, output_dir, clinical_predictors
import os
import matplotlib.pyplot as plt
import seaborn as sns

from utils import admission_confusion_matrix, admission_distplot, save_table3

pyro.set_rng_seed(123)
# device='cuda' if torch.cuda.is_available() else "cpu"
device = 'cpu'
polynomial_degree = 2
mcmc_samples = 5000
mcmc_burnin = 1000
max_tree_depth=10
full_mass=False


# experiment="Contrastive-original-sample-DotProduct32"
experiment = "Contrastive-original-sample-DotProduct32-sepsis"
experiment_file = os.path.join(data_dir, f"results/{experiment}.joblib")
classifier_embedding, test_embedding, seg_train, seg_test = joblib.load(experiment_file)
train_ids = seg_train['id'].unique()
test_ids = seg_test['id'].unique()

classifier_embedding_reduced = np.stack(
    map(lambda id: classifier_embedding[seg_train['id'] == id, :].mean(axis=0), train_ids))
test_embedding_reduced = np.stack(map(lambda id: test_embedding[seg_test['id'] == id, :].mean(axis=0), test_ids))
admitted_train = np.stack(map(lambda id: seg_train.loc[seg_train['id'] == id, 'admitted'].iat[0], train_ids))
admitted_test = np.stack(map(lambda id: seg_test.loc[seg_test['id'] == id, 'admitted'].iat[0], test_ids))
died_test = np.stack(map(lambda id: seg_test.loc[seg_test['id'] == id, 'died'].iat[0], test_ids))

data = pd.read_csv(os.path.join(data_dir, "triage/data.csv"))
# feature engineering
data['spo_transformed'] = 4.314 * np.log10(103.711 - data['Oxygen saturation']) - 37.315

train = data.iloc[[np.where(data['Study No'] == id)[0][0] for id in train_ids], :].copy()
test = data.iloc[[np.where(data['Study No'] == id)[0][0] for id in test_ids], :].copy()

# Imputation
clinical_predictors = ['Weight (kgs)', 'Irritable/restlessness ', 'MUAC (Mid-upper arm circumference) cm',
                       'Can drink / breastfeed?', 'spo_transformed', 'Temperature (degrees celsius)',
                       'Difficulty breathing', 'Heart rate(HR) ']

# predictors_oximeter=['Heart rate(HR) ','spo_transformed',]

for p in clinical_predictors:
    if data[p].dtype == np.float:
        median = train[p].median()
        train[p] = train[p].fillna(median)
        test[p] = test[p].fillna(median)
    else:
        categories = pd.unique(train[p].dropna())
        majority = train[p].value_counts().index[0]
        train[p] = train[p].fillna(majority)
        test[p] = test[p].fillna(majority)
        train[p] = pd.Categorical(train[p], categories=categories)
        test[p] = pd.Categorical(test[p], categories=categories)

# train['Can drink / breastfeed?']=train['Can drink / breastfeed?'].map({"Yes":"No","No":"Yes"})

train_x = pd.get_dummies(train[clinical_predictors], drop_first=True)
test_x = pd.get_dummies(test[clinical_predictors], drop_first=True)

train_oximeter = train_x[['Heart rate(HR) ', 'spo_transformed']].copy()
test_oximeter = test_x[['Heart rate(HR) ', 'spo_transformed']].copy()

preprocess_default = Pipeline([
    ('poly', PolynomialFeatures(degree=polynomial_degree, include_bias=False, interaction_only=True)),
    ('scl', StandardScaler()),

])

preprocess_clinical = Pipeline([
    ('poly', PolynomialFeatures(degree=polynomial_degree, include_bias=False, interaction_only=True)),
    ('scl', StandardScaler()),
    # ('pca',PCA(n_components=8)),

])
preprocess_oximeter = Pipeline([
    ('poly', PolynomialFeatures(degree=polynomial_degree, include_bias=False, interaction_only=False)),
    ('scl', StandardScaler()),
    # ('pca',PCA(n_components=8)),

])
preprocess_ppg = Pipeline([
    ('poly', PolynomialFeatures(degree=polynomial_degree, include_bias=False, interaction_only=False)),
    ('scl', StandardScaler()),
    # ('pca',PCA(n_components=16)),

])
preprocess_concat = Pipeline([
    ('poly', PolynomialFeatures(degree=polynomial_degree, include_bias=False, interaction_only=False)),
    ('scl', StandardScaler()),

])
x_data_clinical = torch.tensor(preprocess_clinical.fit_transform(train_x), dtype=torch.float)
x_data_test_clinical = torch.tensor(preprocess_clinical.transform(test_x), dtype=torch.float)

x_data_oximeter = torch.tensor(preprocess_oximeter.fit_transform(train_oximeter), dtype=torch.float)
x_data_test_oximeter = torch.tensor(preprocess_oximeter.transform(test_oximeter), dtype=torch.float)

x_data_ppg = torch.tensor(preprocess_ppg.fit_transform(classifier_embedding_reduced), dtype=torch.float)
x_data_test_ppg = torch.tensor(preprocess_ppg.transform(test_embedding_reduced), dtype=torch.float)

x_data_concat_clinical = torch.tensor(
    preprocess_default.fit_transform(np.concatenate([train_x.values, classifier_embedding_reduced], axis=1)),
    dtype=torch.float, device=device)
x_data_test_concat_clinical = torch.tensor(
    preprocess_default.transform(np.concatenate([test_x.values, test_embedding_reduced], axis=1)), dtype=torch.float,
    device=device)

x_data_concat_oximeter = torch.tensor(
    preprocess_default.fit_transform(np.concatenate([train_oximeter.values, classifier_embedding_reduced], axis=1)),
    dtype=torch.float, device=device)
x_data_test_concat_oximeter = torch.tensor(
    preprocess_default.transform(np.concatenate([test_oximeter.values, test_embedding_reduced], axis=1)),
    dtype=torch.float, device=device)

# x_data=scl.fit_transform(np.concatenate([classifier_embedding_reduced,train_x],axis=1))
# x_data_test=scl.transform(np.concatenate([test_embedding_reduced,test_x],axis=1))
# poly=PolynomialFeatures(degree=1,include_bias=False)
# x_data=poly.fit_transform(x_data)
# x_data_test=poly.transform(x_data_test)

y_data = torch.Tensor(admitted_train).to(device)
x_data_clinical, x_data_ppg = x_data_clinical.to(device), x_data_ppg.to(device)
x_data_test_clinical, x_data_test_ppg = x_data_test_clinical.to(device), x_data_test_ppg.to(device)
x_data_oximeter, x_data_test_oximeter = x_data_oximeter.to(device), x_data_test_oximeter.to(device)
x_data_concat = torch.concat((x_data_clinical, x_data_ppg), dim=1)
x_data_test_concat = torch.concat((x_data_test_clinical, x_data_test_ppg), dim=1)


###########################
# MODEL DEFINITION
#################################

def logistic(x, y=None):
    P = x.shape[1]
    beta_plate = pyro.plate("beta_plate", P)
    bias = pyro.sample('bias', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(10., device=device)).expand([1]).to_event(1))
    # betas=[]
    loc_beta = pyro.param("loc_beta", torch.zeros(P))
    scale_beta = pyro.param("scale_beta", torch.ones(P), constraint=constraints.positive)

    with beta_plate:
        beta = pyro.sample(f'beta', dist.Normal(loc_beta, scale_beta))
    logits = bias + x @ beta
    mean = pyro.deterministic('mean', value=logits.reshape(-1, 1).sigmoid())
    with pyro.plate('data', x.shape[0]):
        pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
    return mean


def model_horseshoe(x, y=None):
    P = x.shape[1]
    lambdas = pyro.sample("lambdas", dist.HalfCauchy(torch.ones(P)).to_event(1))
    tau = pyro.sample("tau", dist.HalfCauchy(torch.ones(1)))

    bias = pyro.sample('bias', dist.Normal(torch.tensor(0., device=device),
                                           torch.tensor(10., device=device)))
    unscaled_betas = pyro.sample("unscaled_betas", dist.Normal(0.0, torch.ones(P)).to_event(1))
    beta = pyro.deterministic("betas", tau * lambdas * unscaled_betas)
    # print("beta shape: ", beta.shape)
    logits = bias + x @ beta.reshape(-1, 1)
    mean = pyro.deterministic('mean', value=logits.sigmoid())
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Bernoulli(mean).to_event(1), obs=y)
    return mean


#######################################################################################################################
# PPG
##########################################################################################################################
# model_ppg=logistic
model_ppg = model_horseshoe

poutine.trace(model_ppg).get_trace(x_data_ppg).stochastic_nodes
poutine.trace(model_ppg).get_trace(x_data_ppg).param_nodes

pyro.set_rng_seed(1)
ppg_kernel = NUTS(model_ppg, full_mass=full_mass,max_tree_depth=max_tree_depth)
mcmc = MCMC(ppg_kernel, num_samples=mcmc_samples, warmup_steps=mcmc_burnin, num_chains=1)
mcmc.run(x_data_ppg, y_data.reshape(-1, 1))
posterior_samples_ppg = mcmc.get_samples()
predictive = Predictive(model_ppg, posterior_samples_ppg)
samples_ppg = predictive(x_data_test_ppg, None)
joblib.dump(posterior_samples_ppg, os.path.join(data_dir, f"results/weights/LogisticBayesian_ppg_{experiment}.joblib"))
# posterior_samples_ppg=joblib.load(os.path.join(data_dir, f"results/weights/LogisticBayesian_ppg_{experiment}.joblib"))


test_pred_ppg = samples_ppg['mean'].mean(dim=0).reshape(-1).cpu().numpy()
print(">>AUC PPG ", roc_auc_score(admitted_test, test_pred_ppg))

base_rate = admitted_train.mean()
pred_positive_ppg = (test_pred_ppg > base_rate) * 1.0
report = classification_report(admitted_test, pred_positive_ppg, output_dict=True)
recall_ppg = report['1.0']['recall']
precision_ppg = report['1.0']['precision']
f1_ppg = report['1.0']['f1-score']
specificity_ppg = report['0.0']['recall']
acc_ppg = report['accuracy']
auc_ppg = roc_auc_score(admitted_test, test_pred_ppg)
save_table3(model="Contrastive-Bayesian_PPG", precision=precision_ppg, recall=recall_ppg, specificity=specificity_ppg,
            auc=auc_ppg, details=f"{experiment}", other=json.dumps({'host': os.uname()[1], 'f1': f1_ppg,
                                                                    'acc': acc_ppg}))

confusion_ppg = admission_confusion_matrix(admitted_test, pred_positive_ppg)
confusion_ppg.savefig(os.path.join(output_dir, f"Confusion matrix ppg - {experiment}.png"))

displot_ppg = admission_distplot(samples_ppg['mean'].cpu().squeeze(), admitted_test, pred_positive_ppg)
displot_ppg.savefig(os.path.join(output_dir, f'Density plots - ppg {experiment}.png'))

#######################################################################################################################
# OXYMETER
######################################################################################################################
model_oximeter = logistic
# model_oximeter=model_horseshoe
pyro.clear_param_store()
poutine.trace(model_oximeter).get_trace(x_data_oximeter).stochastic_nodes
poutine.trace(model_oximeter).get_trace(x_data_oximeter).param_nodes

pyro.set_rng_seed(2)
oximeter_kernel = NUTS(model_oximeter, full_mass=full_mass,max_tree_depth=max_tree_depth)
mcmc = MCMC(oximeter_kernel, num_samples=mcmc_samples, warmup_steps=mcmc_burnin, num_chains=1)
mcmc.run(x_data_oximeter, y_data.reshape(-1, 1))
posterior_samples_oximeter = mcmc.get_samples()
predictive_oximeter = Predictive(model_oximeter, posterior_samples_oximeter)
samples_oximeter = predictive_oximeter(x_data_test_oximeter, None)
joblib.dump(posterior_samples_oximeter,
            os.path.join(data_dir, f"results/weights/LogisticBayesian_oximeter_{experiment}.joblib"))

test_pred_oximeter = samples_oximeter['mean'].squeeze().mean(dim=0).reshape(-1).cpu().numpy()
print("Auc oximeter: ",roc_auc_score(admitted_test, test_pred_oximeter))

base_rate = admitted_train.mean()
pred_positive_oximeter = (test_pred_oximeter > base_rate) * 1.0

report = classification_report(admitted_test, pred_positive_oximeter, output_dict=True)
recall_oximeter = report['1.0']['recall']
precision_oximeter = report['1.0']['precision']
f1_oximeter = report['1.0']['f1-score']
specificity_oximeter = report['0.0']['recall']
acc_oximeter = report['accuracy']
auc_oximeter = roc_auc_score(admitted_test, test_pred_oximeter)
save_table3(model="Contrastive-Bayesian_oximeter", precision=precision_oximeter, recall=recall_oximeter, specificity=specificity_oximeter,
            auc=auc_oximeter, details=f"{experiment}", other=json.dumps({'host': os.uname()[1], 'f1': f1_oximeter,
                                                                    'acc': acc_oximeter}))

confusion_oximeter=admission_confusion_matrix(admitted_test, pred_positive_oximeter)
confusion_oximeter.savefig(os.path.join(output_dir, f"Confusion matrix oximeter - {experiment}.png"))
distplot_oximeter=admission_distplot(samples_oximeter['mean'].squeeze().cpu(), admitted_test, pred_positive_oximeter)
distplot_oximeter.savefig(os.path.join(output_dir, f'Density plots - oximeter {experiment}.png'))


#########################################################################################################################3
#                 CLINICAL
########################################################################################################################
# model_clinical=SimpleLogistic(in_features=x_data_clinical.shape[1],)
model_clinical = model_horseshoe

poutine.trace(model_clinical).get_trace(x_data_clinical).stochastic_nodes
poutine.trace(model_clinical).get_trace(x_data_clinical).param_nodes

pyro.set_rng_seed(4)
nuts_kernel = NUTS(model_clinical,  full_mass=full_mass,max_tree_depth=max_tree_depth)
mcmc = MCMC(nuts_kernel, num_samples=mcmc_samples, warmup_steps=mcmc_burnin, num_chains=1)
mcmc.run(x_data_clinical, y_data.reshape(-1, 1))
posterior_samples_clinical = mcmc.get_samples()
predictive_clinical = Predictive(model_clinical, posterior_samples_clinical)
samples_clinical = predictive_clinical(x_data_test_clinical, None)
joblib.dump(posterior_samples_clinical,
            os.path.join(data_dir, f"results/weights/LogisticBayesian_clinical_{experiment}.joblib"))
# posterior_samples_clinical=joblib.load(os.path.join(data_dir, f"results/weights/LogisticBayesian_clinical_{experiment}.joblib"))

test_pred_clinical = samples_clinical['mean'].squeeze().mean(dim=0).reshape(-1).cpu().numpy()
print("AUC clinical:",roc_auc_score(admitted_test, test_pred_clinical))

base_rate = admitted_train.mean()
pred_positive_clinical = (test_pred_clinical > base_rate) * 1.0
admission_confusion_matrix(admitted_test, pred_positive_clinical)

admission_distplot(samples_clinical['mean'].squeeze().cpu(), admitted_test, pred_positive_clinical)

report = classification_report(admitted_test, pred_positive_clinical, output_dict=True)
recall_clinical = report['1.0']['recall']
precision_clinical = report['1.0']['precision']
f1_clinical = report['1.0']['f1-score']
specificity_clinical = report['0.0']['recall']
acc_clinical = report['accuracy']
auc_clinical = roc_auc_score(admitted_test, test_pred_clinical)
save_table3(model="Contrastive-Bayesian_clinical", precision=precision_clinical, recall=recall_clinical,
            specificity=specificity_clinical,
            auc=auc_clinical, details=f"{experiment}", other=json.dumps({'host': os.uname()[1], 'f1': f1_clinical,
                                                                         'acc': acc_clinical}))

#######################################################################################################################
# Concatenated with clinical
######################################################################################################################
model_concat_clinical = model_horseshoe
# model_concat=MLP(in_features=x_data_concat.shape[1],dim_layers=[32,16,6,4,1],train_selection=True)
poutine.trace(model_concat_clinical).get_trace(x_data_concat).stochastic_nodes
poutine.trace(model_concat_clinical).get_trace(x_data_concat).param_nodes

pyro.set_rng_seed(5)
concat_clinical_kernel = NUTS(model_concat_clinical,  full_mass=full_mass,max_tree_depth=max_tree_depth)
mcmc = MCMC(concat_clinical_kernel, num_samples=mcmc_samples, warmup_steps=mcmc_burnin, num_chains=1)
mcmc.run(x_data_concat_clinical, y_data.reshape(-1, 1))
posterior_samples_concat_clinical = mcmc.get_samples()
predictive_concat_clinical = Predictive(model_concat_clinical, posterior_samples_concat_clinical)
samples_concat_clinical = predictive_concat_clinical(x_data_test_concat_clinical, None)
joblib.dump(posterior_samples_concat_clinical,
            os.path.join(data_dir, f"results/weights/LogisticBayesian_concat_clinical_{experiment}.joblib"))

test_pred_concat_clinical = samples_concat_clinical['mean'].mean(dim=0).reshape(-1).cpu().numpy()
print("AUC concatinated_clinical ", roc_auc_score(admitted_test, test_pred_concat_clinical))

base_rate = admitted_train.mean()
pred_positive_concat_clinical = (test_pred_concat_clinical > base_rate) * 1.0
confussion_concat_clinical=admission_confusion_matrix(admitted_test, pred_positive_concat_clinical)
confussion_concat_clinical.savefig(os.path.join(output_dir, f"Confusion matrix concat_clinical - {experiment}.png"))
distplot_concat_clinical = admission_distplot(samples_concat_clinical['mean'].squeeze().cpu(), admitted_test,
                                              pred_positive_concat_clinical)
distplot_concat_clinical.savefig(os.path.join(output_dir, f'Density plots - concat_clinical {experiment}.png'))

report = classification_report(admitted_test, pred_positive_concat_clinical, output_dict=True)
recall_concat_clinical = report['1.0']['recall']
precision_concat_clinical = report['1.0']['precision']
f1_concat_clinical = report['1.0']['f1-score']
specificity_concat_clinical = report['0.0']['recall']
acc_concat_clinical = report['accuracy']
auc_concat_clinical = roc_auc_score(admitted_test, test_pred_concat_clinical)
save_table3(model="Contrastive-Bayesian_concat_clinical", precision=precision_concat_clinical,
            recall=recall_concat_clinical, specificity=specificity_concat_clinical,
            auc=auc_concat_clinical, details=f"{experiment}",
            other=json.dumps({'host': os.uname()[1], 'f1': f1_concat_clinical,
                              'acc': acc_concat_clinical}))

p_admission = (samples_concat_clinical['mean'].squeeze() > base_rate).type(torch.float).mean(dim=0)
p_admission_lower = samples_concat_clinical['mean'].squeeze().quantile(q=0.05, dim=0)
p_admission_upper = samples_concat_clinical['mean'].squeeze().quantile(q=0.95, dim=0)
p_interquantile = p_admission_upper - p_admission_lower

fig, ax = plt.subplots()
sns.scatterplot(y=test_pred_concat_clinical, x=p_interquantile,
                # hue=admitted_test,
                hue=np.where(admitted_test, "Yes", "No"),
                # style=died_test,
                ax=ax,
                hue_order=["No", "Yes"],
                # hue_norm=(0.2,0.7),
                )
ax.axline((0, base_rate), slope=0)
ax.set_xlabel("Inter-quantile range of predicted probability")
ax.set_ylabel("Predicted probability")
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend()
plt.show()

#######################################################################################################################
# Concatenated with clinical
######################################################################################################################
model_concat_oximeter = model_horseshoe
# model_concat=MLP(in_features=x_data_concat.shape[1],dim_layers=[32,16,6,4,1],train_selection=True)
poutine.trace(model_concat_oximeter).get_trace(x_data_concat_oximeter).stochastic_nodes
poutine.trace(model_concat_oximeter).get_trace(x_data_concat_oximeter).param_nodes

pyro.set_rng_seed(6)
concat_oximeter_kernel = NUTS(model_concat_oximeter,  full_mass=full_mass,max_tree_depth=max_tree_depth)
mcmc = MCMC(concat_oximeter_kernel, num_samples=mcmc_samples, warmup_steps=mcmc_burnin, num_chains=1)
mcmc.run(x_data_concat_oximeter, y_data.reshape(-1, 1))
posterior_samples_concat_oximeter = mcmc.get_samples()
predictive = Predictive(model_concat_oximeter, posterior_samples_concat_oximeter)
samples_concat_oximeter = predictive(x_data_test_concat_oximeter, None)
test_pred_concat_oximeter = samples_concat_oximeter['mean'].mean(dim=0).reshape(-1).cpu().numpy()
joblib.dump(posterior_samples_concat_oximeter,
            os.path.join(data_dir, f"results/weights/LogisticBayesian_concat_oximeter_{experiment}.joblib"))

print("AUC concatinated_oximeter ", roc_auc_score(admitted_test, test_pred_concat_oximeter))

base_rate = admitted_train.mean()
pred_positive_concat_oximeter = (test_pred_concat_oximeter > base_rate) * 1.0
report = classification_report(admitted_test, pred_positive_concat_oximeter, output_dict=True)
recall_concat_oximeter = report['1.0']['recall']
precision_concat_oximeter = report['1.0']['precision']
f1_concat_oximeter = report['1.0']['f1-score']
specificity_concat_oximeter = report['0.0']['recall']
acc_concat_oximeter = report['accuracy']
auc_concat_oximeter = roc_auc_score(admitted_test, test_pred_concat_oximeter)
save_table3(model="Contrastive-Bayesian_concat_oximeter", precision=precision_concat_oximeter,
            recall=recall_concat_oximeter, specificity=specificity_concat_oximeter,
            auc=auc_concat_oximeter, details=f"{experiment}",
            other=json.dumps({'host': os.uname()[1], 'f1': f1_concat_oximeter,
                              'acc': acc_concat_oximeter}))


###################################################################################################################
## Concatinate predictions clinical & SSL
####################################################################################################################
prediction_concat_clinical=np.concatenate([samples_clinical['mean'].squeeze().cpu().numpy(),samples_ppg['mean'].squeeze().cpu().numpy()],axis=0)
test_pred_concat_clinical2=prediction_concat_clinical.mean(axis=0).reshape(-1)
roc_auc_score(admitted_test,test_pred_concat_clinical2)


base_rate=admitted_train.mean()
pred_positive_both=(test_pred_concat_clinical2>base_rate)*1.0
admission_confusion_matrix(admitted_test,pred_positive_both)

distplot_both=admission_distplot(prediction_concat_clinical.squeeze(),admitted_test,pred_positive_both)