import matplotlib.pyplot as plt
import os
import json
import sys
import multiprocessing

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler





from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,PolynomialFeatures
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif,SelectPercentile,VarianceThreshold,RFECV
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,LeaveOneOut,StratifiedKFold,RandomizedSearchCV,RepeatedStratifiedKFold
from settings import data_dir,weights_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import save_table3
import ray
from ray import tune

ray.init(dashboard_host="0.0.0.0")


rng=np.random.RandomState(123)
# device="cuda" if torch.cuda.is_available() else "cpu"

experiment="Contrastive-original-sample-DotProduct32"
weights_file=os.path.join(weights_dir,f"Classification_{experiment}.joblib")
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")


classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)

train_ids=train['id'].unique()
test_ids=test['id'].unique()
classifier_embedding_reduced=np.stack(map(lambda id:classifier_embedding[train['id']==id,:].mean(axis=0) ,train_ids))
test_embedding_reduced=np.stack(map(lambda id:test_embedding[test['id']==id,:].mean(axis=0) ,test_ids))
admitted_train=np.stack(map(lambda id:train.loc[train['id']==id,'admitted'].iat[0],train_ids))
admitted_test=np.stack(map(lambda id:test.loc[test['id']==id,'admitted'].iat[0],test_ids))

# #preprocessing
# preprocess_pipeline = Pipeline([
#     ('var_threshold', VarianceThreshold()),
#     ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
#     ('scl',StandardScaler())
# ]
# )

#Tensor datasets
val_ids=rng.choice([True,False],size=classifier_embedding.shape[0],p=(0.2,0.8))
train_x=classifier_embedding_reduced#[~val_ids,:]
train_y=admitted_train#[~val_ids]
val_x=classifier_embedding_reduced[val_ids,:]
val_y=admitted_train[val_ids]
test_x=test_embedding_reduced
test_y=admitted_test

train_x_scl=preprocess_pipeline.fit_transform(train_x)
val_x_scl=preprocess_pipeline.transform(val_x)
test_x_scl=preprocess_pipeline.transform(test_x)

train_dataset=TensorDataset(torch.tensor(train_x_scl),torch.tensor(train_y))
val_dataset=TensorDataset(torch.tensor(val_x_scl),torch.tensor(val_y))
test_dataset=TensorDataset(torch.tensor(test_x_scl),torch.tensor(test_y))







class Net(nn.Module):
    def __init__(self,dim_x=32):
        super().__init__()
        self.linear=nn.Linear(dim_x,1)
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.constant_(self.linear.bias,0.0)

    def forward(self,x):
        return self.linear(x)



def make_loader(ds,**kwargs):
    y=[]
    for i in range(len(ds)):
        _,y_=ds.__getitem__(i)
        y.append(y_)
    y=torch.as_tensor(y).numpy()
    class_sample_count = np.array(
        [len(np.where(y == t)[0]) for t in np.unique(y)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in y])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    loader=DataLoader(ds,sampler=sampler,**kwargs)
    return loader



class SmoothLoss(nn.BCEWithLogitsLoss):
    def __init__(self,smoothing=0.0,**kwargs):
        super(SmoothLoss, self).__init__(**kwargs)
        self.smoothing=smoothing
    def forward(self, input, target):
        with torch.no_grad():
            true_dist = target * (1 - self.smoothing) + self.smoothing / 2
        return super().forward(input,true_dist)







grid_parameters = {
    'lr':[0.01,0.001,0.0001,0.00001,0.000001],
    'weight_decay':[0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0],
    'momentum':[0.9,0.99],
    'smoothing':[0.0,0.1,0.2,0.3,0.5],
    'batch_size':[32,64,128,256],
    # 'max_epochs':[10,30,50,100,300],



}









test_pred=clf.predict_proba(test_x_scl)[:,1]

print(classification_report(test_y,test_pred>0.5))
print("AUC: ",roc_auc_score(test_y,test_pred))



report=classification_report(admitted_test,(test_pred>0.5)*1.0,output_dict=True)
recall=report['1.0']['recall']
precision=report['1.0']['precision']
f1=report['1.0']['f1-score']
specificity=report['0.0']['recall']
acc=report['accuracy']
auc=roc_auc_score(admitted_test,test_pred)
save_table3(model="Contrastive",precision=precision,recall=recall,specificity=specificity,
            auc=auc,details=experiment,other=json.dumps({'host':os.uname()[1],'f1':f1,
                                                       'acc':acc}))

joblib.dump(clf,weights_file)
