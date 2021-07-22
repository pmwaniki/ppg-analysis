import gc

import pandas as pd
import numpy as np
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler,random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
# import torchaudio

import os
import json
from models.cnn.networks import resnet50_1d,EncoderRaw,wideresnet50_1d
from models.cnn.wavenet2 import WaveNetModel
from datasets.signals import stft,gaus_noise,rand_sfft,permute
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,classification_report,roc_curve,f1_score
from functools import partial
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine
import itertools
from datasets.loaders import TriageDataset,TriagePairs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from settings import data_dir,weights_dir
from settings import checkpoint_dir as log_dir
from utils import save_table3


import ray
ray.init(address="auto")

# ray.init( num_cpus=12,dashboard_host="0.0.0.0")
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import Stopper

import joblib
import copy
from utils import save_table3

display=os.environ.get('DISPLAY',None) is not None



enc_representation_size="32"
res_type="original" # wide,original
init= None #
# init = "Contrastive-original-sample-DotProduct32-sepsis" #contrastive experiment name or None
# init = "Contrastive-original-sample-DotProduct32" #contrastive experiment name or None

if init:
    weights_file = os.path.join(weights_dir, f"Contrastive_{init}.pt")
    init_weights=torch.load(weights_file)
    base_model_weights=OrderedDict()
    for k,v in init_weights.items():
        if "base_model" in k:
            base_model_weights[k.replace("base_model.","")]=v
# enc_distance="DotProduct" #LpDistance Dotproduct Cosine
# distance_fun="euclidean" if enc_distance=="LpDistance" else cosine
experiment=f"Supervised2-{res_type}-{enc_representation_size}"
if init: experiment=experiment + "__" + init

#
data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
triage_segments=pd.read_csv(os.path.join(data_dir,'triage/segments.csv'))
triage_segments['label']=pd.factorize(triage_segments['id'])[0]

train_val_ids,test_ids=joblib.load(os.path.join(data_dir,"triage/ids.joblib"))
np.random.seed(123)

#Train, validation and Test sets
train_val=triage_segments.loc[triage_segments['id'].isin(train_val_ids),:]
np.random.seed(254)
train_ids_ = np.random.choice(train_val_ids, size=int(len(train_val_ids) * 0.8), replace=False)
train = train_val[train_val['id'].isin(train_ids_)]
val = train_val[~train_val['id'].isin(train_ids_)]


test=triage_segments.loc[triage_segments['id'].isin(test_ids),:]







# sample weights
def balanced_sampler(y):
    class_sample_count = np.array(
        [len(np.where(y == t)[0]) for t in np.unique(y)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in y])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def get_train_loader(config,train_data):
    aug_raw = [
        partial(gaus_noise, min_sd=1e-5, max_sd=1e-1, p=config['aug_gaus']),
        partial(permute, n_segments=config['aug_num_seg'], p=config['aug_prop_seg']),
    ]

    train_ds = TriageDataset(train_data, labels='admitted', stft_fun=None,
                             transforms=None,
                             aug_raw=aug_raw,
                             normalize=True,
                             sample_by="id"
                             )



    # sample weights
    train_labels = []
    for i in range(len(train_ds)):
        _, lab = train_ds.__getitem__(i)
        train_labels.append(lab.numpy()[0])

    sampler = balanced_sampler(train_labels)

    train_loader = DataLoader(train_ds,
                              batch_size=int(config["batch_size"]),
                              shuffle=False, sampler=sampler, num_workers=5)


    return train_loader

def get_test_loader(val_data,sample_by=None):
    val_ds = TriageDataset(val_data, labels='admitted', stft_fun=None,
                           transforms=None,
                           aug_raw=[],
                           normalize=True,
                           sample_by=sample_by,
                           )
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=16, num_workers=5)
    return val_loader




def get_model(config):
    if res_type=="original":
        base_model = resnet50_1d(num_classes=32)
    elif res_type=="wide":
        base_model=wideresnet50_1d(num_classes=32)
    if init:
        base_model.load_state_dict(base_model_weights,strict=False)

    model = EncoderRaw(base_model, representation_size=config['representation_size'],
                       dropout=config['dropout'], num_classes=1)
    return model

def get_optimizer(config,model):
    optimizer = torch.optim.Adam(params=[
        {'params': model.base_model.parameters()},
        {'params': model.fc0.parameters(), 'lr': config['lr_fc'], 'weight_decay': config['l2_fc']},
        {'params': model.fc.parameters(), 'lr': config['lr_fc'], 'weight_decay': config['l2_fc']}
    ], lr=config['lr'], weight_decay=config['l2'],)

    return optimizer


device = "cuda" if torch.cuda.is_available() else "cpu"




def train_fun(model,optimizer,criterion,device,train_loader,val_loader,scheduler,max_iter):
    epoch = None if scheduler is None else scheduler.state_dict()['_step_count']
    if (epoch <= 40) and (init is not None):
        for param in model.base_model.parameters():
            param.requires_grad=False
    else:
        for param in model.base_model.parameters():
            param.requires_grad = True
    train_loss = 0

    model.train()
    for batch_x, batch_y in train_loader:
        batch_x,batch_y = batch_x.to(device, dtype=torch.float),batch_y.to(device,dtype=torch.float)
        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(train_loader)


    model.eval()
    val_loss = 0
    pred_val=[]
    obs_val=[]
    with torch.no_grad():
        for batch_x,batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device)
            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            val_loss += loss.item() / len(val_loader)
            pred_val.append(logits.sigmoid().squeeze().cpu().numpy().reshape(-1))
            obs_val.append(batch_y.squeeze().cpu().numpy().reshape(-1))
    if scheduler: scheduler.step()
    pred_val = np.concatenate(pred_val)
    obs_val = np.concatenate(obs_val)
    f1 = f1_score(obs_val,(pred_val>0.5)*1.0)
    auc=roc_auc_score(obs_val,pred_val)
    stop=epoch>=max_iter
    return train_loss,val_loss,f1,auc,stop

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        assert 0 <= self.smoothing < 1

    def forward(self, pred, target):
        with torch.no_grad():
            true_dist =target * (1 - self.smoothing) + self.smoothing / 2
        return torch.nn.functional.binary_cross_entropy_with_logits(pred,true_dist)

class Trainer(tune.Trainable):
    def setup(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model=get_model(config).to(self.device)
        self.optimizer=get_optimizer(config,self.model)

        # self.criterion=nn.BCEWithLogitsLoss().to(self.device)
        self.criterion=LabelSmoothingLoss(smoothing=config['smoothing']).to(self.device)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.5)
        self.train_loader=get_train_loader(config,train)
        self.val_loader=get_test_loader(val)
        self.max_iter=config['max_iter']


    def step(self):
        train_loss,loss,f1,auc,stop=train_fun(self.model,self.optimizer,self.criterion,
                            self.device,self.train_loader,self.val_loader,self.scheduler,self.max_iter)
        return {'loss':loss,'f1':f1,'auc':auc,'train_loss':train_loss,'stop':stop}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save((self.model.state_dict(),self.optimizer.state_dict()), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        model_state,optimizer_state=torch.load(checkpoint_path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)


configs = {
    'dropout':tune.loguniform(0.00001,0.5),
    'representation_size':tune.choice([32,]),
    'batch_size':tune.choice([8,16,32,64,128]),
    'smoothing':tune.choice([0.0,0.001,0.01,0.1,]),
    'lr':tune.loguniform(0.0001,0.1),
    'l2':tune.loguniform(0.000001,1.0),
    'lr_fc':tune.loguniform(0.0001,0.1),
    'l2_fc':tune.loguniform(0.000001,1.0),
    'aug_gaus':tune.choice([0,0.2,0.5,0.8,1.0]),
    'aug_num_seg':tune.choice([2,5,10,20,40,80]),
    'aug_prop_seg':tune.choice([0.05,0.1,0.3,0.5,0.9]),
    'max_iter':tune.choice([50,100,150,250,]),

}
# config={i:v.sample() for i,v in configs.items()}
# best_config={i:v.sample() for i,v in configs.items()}
epochs=250
scheduler = ASHAScheduler(
        metric="auc",
        mode="max",
        max_t=epochs,
        grace_period=50,
        reduction_factor=4)

class MaxIterStopper(Stopper):
    def __init__(self):
        pass

    def __call__(self, trial_id, result):
        return result['stop']

    def stop_all(self):
        return False

reporter = CLIReporter(
    metric_columns=["loss", "auc","f1", "training_iteration"])
# early_stopping=tune.stopper.EarlyStopping(metric='auc',top=10,mode='max',patience=10)
result = tune.run(
    Trainer,
    # metric='loss',
    # mode='min',
    checkpoint_at_end=True,
    resources_per_trial={"cpu": 4, "gpu": 0.3},
    config=configs,
    local_dir=os.path.join(log_dir, "Supervised"),
    num_samples=500,
    name=experiment,
    stop=MaxIterStopper(),
    resume=True,
    scheduler=scheduler,
    progress_reporter=reporter,
    reuse_actors=False,
    raise_on_failed_trial=False,
    # max_failures=1
)


df = result.results_df
metric='auc';mode="max"; scope='last'
print(result.get_best_trial(metric,mode,scope=scope).last_result)
# df.to_csv(os.path.join(data_dir, "results/hypersearch.csv"), index=False)
best_trial = result.get_best_trial(metric, mode, scope=scope)
best_config=result.get_best_config(metric,mode,scope=scope)


#dummy
# @ray.remote(num_cpus=3,num_gpus=0.25)
# def dummy(x):
#     return np.random.rand()


@ray.remote(num_cpus=3,num_gpus=0.5, max_calls=1)
def fit_bag(train_data):
    best_trainer = Trainer(best_config)
    train_loader=get_train_loader(best_config,train_data)
    test_loader=get_test_loader(test)
    best_trainer.train_loader = train_loader
    best_trainer.val_loader = test_loader
    metrics = []
    for epoch in range(best_config['max_iter']):
        r = best_trainer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch} | train loss {r["train_loss"]:.3f} | loss {r["loss"]:.3f} | auc {r["auc"]:.2f}')
        metrics.append(r)
    model_state=copy.deepcopy(best_trainer.model.state_dict())
    del best_trainer
    # gc.collect()
    return {'weights':model_state,'metrics':metrics}


bagging_results = []

remaining_trials=[]

for _ in range(10):
    bootstrap_data = train_val.sample(replace=True,frac=1.0)
    trial_id=fit_bag.remote(bootstrap_data)
    remaining_trials.append(trial_id)

jobs=0
while remaining_trials:
    done_trials, remaining_trials = ray.wait(remaining_trials)
    result_id = done_trials[0]
    done_result = ray.get(result_id)
    bagging_results.append(done_result)
    print(f"No of jobs done: {jobs+1}")
    jobs+=1

# bagging prediction
test_loader=get_test_loader(test)
bagging_test_pred=[]
for bag in bagging_results:
    best_model=get_model(best_config)
    bag_weights=bag['weights']
    best_model.load_state_dict(bag_weights)
    best_model.to(device)
    best_model.eval()
    pred_test = []
    obs_test = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader):
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device)
            logits = best_model(batch_x)
            pred_test.append(logits.sigmoid().squeeze().cpu().numpy().reshape(-1))
            obs_test.append(batch_y.squeeze().cpu().numpy().reshape(-1))
    pred_test = np.concatenate(pred_test)
    obs_test = np.concatenate(obs_test)
    bagging_test_pred.append(pred_test)

bagging_test_pred2=np.stack(bagging_test_pred)
bagging_test_pred2=bagging_test_pred2.mean(axis=0)









test_d=test.copy()
test_d['pred']=bagging_test_pred2
test_d2=test_d.groupby(['id'])[['admitted','pred']].mean()

f1 = f1_score(test_d2['admitted'],(test_d2['pred']>0.5)*1.0)
auc=roc_auc_score(test_d2['admitted'],test_d2['pred'])
report=classification_report(test_d2['admitted'],(test_d2['pred']>0.5)*1.0,output_dict=True)
recall=report['1.0']['recall']
precision=report['1.0']['precision']
f1=report['1.0']['f1-score']
specificity=report['0.0']['recall']
acc=report['accuracy']

torch.save(bagging_results,os.path.join(weights_dir,"end to end.pth"))

save_table3(model="End to end",precision=precision,recall=recall,specificity=specificity,auc=auc,
            details=json.dumps({'init':init,'exp':experiment}),
            other=json.dumps({'host':os.uname()[1],'config':best_config}))



