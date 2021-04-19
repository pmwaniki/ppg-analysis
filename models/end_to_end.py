import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler,random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
# import torchaudio

import os
import json
from models.networks import init_fun
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
from ray.tune.schedulers import ASHAScheduler,HyperBandScheduler,AsyncHyperBandScheduler

import joblib
import copy
from utils import save_table3

display=os.environ.get('DISPLAY',None) is not None



enc_representation_size="32"
res_type="original" # wide,original
init="random"
# enc_distance="DotProduct" #LpDistance Dotproduct Cosine
# distance_fun="euclidean" if enc_distance=="LpDistance" else cosine
experiment=f"Supervised-{res_type}-{enc_representation_size}"
# enc_output_size=64
# enc_batch_size=64
# enc_temp = 0.05
# weights_file=os.path.join(weights_dir,f"triplet_lr{enc_lr_}_l2{enc_l2_}_z{enc_representation_size}_x{enc_output_size}_bs{enc_batch_size}.pt")
#
# fs=Fs
#
# slice_sec=2.5
# slide_sec=0.2
#
#
#
# nperseg=int(slice_sec*fs)
# step=int(slide_sec*fs)
# noverlap=nperseg-step
#
# weights_file=os.path.join(weights_dir,f"triplet_lr{enc_lr_}_l2{enc_l2_}_z{enc_representation_size}_x{enc_output_size}_bs{enc_batch_size}.pt")
# details_=f"z{enc_representation_size}_l2{enc_l2_}_x{enc_output_size}_lr{enc_lr_}_bs{enc_batch_size}"
#
#
data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
triage_segments=pd.read_csv(os.path.join(data_dir,'triage/segments.csv'))
triage_segments['label']=pd.factorize(triage_segments['id'])[0]

train_ids,test_ids=joblib.load(os.path.join(data_dir,"triage/ids.joblib"))
np.random.seed(123)
# val_ids=np.random.choice(train_ids,size=165,replace=False)
#
#
train=triage_segments.loc[triage_segments['id'].isin(train_ids),:]
# val=train[train['id'].isin(val_ids)]
# train=train[~train['id'].isin(val_ids)]


test=triage_segments.loc[triage_segments['id'].isin(test_ids),:]


# encoder_test_dataset = TriagePairs(test_encoder, id_var="id", stft_fun=None, aug_raw=[],normalize=True)
# encoder_test_loader = DataLoader(encoder_test_dataset, batch_size=16, shuffle=False, num_workers=5)
#
# train_dataset = TriagePairs(train_encoder, id_var="id", stft_fun=None,
#                             transforms=None,
#                             # aug_raw=aug_raw,normalize=True
#                             )
# encoder_train_loader = DataLoader(encoder_train_dataset, batch_size=enc_batch_size, shuffle=True, num_workers=50)
#
# encoder_test_dataset = TriagePairs(test_encoder, id_var="id", stft_fun=None, aug_raw=[],normalize=True)
# encoder_test_loader = DataLoader(encoder_test_dataset, batch_size=16, shuffle=False, num_workers=5)
#
# classifier_train_dataset=TriageDataset(train_full,normalize=True)
# classifier_train_loader=DataLoader(classifier_train_dataset,batch_size=16,shuffle=False,num_workers=5)
#
# classifier_test_dataset=TriageDataset(test_full,normalize=True)
# classifier_test_loader=DataLoader(classifier_test_dataset,batch_size=16,shuffle=False,num_workers=5)
#
# iter_ = iter(encoder_train_loader)
# x1_raw,  x2_raw = iter_.next()
# print("Input shape", x1_raw.shape)
#
# if display:
#     fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#
#     for r in range(2):
#         for c in range(3):
#             axs[r, c].plot(x1_raw.numpy()[c, r, :],
#                              # vmin=batch_x.numpy()[:, r, :, :].min(),
#                              # vmax=batch_x.numpy()[:, r, :, :].max(),
#                              # norm=colors.LogNorm()
#                              )
#
#     plt.show()
# if display:
#     fig, axs = plt.subplots(4, 3, figsize=(12, 5))
#     for r in range(2):
#         for c in range(3):
#             axs[r * 2 + 1, c].imshow(x1_stft.numpy()[c, r, :, :],
#                                      # vmin=batch_x.numpy()[:, r, :, :].min(),
#                                      # vmax=batch_x.numpy()[:, r, :, :].max(),
#                                      norm=colors.LogNorm()
#                                      )
#             axs[r * 2 + 1, c].invert_yaxis()
#     for r in range(2):
#         for c in range(3):
#             axs[r * 2, c].plot(x1_raw.numpy()[c, r, :],
#                                # vmin=batch_x.numpy()[:, r, :, :].min(),
#                                # vmax=batch_x.numpy()[:, r, :, :].max(),
#                                # norm=colors.LogNorm()
#                                )
#
#     plt.show()






def get_loader(config):
    aug_raw=[
    partial(gaus_noise,min_sd=1e-5,max_sd=1e-1,p=config['aug_gaus']),
    partial(permute,n_segments=config['aug_num_seg'],p=config['aug_prop_seg']),
    ]
    all_ids = train['id'].unique()
    np.random.seed(123)
    train_ids_ = np.random.choice(all_ids, size=int(len(all_ids) * 0.85), replace=False)
    train_csv = train[train['id'].isin(train_ids_)]
    val_csv = train[~train['id'].isin(train_ids_)]
    train_ds = TriageDataset(train_csv, labels='admitted', stft_fun=None,
                             transforms=None,
                             aug_raw=aug_raw,
                             normalize=True,
                             sample_by="id"
                             )
    val_ds = TriageDataset(val_csv, labels='admitted', stft_fun=None,
                           transforms=None,
                           aug_raw=[],
                           normalize=True,
                           sample_by='id'
                           )

    # sample weights
    train_labels=[]
    for i in range(len(train_ds)):
        _,lab=train_ds.__getitem__(i)
        train_labels.append(lab.numpy()[0])

    class_sample_count = np.array(
        [len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in train_labels])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_ds,
                              batch_size=int(config["batch_size"]),
                              shuffle=False,sampler=sampler, num_workers=50)
    val_loader = DataLoader(val_ds,
                            batch_size=int(config["batch_size"]),
                            shuffle=False, num_workers=50)
    return train_loader,val_loader

def get_model(config):
    if res_type=="original":
        base_model = resnet50_1d(num_classes=32)
    elif res_type=="wide":
        base_model=wideresnet50_1d(num_classes=32)

    model = EncoderRaw(base_model, representation_size=config['representation_size'],
                       dropout=config['dropout'], num_classes=1)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    return model

def get_optimizer(config,model):
    optimizer = torch.optim.SGD(params=[
        {'params': model.base_model.parameters()},
        {'params': model.fc0.parameters(), 'lr': config['lr_fc'], 'weight_decay': config['l2_fc']},
        {'params': model.fc.parameters(), 'lr': config['lr_fc'], 'weight_decay': config['l2_fc']}
    ], lr=config['lr'], weight_decay=config['l2'],momentum=0.99)

    return optimizer


device = "cuda" if torch.cuda.is_available() else "cpu"




def train_fun(model,optimizer,criterion,device,train_loader,val_loader,scheduler):
    train_loss = 0

    model.train()
    # print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch, optimizer.param_groups[0]['lr']))
    for batch_x, batch_y in train_loader:
        batch_x,batch_y = batch_x.to(device, dtype=torch.float),batch_y.to(device)
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
    return train_loss,val_loss,f1,auc


class Trainer(tune.Trainable):
    def setup(self, config):
        self.model=get_model(config).to(device)
        self.optimizer=get_optimizer(config,self.model)

        self.criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config['pos_weight'])).to(device)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.5)
        self.train_loader,self.val_loader=get_loader(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def step(self):
        train_loss,loss,f1,auc=train_fun(self.model,self.optimizer,self.criterion,
                            self.device,self.train_loader,self.val_loader,self.scheduler)
        return {'loss':loss,'f1':f1,'auc':auc,'train_loss':train_loss}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save((self.model.state_dict(),self.optimizer.state_dict()), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        model_state,optimizer_state=torch.load(checkpoint_path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)


configs = {
    'dropout':tune.loguniform(0.01,0.5),
    'representation_size':tune.choice([32,]),
    'batch_size':tune.choice([8,16,32,64,128]),
    'pos_weight':tune.choice([1.0,]),
    'lr':tune.loguniform(0.00001,0.1),
    'l2':tune.loguniform(0.000001,0.5),
    'lr_fc':tune.loguniform(0.00001,0.5),
    'l2_fc':tune.loguniform(0.000001,1.0),
    'aug_gaus':tune.choice([0,0.2,0.5,0.8,1.0]),
    'aug_num_seg':tune.choice([2,5,10,20,40,80]),
    'aug_prop_seg':tune.choice([0.05,0.1,0.3,0.5,0.9]),

}
config={i:v.sample() for i,v in configs.items()}

scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=700,
        grace_period=50,
        reduction_factor=2)
# scheduler=AsyncHyperBandScheduler(
#         time_attr="training_iteration",
#         metric="loss",
#         mode="min",
#         grace_period=5,
#         max_t=700)
# scheduler = HyperBandScheduler(metric="loss", mode="min")


reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy","f1", "training_iteration"])
result = tune.run(
    Trainer,
    # metric='loss',
    # mode='min',
    checkpoint_at_end=True,
    resources_per_trial={"cpu": 4, "gpu": 0.3},
    config=configs,
    local_dir=os.path.join(log_dir, "Supervised"),
    num_samples=200,
    name=experiment,
    resume=False,
    scheduler=scheduler,
    progress_reporter=reporter,
    reuse_actors=False,
    raise_on_failed_trial=False)
df = result.results_df
# df.to_csv(os.path.join(data_dir, "results/hypersearch.csv"), index=False)
best_trial = result.get_best_trial("auc", "max", "last")
best_config=result.get_best_config('auc','max')
best_trainer=Trainer(best_config)


aug_raw=[
    partial(gaus_noise,min_sd=1e-5,max_sd=1e-1,p=best_config['aug_gaus']),
    partial(permute,n_segments=best_config['aug_num_seg'],p=best_config['aug_prop_seg']),
    ]
train_dataset = TriageDataset(train, labels='admitted', stft_fun=None,
                         transforms=None,
                         aug_raw=aug_raw,
                         normalize=True,
                         sample_by="id"
                         )


# sample weights
train_labels=[]
for i in range(len(train_dataset)):
    _,lab=train_dataset.__getitem__(i)
    train_labels.append(lab.numpy()[0])

class_sample_count = np.array(
    [len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[int(t)] for t in train_labels])

samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = DataLoader(train_dataset,
                          batch_size=int(best_config["batch_size"]),
                          shuffle=False,sampler=sampler, num_workers=15)

test_dataset=TriageDataset(test, labels='admitted', stft_fun=None,
                           transforms=None,
                           aug_raw=[],
                           normalize=True,
                           sample_by=None
                           )
test_loader=DataLoader(test_dataset,shuffle=False,batch_size=16,num_workers=15)

best_trainer.train_loader=train_loader
best_trainer.val_loader=test_loader


metrics=[]
for epoch in range(700):
    r=best_trainer.step()
    print(f'Epoch {epoch} | train loss {r["train_loss"]:.3f} | loss {r["loss"]:.3f} | auc {r["auc"]:.2f}')
    # metrics.append((train_loss,val_loss,f1,auc))

best_model=best_trainer.model


best_model.to(device)
# Test model accuracy


# criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5)).to(device)
best_model.eval()
# test_loss = 0
pred_test=[]
obs_test=[]
with torch.no_grad():
    for batch_x,batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device)
        logits = best_model(batch_x)
        # loss = criterion(logits, batch_y)
        # test_loss += loss.item() / len(test_loader)
        pred_test.append(logits.sigmoid().squeeze().cpu().numpy().reshape(-1))
        obs_test.append(batch_y.squeeze().cpu().numpy().reshape(-1))

pred_test = np.concatenate(pred_test)
obs_test = np.concatenate(obs_test)

test_d=test.copy()
test_d['pred']=pred_test
test_d2=test_d.groupby(id)[['admitted','pred']].mean()

f1 = f1_score(test_d2['admitted'],(test_d2['pred']>0.5)*1.0)
auc=roc_auc_score(test_d2['admitted'],test_d2['pred'])
report=classification_report(test_d2['admitted'],(test_d2['pred']>0.5)*1.0,output_dict=True)
recall=report['1.0']['recall']
precision=report['1.0']['precision']
f1=report['1.0']['f1-score']
specificity=report['0.0']['recall']
acc=report['accuracy']

torch.save(best_model.state_dict(),os.path.join(weights_dir,"end to end.pth"))

save_table3(model="End to end",precision=precision,recall=recall,specificity=specificity,auc=auc,
            details=json.dumps({'init':'random','exp':experiment}),
            other=json.dumps({'host':os.uname()[1],'config':best_config}))



