import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler,random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
# import torchaudio
from settings import data_dir
from settings import checkpoint_dir as log_dir
import os
import json
from models.networks import init_fun
from models.cnn.networks import resnet1d,EncoderRaw,MLP
from models.cnn.wavenet2 import WaveNetModel
from datasets.signals import stft,gaus_noise,rand_sfft,permute
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,classification_report,roc_curve
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
from settings import Fs,weights_dir
from pytorch_metric_learning import distances,regularizers,losses,testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import ray
ray.init( num_cpus=12,dashboard_host="0.0.0.0")
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler,HyperBandScheduler,AsyncHyperBandScheduler

import joblib
import copy
from utils import save_table3

display=os.environ.get('DISPLAY',None) is not None


# enc_lr_=0.001
# enc_l2_=5e-4
# enc_l1_=0.001
# dropout_=0.05
enc_representation_size="32"
enc_distance="DotProduct" #LpDistance Dotproduct Cosine
distance_fun="euclidean" if enc_distance=="LpDistance" else cosine
pretext="sample" #sample, augment
experiment=f"Contrastive-{pretext}-{enc_distance}{enc_representation_size}e"


# weights_file=os.path.join(weights_dir,f"triplet_lr{enc_lr_}_l2{enc_l2_}_z{enc_representation_size}_x{enc_output_size}_bs{enc_batch_size}.pt")
# details_=f"z{enc_representation_size}_l2{enc_l2_}_x{enc_output_size}_lr{enc_lr_}_bs{enc_batch_size}"
#
#
data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
triage_segments=pd.read_csv(os.path.join(data_dir,'triage/segments.csv'))
# triage_segments['label']=pd.factorize(triage_segments['id'])[0]

train_ids,test_ids=joblib.load(os.path.join(data_dir,"triage/ids.joblib"))
# np.random.seed(123)
# train_encoder_ids=np.random.choice(train_ids,size=660,replace=False)
#
#
train=triage_segments[triage_segments['id'].isin(train_ids)]
# train_full=train.copy()
non_repeated_ids = [k for k, v in train['id'].value_counts().items() if v <= 15]
# train = train[~train['id'].isin(non_repeated_ids)]  # remove non repeating ids
train_encoder = train[~train['id'].isin(non_repeated_ids)]
# test_encoder = train[~train['id'].isin(train_encoder_ids)]
# test=triage_segments[triage_segments['id'].isin(test_ids)]
#
#
test=triage_segments[triage_segments['id'].isin(test_ids)]
# test_full=test.copy()
non_repeated_test_ids = [k for k, v in test['id'].value_counts().items() if v <= 15]
test_encoder = test[~test['id'].isin(non_repeated_test_ids)]

encoder_test_dataset = TriagePairs(test_encoder, id_var="id", stft_fun=None,
                                        transforms=None,
                                        aug_raw=[],
                                        normalize=True,pretext=pretext)
encoder_test_loader = DataLoader(encoder_test_dataset, batch_size=16, shuffle=False, num_workers=5)
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
classifier_train_dataset=TriageDataset(train,normalize=True)
classifier_train_loader=DataLoader(classifier_train_dataset,batch_size=16,shuffle=False,num_workers=5)

classifier_test_dataset=TriageDataset(test,normalize=True)
classifier_test_loader=DataLoader(classifier_test_dataset,batch_size=16,shuffle=False,num_workers=5)
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




def accuracy_fun(xis_embeddings, xjs_embeddings, distance="minkowski"):
    knn = KNeighborsClassifier(n_neighbors=1,metric=distance)
    knn.fit(xis_embeddings, np.arange(xis_embeddings.shape[0]))
    knn_pred = knn.predict(xjs_embeddings)
    accuracy = np.mean(np.arange(xis_embeddings.shape[0]) == knn_pred)
    return accuracy

def get_loader(config):
    aug_raw=[
    partial(gaus_noise,min_sd=1e-5,max_sd=1e-1,p=config['aug_gaus']),
    partial(permute,n_segments=config['aug_num_seg'],p=config['aug_prop_seg']),
    ]
    all_ids = train_encoder['id'].unique()
    np.random.seed(123)
    train_ids = np.random.choice(all_ids, size=int(len(all_ids) * 0.85), replace=False)
    train_csv = train_encoder[train_encoder['id'].isin(train_ids)]
    val_csv = train_encoder[~train_encoder['id'].isin(train_ids)]
    train_ds=TriagePairs(train_csv, id_var="id", stft_fun=None,
                                        transforms=None,
                                        aug_raw=aug_raw,
                                        normalize=True,
                                        pretext=pretext
                                        )
    val_ds=TriagePairs(val_csv, id_var="id", stft_fun=None,
                                        transforms=None,
                                        aug_raw=[],
                                        normalize=True,pretext=pretext
                                        )




    train_loader = DataLoader(train_ds,
                              batch_size=int(config["batch_size"]),
                              shuffle=True, num_workers=50)
    val_loader = DataLoader(val_ds,
                            batch_size=int(config["batch_size"]),
                            shuffle=False, num_workers=50)
    return train_loader,val_loader

def get_model(config):
    base_model = resnet1d(num_classes=512)

    model = EncoderRaw(base_model, representation_size=config['representation_size'],
                       dropout=config['dropout'], num_classes=config['enc_output_size'])
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    return model

def get_optimizer(config,model):
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config['enc_lr'],
                                 weight_decay=config['enc_l2'])
    return optimizer


device = "cuda" if torch.cuda.is_available() else "cpu"




def train_fun(model,optimizer,criterion,device,train_loader,val_loader,scheduler):
    train_loss = 0

    model.train()
    # print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch, optimizer.param_groups[0]['lr']))
    for x1_raw, x2_raw in train_loader:
        x1_raw = x1_raw.to(device, dtype=torch.float)
        xis = model(x1_raw)
        x2_raw = x2_raw.to(device, dtype=torch.float)
        xjs = model(x2_raw)
        xis = nn.functional.normalize(xis, dim=1)
        xjs = nn.functional.normalize(xjs, dim=1)
        embeddings = torch.cat([xis, xjs], dim=0)
        labels = torch.cat([torch.arange(xis.shape[0]), ] * 2).to(device)
        loss = criterion(embeddings, labels)
        # l1_reg = config['enc_l1'] * torch.norm(embeddings, p=1, dim=1).mean()
        # loss += l1_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(train_loader)
        # metrics=calculator.get_accuracy(xis.cpu().detach().numpy(),xjs.cpu().detach().numpy(),
        #                         np.arange(xis.shape[0]),np.arange(xis.shape[0]),
        #                         False)

    model.eval()
    val_loss = 0
    xis_embeddings = []
    xjs_embeddings = []
    with torch.no_grad():
        for x1_raw, x2_raw in val_loader:
            x1_raw = x1_raw.to(device, dtype=torch.float)
            xis = model(x1_raw)
            x2_raw = x2_raw.to(device, dtype=torch.float)
            xjs = model(x2_raw)
            xis = nn.functional.normalize(xis, dim=1)
            xjs = nn.functional.normalize(xjs, dim=1)
            embeddings = torch.cat([xis, xjs], dim=0)
            labels = torch.cat([torch.arange(xis.shape[0]), ] * 2)
            loss = criterion(embeddings, labels)
            # l1_reg = config['enc_l1'] * torch.norm(embeddings, p=1, dim=1).sum()
            # loss += l1_reg

            val_loss += loss.item() / len(val_loader)
            xis_embeddings.append(xis.cpu().detach().numpy())
            xjs_embeddings.append(xjs.cpu().detach().numpy())
            # metrics = calculator.get_accuracy(xis.cpu().detach().numpy(), xjs.cpu().detach().numpy(),
            #                                   np.arange(xis.shape[0]), np.arange(xis.shape[0]),
            #                                   False)
        scheduler.step(val_loss)
        xis_embeddings = np.concatenate(xis_embeddings)
        xjs_embeddings = np.concatenate(xjs_embeddings)
        accuracy = accuracy_fun(xis_embeddings, xjs_embeddings,distance=distance_fun)
    return train_loss,val_loss,accuracy


class Trainer(tune.Trainable):
    def setup(self, config):
        self.model=get_model(config).to(device)
        self.optimizer=get_optimizer(config,self.model)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,factor=0.5,mode='min',patience=200)
        if enc_distance == "LpDistance":
            dist_fun=distances.DotProductSimilarity(normalize_embeddings=False)
        elif enc_distance =="DotProduct":
            dist_fun=distances.DotProductSimilarity(normalize_embeddings=False)
        elif enc_distance == "Cosine":
            dist_fun=distances.CosineSimilarity()
        else:
            raise NotImplementedError(f"{enc_distance} not implemented yet!!!")
        self.criterion=losses.NTXentLoss(temperature=config['enc_temp'],
                                  distance=dist_fun,
                                  ).to(device)
        self.train_loader,self.val_loader=get_loader(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def step(self):
        train_loss,loss,accuracy=train_fun(self.model,self.optimizer,self.criterion,
                            self.device,self.train_loader,self.val_loader,self.scheduler)
        return {'loss':loss,'accuracy':accuracy,'train_loss':train_loss}

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
    'enc_output_size':tune.choice([32,]),
    'representation_size':tune.choice([32,]),
    'batch_size':tune.choice([8,16,32,64,128]),
    'enc_temp':tune.loguniform(0.0001,0.5),
    'enc_lr':tune.loguniform(0.0001,0.5),
    'enc_l2':tune.loguniform(0.00001,1.0),
    # 'enc_l1':tune.loguniform(0.000001,0.05),
    'aug_gaus':tune.choice([0.0,0.2,0.5,0.8,1.0]) if pretext == "sample" else tune.choice([1.0,]),
    'aug_num_seg':tune.choice([2,5,7,10]),
    'aug_prop_seg':tune.choice([0.0,0.2,0.5,0.8,1.0]) if pretext == "sample" else tune.choice([1.0,]),

}

config={i:v.sample() for i,v in configs.items()}

scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=700,
        grace_period=10,
        reduction_factor=2)
# scheduler=AsyncHyperBandScheduler(
#         time_attr="training_iteration",
#         metric="loss",
#         mode="min",
#         grace_period=5,
#         max_t=700)
# scheduler = HyperBandScheduler(metric="loss", mode="min")

if __name__ == "__main__":

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        Trainer,
        # metric='loss',
        # mode='min',
        checkpoint_at_end=True,
        resources_per_trial={"cpu": 4, "gpu": 0.3},
        config=configs,
        local_dir=os.path.join(log_dir, "contrastive"),
        num_samples=500,
        name=experiment,
        # resume=True,
        scheduler=scheduler,
        progress_reporter=reporter,
        reuse_actors=True,
        raise_on_failed_trial=False)

    df = result.results_df
    df.to_csv(os.path.join(data_dir, f"results/hypersearch-{experiment}.csv"), index=False)
    best_trial = result.get_best_trial("loss", "min", "last")
    best_config=result.get_best_config('loss','min')

    best_model=get_model(best_config)
    best_trial=result.get_best_trial('loss','min')
    best_checkpoint=result.get_best_checkpoint(best_trial,'loss','min')
    model_state,optimizer_state=torch.load(best_checkpoint)


    best_model.load_state_dict(model_state)
    best_model.to(device)
    # Test model accuracy
    best_model.eval()
    xis_embeddings = []
    xjs_embeddings = []
    with torch.no_grad():
        for x1_raw, x2_raw in encoder_test_loader:
            x1_raw = x1_raw.to(device, dtype=torch.float)
            xis = best_model(x1_raw)
            x2_raw = x2_raw.to(device, dtype=torch.float)
            xjs = best_model(x2_raw)
            xis = nn.functional.normalize(xis, dim=1)
            xjs = nn.functional.normalize(xjs, dim=1)
            xis_embeddings.append(xis.cpu().detach().numpy())
            xjs_embeddings.append(xjs.cpu().detach().numpy())

        xis_embeddings = np.concatenate(xis_embeddings)
        xjs_embeddings = np.concatenate(xjs_embeddings)
        accuracy = accuracy_fun(xis_embeddings, xjs_embeddings,distance=distance_fun)

        best_model.fc=nn.Identity()
        best_model.eval()

        classifier_embedding=[]
        with torch.no_grad():
            for x1_raw in tqdm(classifier_train_loader):
                x1_raw = x1_raw.to(device, dtype=torch.float)
                xis = best_model( x1_raw)
                xis=nn.functional.normalize(xis,dim=1)
                classifier_embedding.append(xis.cpu().detach().numpy())

        test_embedding=[]
        with torch.no_grad():
            for x1_raw in tqdm(classifier_test_loader):
                x1_raw = x1_raw.to(device, dtype=torch.float)
                xis = best_model( x1_raw)
                xis = nn.functional.normalize(xis, dim=1)
                test_embedding.append(xis.cpu().detach().numpy())

        classifier_embedding=np.concatenate(classifier_embedding)
        test_embedding=np.concatenate(test_embedding)

        joblib.dump((classifier_embedding,test_embedding,train,test),
                    os.path.join(data_dir,f"results/{experiment}.joblib"))



