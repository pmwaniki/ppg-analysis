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
ray.init( num_cpus=10)
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler,HyperBandScheduler,AsyncHyperBandScheduler

import joblib
import copy
from utils import save_table3

display=os.environ.get('DISPLAY',None) is not None


enc_representation_size="32"
enc_distance="DotProduct" #LpDistance Dotproduct Cosine
distance_fun="euclidean" if enc_distance=="LpDistance" else cosine
experiment=f"Contrastive-Augmentation-{enc_distance}{enc_representation_size}"

data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
triage_segments=pd.read_csv(os.path.join(data_dir,'triage/segments.csv'))
triage_segments['label']=pd.factorize(triage_segments['id'])[0]

train_ids,test_ids=joblib.load(os.path.join(data_dir,"triage/ids.joblib"))
np.random.seed(123)
train_encoder_ids=np.random.choice(train_ids,size=660,replace=False)
#
#
train=triage_segments[triage_segments['id'].isin(train_ids)]
train_full=train.copy()
non_repeated_ids = [k for k, v in train['id'].value_counts().items() if v <= 15]
train = train[~train['id'].isin(non_repeated_ids)]  # remove non repeating ids
# train_encoder = train[train['id'].isin(train_encoder_ids)]
# test_encoder = train[~train['id'].isin(train_encoder_ids)]
# test=triage_segments[triage_segments['id'].isin(test_ids)]
#
#
test=triage_segments[triage_segments['id'].isin(test_ids)]
test_full=test.copy()
# non_repeated_test_ids = [k for k, v in test['id'].value_counts().items() if v <= 15]
# test_encoder = test[~test['id'].isin(non_repeated_test_ids)]

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
classifier_train_dataset=TriageDataset(train_full,normalize=True)
classifier_train_loader=DataLoader(classifier_train_dataset,batch_size=16,shuffle=False,num_workers=5)

classifier_test_dataset=TriageDataset(test_full,normalize=True)
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
    partial(gaus_noise,min_sd=1e-5,max_sd=1e-1,p=1.0),
    partial(permute,n_segments=config['aug_num_seg'],p=1.0),
    ]
    all_ids = train['id'].unique()
    np.random.seed(123)
    train_ids_ = np.random.choice(all_ids, size=int(len(all_ids) * 0.85), replace=False)
    train_csv = train[train['id'].isin(train_ids_)]
    val_csv = train[~train['id'].isin(train_ids_)]
    train_ds=TriagePairs(train_csv, id_var="id", stft_fun=None,
                                        transforms=None,
                                        aug_raw=aug_raw,
                                        normalize=True
                                        )
    val_ds=TriagePairs(val_csv, id_var="id", stft_fun=None,
                                        transforms=None,
                                        aug_raw=[],
                                        normalize=True
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




def train_fun(model,optimizer,criterion,device,train_loader,val_loader):
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
        l1_reg = config['enc_l1'] * torch.norm(embeddings, p=1, dim=1).mean()
        loss += l1_reg
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
            l1_reg = config['enc_l1'] * torch.norm(embeddings, p=1, dim=1).sum()
            loss += l1_reg

            val_loss += loss.item() / len(val_loader)
            xis_embeddings.append(xis.cpu().detach().numpy())
            xjs_embeddings.append(xjs.cpu().detach().numpy())
            # metrics = calculator.get_accuracy(xis.cpu().detach().numpy(), xjs.cpu().detach().numpy(),
            #                                   np.arange(xis.shape[0]), np.arange(xis.shape[0]),
            #                                   False)
        xis_embeddings = np.concatenate(xis_embeddings)
        xjs_embeddings = np.concatenate(xjs_embeddings)
        accuracy = accuracy_fun(xis_embeddings, xjs_embeddings,distance=distance_fun)
    return val_loss,accuracy


class Trainer(tune.Trainable):
    def setup(self, config):
        self.model=get_model(config).to(device)
        self.optimizer=get_optimizer(config,self.model)
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
        loss,accuracy=train_fun(self.model,self.optimizer,self.criterion,
                            self.device,self.train_loader,self.val_loader)
        return {'loss':loss,'accuracy':accuracy}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


configs = {
    'dropout':tune.loguniform(0.0001,0.5),
    'enc_output_size':tune.choice([16,]),
    'representation_size':tune.choice([32,]),
    'batch_size':tune.choice([8,16,32,64,128]),
    'enc_temp':tune.loguniform(0.0001,0.5),
    'enc_lr':tune.loguniform(0.00001,0.01),
    'enc_l2':tune.loguniform(0.0000001,0.05),
    'enc_l1':tune.loguniform(0.000001,0.05),
    # 'aug_gaus':tune.choice([0,0.2,0.5,0.8,1.0]),
    'aug_num_seg':tune.choice([2,5,10,20,40,80]),
    # 'aug_prop_seg':tune.choice([0.05,0.1,0.3,0.5,0.9]),

}
config={i:v.sample() for i,v in configs.items()}

scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=700,
        grace_period=5,
        reduction_factor=2)
# scheduler=AsyncHyperBandScheduler(
#         time_attr="training_iteration",
#         metric="loss",
#         mode="min",
#         grace_period=5,
#         max_t=700)
# scheduler = HyperBandScheduler(metric="loss", mode="min")

if __name__=="__main__":

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        Trainer,
        # metric='loss',
        # mode='min',
        checkpoint_at_end=True,
        resources_per_trial={"cpu": 10, "gpu": 1},
        config=configs,
        local_dir=os.path.join(log_dir, "contrastive"),
        num_samples=700,
        name=experiment,
        # resume=False,
        scheduler=scheduler,
        progress_reporter=reporter,
        # sync_to_driver=False,
        raise_on_failed_trial=False)
    df = result.results_df
    df.to_csv(os.path.join(data_dir, "results/hypersearch.csv"), index=False)
    best_trial = result.get_best_trial("loss", "min", "last")
    best_config=result.get_best_config('loss','min')

best_model=get_model(best_config)
best_trial=result.get_best_trial('loss','min')
best_checkpoint=result.get_best_checkpoint(best_trial,'loss','min')
model_state=torch.load(best_checkpoint)


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

joblib.dump((classifier_embedding,test_embedding,train_full,test_full),
            os.path.join(data_dir,f"results/{experiment}.joblib"))



# best_trained_model=get_model(best_trial.config)
# best_checkpoint_dir = result.get_best_logdir("loss",mode="min")
# model_state, optimizer_state = torch.load(os.path.join(
#     best_checkpoint_dir, "checkpoint"))
# best_trained_model.load_state_dict(model_state)

# model=Classifier(in_features=2,hid_dim=64,z_dim=64)
# base_model=resnet1d(num_classes=enc_representation_size)
# # model=WaveNetModel(layers=6,blocks=6,dilation_channels=32,residual_channels=32,skip_channels=1024,
# #                         classes=enc_output_size,kernel_size=3,input_length=800)
# model=EncoderRaw(base_model,representation_size=enc_representation_size,
#                  dropout=dropout_,num_classes=enc_output_size)
# model.fc=MLP(enc_representation_size,num_classes=enc_output_size)
# model.fc = nn.Sequential(
#     nn.BatchNorm1d(model.out_features),
#     nn.ReLU(),
#     nn.Dropout(dropout_),
#     nn.Linear(model.out_features,enc_representation_size),
#     nn.BatchNorm1d(enc_representation_size),
#     nn.ReLU(),
#     nn.Dropout(dropout_),
#     nn.Linear(enc_representation_size,enc_output_size)
# )

# model.apply(init_fun)
#
# model=model.to(device)
# optimizer=torch.optim.Adam(params=model.parameters(),lr=enc_lr_ ,weight_decay=enc_l2_)
# # optimizer=torch.optim.SGD(params=model.parameters(),lr=enc_lr_ ,weight_decay=enc_l2_,momentum=0.99)
# # criterion=losses.TripletMarginLoss(margin=2.0).to(device)
# criterion=losses.NTXentLoss(temperature=enc_temp,
#                             distance=distances.LpDistance(normalize_embeddings=False),
#                             # embedding_regularizer=regularizers.LpRegularizer(p=1)
#                             ).to(device)
# calculator=AccuracyCalculator(k=1)
# epochs=700
# scheduler=ReduceLROnPlateau(optimizer,factor=0.5,patience=10,verbose=True,min_lr=1e-6)
# # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-1,epochs=epochs,steps_per_epoch=len(train_loader))
#
#
#
# epoch=0
# best_model={'loss':np.Inf,'params':None,'auc':0}
# early_stop_counter=0
# losses_=[]
# aucs=[]
# for epoch in range(epoch,epochs):
#     train_loss=0
#     test_loss=0
#
#     model.train()
#     print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch,optimizer.param_groups[0]['lr']))
#     for x1_raw,  x2_raw in tqdm(encoder_train_loader):
#         x1_raw = x1_raw.to(device, dtype=torch.float)
#         xis = model( x1_raw)
#         x2_raw = x2_raw.to(device, dtype=torch.float)
#         xjs = model( x2_raw)
#         xis = nn.functional.normalize(xis, dim=1)
#         xjs = nn.functional.normalize(xjs, dim=1)
#         embeddings=torch.cat([xis,xjs],dim=0)
#         labels=torch.cat([torch.arange(xis.shape[0]),]*2).to(device)
#         loss = criterion(embeddings,labels)
#         l1_reg=enc_l1_ * torch.norm(embeddings,p=1,dim=1).mean()
#         loss+=l1_reg
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() / len(encoder_train_loader)
#         # metrics=calculator.get_accuracy(xis.cpu().detach().numpy(),xjs.cpu().detach().numpy(),
#         #                         np.arange(xis.shape[0]),np.arange(xis.shape[0]),
#         #                         False)
#
#
#     model.eval()
#     test_loss=0
#     with torch.no_grad():
#         for x1_raw, x2_raw in tqdm(encoder_test_loader):
#             x1_raw= x1_raw.to(device, dtype=torch.float)
#             xis = model( x1_raw)
#             x2_raw= x2_raw.to(device, dtype=torch.float)
#             xjs = model( x2_raw)
#             xis = nn.functional.normalize(xis, dim=1)
#             xjs = nn.functional.normalize(xjs, dim=1)
#             embeddings = torch.cat([xis, xjs], dim=0)
#             labels = torch.cat([torch.arange(xis.shape[0]), ] * 2)
#             loss = criterion(embeddings, labels)
#             l1_reg = enc_l1_ * torch.norm(embeddings, p=1, dim=1).mean()
#             loss += l1_reg
#
#             test_loss += loss.item() / len(encoder_test_loader)
#             # metrics = calculator.get_accuracy(xis.cpu().detach().numpy(), xjs.cpu().detach().numpy(),
#             #                                   np.arange(xis.shape[0]), np.arange(xis.shape[0]),
#             #                                   False)
#
#
#     print("Epoch: %d: loss %.3f, val_loss %.3f" % (epoch, train_loss, test_loss))
#     losses_.append((train_loss, test_loss))
#     scheduler.step(test_loss)
#     if test_loss < best_model['loss']:
#         best_model['loss']=test_loss
#         best_model['params']=copy.deepcopy(model.state_dict())
#         best_model['epoch']=epoch
#         early_stop_counter=0
#     else:
#         early_stop_counter+=1
#         if early_stop_counter>=70:
#             print("Early stopping ...")
#             break



# torch.save(best_model['params'],weights_file)
#
# fig,ax=plt.subplots(1)
# ax.plot([train for train, test in losses_],label="train")
# ax.plot([test for train, test in losses_],label="test")
# ax.set_ylim(0,3)
# plt.legend()
# plt.savefig("/home/pmwaniki/Dropbox/tmp/simclr_%s__lr%.5f_l2%.5f__size%d.png" % (os.uname()[1],enc_lr_,enc_l2_,enc_representation_size))
# if display:
#     plt.show()
# else:
#     plt.close()
#
#
# model.load_state_dict(best_model['params'])
#
# best_trained_model.fc=nn.Identity()
# best_trained_model.eval()
#
# classifier_embedding=[]
# with torch.no_grad():
#     for x1_raw in tqdm(classifier_train_loader):
#         x1_raw = x1_raw.to(device, dtype=torch.float)
#         xis = best_trained_model( x1_raw)
#         xis=nn.functional.normalize(xis,dim=1)
#         classifier_embedding.append(xis.cpu().detach().numpy())
#
# test_embedding=[]
# with torch.no_grad():
#     for x1_raw in tqdm(classifier_test_loader):
#         x1_raw = x1_raw.to(device, dtype=torch.float)
#         xis = best_trained_model( x1_raw)
#         xis = nn.functional.normalize(xis, dim=1)
#         test_embedding.append(xis.cpu().detach().numpy())
#
# classifier_embedding=np.concatenate(classifier_embedding)
# test_embedding=np.concatenate(test_embedding)
# test_identifier=test['id']
# subject_embeddings=[]
# subject_ids=[]
# subject_admitted=[]
# for id in test_identifier.unique():
#     temp_=test_embedding[test_identifier==id]
#     subject_embeddings.append(temp_.mean(axis=0))
#     subject_ids.append(id)
#     subject_admitted.append(test.loc[test['id']==id,'admitted'].iloc[0])
#
#
# subject_embeddings=np.stack(subject_embeddings)
# scl=StandardScaler()
# subject_scl=scl.fit_transform(subject_embeddings)
# pca=PCA(n_components=6)
# subject_pca=pca.fit_transform(subject_scl)
# subject_admitted=np.array(subject_admitted)
#
#
#
#
# fig,axs=plt.subplots(3,5,figsize=(15,10))
# for ax,vals in zip(axs.flatten(),itertools.combinations(range(6),2)):
#     r,c=vals
#     ax.scatter(subject_pca[subject_admitted == 0, r], subject_pca[subject_admitted == 0, c],
#                                           marker="o", label="No",
#                                           alpha=0.5)
#     ax.scatter(subject_pca[subject_admitted == 1, r], subject_pca[subject_admitted == 1, c],
#                                           marker="o", label="Yes",
#                                           alpha=0.5)
#     ax.set_xlabel(f"PCA {r + 1}")
#     ax.set_ylabel(f"PCA {c + 1}")
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# plt.savefig(f"/home/pmwaniki/Dropbox/tmp/triplet_embedding_lr{enc_lr_}_l2{enc_l2_}_size{enc_representation_size}.png")
# if display:
#     plt.show(block=False)
# else:
#     plt.close()
#
#
# scl=StandardScaler()
# scl_classifier_embedding=scl.fit_transform(classifier_embedding)
# scl_test_embedding=scl.transform(test_embedding)

# pca=PCA(n_components=6)
# train_pca=pca.fit_transform(scl_classifier_embedding)
# test_pca=pca.transform(scl_test_embedding)




# base_clf=SVC(probability=True,class_weight='balanced')
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-2,1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# base_clf=LogisticRegression(class_weight='balanced',max_iter=10000,penalty='l2')
# base_clf=Pipeline([
#     ('scl',StandardScaler()),
#     ('pca',PCA()),
#     ('clf',LogisticRegression(class_weight='balanced',max_iter=100000,penalty='elasticnet',solver='saga')),
# ])
# tuned_parameters = { 'clf__C': [1e-3,1e-2,1e-1,1,],
#                      'pca__n_components':[int(enc_representation_size/2**i) for i in range(5)],
#                      'clf__l1_ratio':[0.2,0.5,0.8]}
#
# clf=GridSearchCV(base_clf,param_grid=tuned_parameters,cv=StratifiedKFold(10,),
#                  verbose=1,n_jobs=10,
#                  scoring=['f1','roc_auc','recall','precision'],refit='roc_auc')
# clf.fit(classifier_embedding,train['admitted'])
#
# cv_results=pd.DataFrame({'params':clf.cv_results_['params'], 'auc':clf.cv_results_['mean_test_roc_auc'],
#               'f1':clf.cv_results_['mean_test_f1'],'recall':clf.cv_results_['mean_test_recall'],
#                           'precision':clf.cv_results_['mean_test_precision']})
# print(cv_results)
#
# test_pred=clf.predict_proba(test_embedding)[:,1]
#
# print(classification_report(test['admitted'],test_pred>0.5))
# print("AUC: ",roc_auc_score(test['admitted'],test_pred))
#
# final_predictions=pd.DataFrame({'admitted':test['admitted'],
#                                  'id':test['id'],
#                                  'prediction':test_pred})
# final_predictions2=final_predictions.groupby('id').agg('mean')
# print(classification_report(final_predictions2['admitted'],(final_predictions2['prediction']>0.5)*1.0))
#
# print("AUC: %.2f" % roc_auc_score(final_predictions2['admitted'],final_predictions2['prediction']))
#
#
# report=classification_report(final_predictions2['admitted'],(final_predictions2['prediction']>0.5)*1.0,output_dict=True)
# recall=report['1.0']['recall']
# precision=report['1.0']['precision']
# f1=report['1.0']['f1-score']
# specificity=report['0.0']['recall']
# acc=report['accuracy']
# auc=roc_auc_score(final_predictions2['admitted'],final_predictions2['prediction'])
#
#
#
# save_table3(model="Triplet",precision=precision,recall=recall,specificity=specificity,
#             auc=auc,details=details_,other=json.dumps({'host':os.uname()[1],'f1':f1,
#                                                        'acc':acc,'batch_size':enc_batch_size}))
#
# joblib.dump(clf,weights_file.replace(".pt",".joblib"))

# def train_fun(config,checkpoint_dir=None):
#     encoder_train_dataset=TriagePairs(train_encoder, id_var="id", stft_fun=None,
#                             transforms=None,
#                             # aug_raw=aug_raw,normalize=True
#                             )
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     base_model = resnet1d(num_classes=512)
#
#     model = EncoderRaw(base_model, representation_size=config['representation_size'],
#                        dropout=config['dropout'], num_classes=config['enc_output_size'])
#
#     test_abs = int(len(encoder_train_dataset) * 0.9)
#     train_ds,val_ds=random_split(encoder_train_dataset,lengths=[test_abs,len(encoder_train_dataset)-test_abs])
#
#     train_loader=DataLoader(train_ds,
#                             batch_size=int(config["batch_size"]),
#                             shuffle=True, num_workers=50)
#     val_loader=DataLoader(val_ds,
#                             batch_size=int(config["batch_size"]),
#                             shuffle=False, num_workers=50)
#     model = model.to(device)
#     optimizer = torch.optim.Adam(params=model.parameters(), lr=config['enc_lr'], weight_decay=config['enc_l2'])
#     if checkpoint_dir:
#         model_state, optimizer_state = torch.load(
#             os.path.join(checkpoint_dir, "checkpoint.pth"))
#         model.load_state_dict(model_state)
#         optimizer.load_state_dict(optimizer_state)
#     # optimizer=torch.optim.SGD(params=model.parameters(),lr=enc_lr_ ,weight_decay=enc_l2_,momentum=0.99)
#     # criterion=losses.TripletMarginLoss(margin=2.0).to(device)
#     criterion = losses.NTXentLoss(temperature=config['enc_temp'],
#                                   # distance=distances.LpDistance(normalize_embeddings=False),
#                                   distance=distances.DotProductSimilarity(normalize_embeddings=False),
#                                   ).to(device)
#     # calculator = AccuracyCalculator(k=1)
#     epochs = 600
#     scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True, min_lr=1e-6)
#
#     for epoch in range(0, epochs):
#         train_loss = 0
#
#         model.train()
#         # print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch, optimizer.param_groups[0]['lr']))
#         for x1_raw, x2_raw in train_loader:
#             x1_raw = x1_raw.to(device, dtype=torch.float)
#             xis = model(x1_raw)
#             x2_raw = x2_raw.to(device, dtype=torch.float)
#             xjs = model(x2_raw)
#             xis = nn.functional.normalize(xis, dim=1)
#             xjs = nn.functional.normalize(xjs, dim=1)
#             embeddings = torch.cat([xis, xjs], dim=0)
#             labels = torch.cat([torch.arange(xis.shape[0]), ] * 2).to(device)
#             loss = criterion(embeddings, labels)
#             l1_reg = config['enc_l1'] * torch.norm(embeddings, p=1, dim=1).mean()
#             loss += l1_reg
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() / len(train_loader)
#             # metrics=calculator.get_accuracy(xis.cpu().detach().numpy(),xjs.cpu().detach().numpy(),
#             #                         np.arange(xis.shape[0]),np.arange(xis.shape[0]),
#             #                         False)
#
#         model.eval()
#         val_loss = 0
#         xis_embeddings=[]
#         xjs_embeddings=[]
#         with torch.no_grad():
#             for x1_raw, x2_raw in val_loader:
#                 x1_raw = x1_raw.to(device, dtype=torch.float)
#                 xis = model(x1_raw)
#                 x2_raw = x2_raw.to(device, dtype=torch.float)
#                 xjs = model(x2_raw)
#                 xis = nn.functional.normalize(xis, dim=1)
#                 xjs = nn.functional.normalize(xjs, dim=1)
#                 embeddings = torch.cat([xis, xjs], dim=0)
#                 labels = torch.cat([torch.arange(xis.shape[0]), ] * 2)
#                 loss = criterion(embeddings, labels)
#                 l1_reg = config['enc_l1'] * torch.norm(embeddings, p=1, dim=1).mean()
#                 loss += l1_reg
#
#                 val_loss += loss.item() / len(val_loader)
#                 xis_embeddings.append(xis.cpu().detach().numpy())
#                 xjs_embeddings.append(xjs.cpu().detach().numpy())
#                 # metrics = calculator.get_accuracy(xis.cpu().detach().numpy(), xjs.cpu().detach().numpy(),
#                 #                                   np.arange(xis.shape[0]), np.arange(xis.shape[0]),
#                 #                                   False)
#             xis_embeddings=np.concatenate(xis_embeddings)
#             xjs_embeddings=np.concatenate(xjs_embeddings)
#             accuracy=test(xis_embeddings,xjs_embeddings)
#
#         # scheduler.step(val_loss)
#
#         with tune.checkpoint_dir(99) as checkpoint_dir:
#             path = os.path.join(checkpoint_dir, "checkpoint.pth")
#             torch.save((model.state_dict(), optimizer.state_dict()), path)
#         tune.report(loss=val_loss, accuracy=accuracy)
#
#     print("Finished Training")