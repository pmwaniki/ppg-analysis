import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler,random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from torchsummary import summary
from settings import data_dir
from settings import checkpoint_dir as log_dir
import os
from models.cnn.networks import resnet50_1d,EncoderRaw,wideresnet50_1d
from datasets.signals import gaus_noise,permute

from functools import partial

from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine
from datasets.loaders import TriageDataset,TriagePairs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from settings import weights_dir
from pytorch_metric_learning import distances,regularizers,losses,testers

import ray
# ray.init( num_cpus=12,dashboard_host="0.0.0.0")
ray.init(address="auto")
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler,PopulationBasedTraining,HyperBandScheduler,AsyncHyperBandScheduler

import joblib


display=os.environ.get('DISPLAY',None) is not None



include_sepsis=True
enc_representation_size="32"
res_type="original" # wide,original
enc_distance="DotProduct" #LpDistance Dotproduct Cosine
distance_fun="euclidean" if enc_distance=="LpDistance" else cosine
pretext="sample" #sample, augment
experiment=f"Contrastive-{res_type}-{pretext}-{enc_distance}{enc_representation_size}"

if include_sepsis: experiment = experiment + "-sepsis"
weights_file=os.path.join(weights_dir,f"Contrastive_{experiment}.pt")


#
if include_sepsis:
    sepsis_segments1=pd.read_csv(os.path.join(data_dir,"segments-sepsis.csv"))
    sepsis_segments2 = pd.read_csv(os.path.join(data_dir, "segments-sepsis_0m.csv"))
    sepsis_segments= pd.concat([sepsis_segments1,sepsis_segments2])

    sepsis_segments['id']=sepsis_segments['id'] + "-" + sepsis_segments['episode']
    non_repeated_ids = [k for k, v in sepsis_segments['id'].value_counts().items() if v <= 15]
    sepsis_segments = sepsis_segments[~sepsis_segments['id'].isin(non_repeated_ids)]


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
    if include_sepsis:
        train_csv=pd.concat([train_csv,sepsis_segments],axis=0)
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
                              shuffle=True, num_workers=10)
    val_loader = DataLoader(val_ds,
                            batch_size=int(config["batch_size"]),
                            shuffle=False, num_workers=10)
    return train_loader,val_loader

def get_model(config):
    if res_type == "original":
        base_model = resnet50_1d(num_classes=32)
    elif res_type == "wide":
        base_model = wideresnet50_1d(num_classes=32)

    model = EncoderRaw(base_model, representation_size=config['representation_size'],
                       dropout=config['dropout'], num_classes=config['enc_output_size'])
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    return model

def get_optimizer(config,model):
    optimizer=torch.optim.Adam(params=[
        {'params':model.base_model.parameters()},
        {'params':model.fc0.parameters(),'lr':config['lr_fc'],'weight_decay':config['l2_fc']},
        {'params':model.fc.parameters(),'lr':config['lr_fc'],'weight_decay':config['l2_fc']}
    ],lr=config['lr'],weight_decay=config['l2'])
    # optimizer = torch.optim.Adam(params=model.parameters(),
    #                              lr=config['enc_lr'],
    #                              weight_decay=config['enc_l2'])
    return optimizer


device = "cuda" if torch.cuda.is_available() else "cpu"




def train_fun(model,optimizer,criterion,device,train_loader,val_loader,scheduler=None):
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
        if scheduler is not None: scheduler.step()
        xis_embeddings = np.concatenate(xis_embeddings)
        xjs_embeddings = np.concatenate(xjs_embeddings)
        accuracy = accuracy_fun(xis_embeddings, xjs_embeddings,distance=distance_fun)
    return train_loss,val_loss,accuracy


class Trainer(tune.Trainable):
    def setup(self, config):
        self.model=get_model(config).to(device)
        self.optimizer=get_optimizer(config,self.model)
        self.scheduler=StepLR(self.optimizer,step_size=200,gamma=.5)
        if enc_distance == "LpDistance":
            dist_fun=distances.DotProductSimilarity(normalize_embeddings=False)
        elif enc_distance =="DotProduct":
            dist_fun=distances.DotProductSimilarity(normalize_embeddings=False)
        elif enc_distance == "Cosine":
            dist_fun=distances.CosineSimilarity()
        else:
            raise NotImplementedError(f"{enc_distance} not implemented yet!!!")
        self.dist_fun=dist_fun
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
    'batch_size':tune.choice([8,16,32,64,]),
    'enc_temp':tune.loguniform(0.0001,0.5),
    'lr':tune.loguniform(0.00001,0.5),
    'l2':tune.loguniform(0.000001,1.0),
    'lr_fc':tune.loguniform(0.00001,0.5),
    'l2_fc':tune.loguniform(0.000001,1.0),
    # 'enc_l1':tune.loguniform(0.000001,0.05),
    'aug_gaus':tune.choice([0.0,0.2,0.5,0.8,1.0]) if pretext == "sample" else tune.choice([1.0,]),
    'aug_num_seg':tune.choice([2,5,8,10]),
    'aug_prop_seg':tune.choice([0.0,0.2,0.5,0.8,1.0]) if pretext == "sample" else tune.choice([1.0,]),

}

# config={i:v.sample() for i,v in configs.items()}

scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=700,
        grace_period=20,
        reduction_factor=2)

# scheduler = PopulationBasedTraining(
#         time_attr="training_iteration",
#         metric="loss",
#         mode="min",
#         perturbation_interval=10,
#         hyperparam_mutations={
#             # distribution for resampling
#             "enc_lr": lambda: np.random.uniform(1e-1, 1e-5),
#             "enc_l2": lambda: np.random.uniform(1e-1, 1e-5),
#         })

# scheduler=AsyncHyperBandScheduler(
#         time_attr="training_iteration",
#         metric="loss",
#         mode="min",
#         grace_period=5,
#         max_t=700)
# scheduler = HyperBandScheduler(metric="loss", mode="min")



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
    local_dir=log_dir,
    num_samples=500,
    name=experiment,
    resume=True,
    scheduler=scheduler,
    progress_reporter=reporter,
    reuse_actors=False,
    raise_on_failed_trial=False,)

df = result.results_df
df.to_csv(os.path.join(data_dir, f"results/hypersearch-{experiment}.csv"), index=False)
metric='accuracy';mode='max'
best_trial = result.get_best_trial(metric, mode, "last-5-avg")
best_config=result.get_best_config(metric,mode,scope="last-5-avg")

best_model=get_model(best_config)
# best_trial=result.get_best_trial('loss','min')
best_checkpoint=result.get_best_checkpoint(best_trial,metric=metric,mode=mode)
model_state,optimizer_state=torch.load(best_checkpoint)
torch.save(model_state,weights_file)

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
    print(f"Accuracy: {accuracy}")

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



