import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio
from settings import data_dir
import os
import json
from models.cnn.networks import resnet1d,Encoder,wideresnet50,MLP
from datasets.signals import stft,gaus_noise,rand_sfft
from sklearn.metrics import roc_auc_score,classification_report
from functools import partial

from datasets.loaders import TriageDataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from settings import Fs,weights_dir
import joblib
import copy
from utils import save_table3

display=os.environ.get('DISPLAY',None) is not None


initialize="random" # simclr random
enc_lr_=1.0
enc_l2_=5e-4
enc_representation_size=64
enc_output_size=64
enc_temp=0.05
enc_batch_size=64
# weights_file=os.path.join(weights_dir,f"simclr_lr{enc_lr_}_l2{enc_l2_}_z{enc_representation_size}_x{enc_output_size}_temp{enc_temp}_bs{enc_batch_size}.pt")
enc_weights_file=os.path.join(weights_dir,f"triplet_lr{enc_lr_}_temp{enc_temp}_l2{enc_l2_}_z{enc_representation_size}_x{enc_output_size}_bs{enc_batch_size}.pt")
fs=Fs

slice_sec=2.5
slide_sec=0.2

dropout_=0.2
l2_=5e-3
lr_=0.01
conv_lr_factor=1.0
pos_weight=1.5
representation_size=enc_representation_size
batch_size=128

nperseg=int(slice_sec*fs)
step=int(slide_sec*fs)
noverlap=nperseg-step

details_=f"z{enc_representation_size}_l2{l2_}_lr_factor{conv_lr_factor}_drop{dropout_}_lr{lr_}_weight{pos_weight}"


data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
triage_segments=pd.read_csv(os.path.join(data_dir,'triage/segments.csv'))

train_ids,test_ids=joblib.load(os.path.join(data_dir,"triage/ids.joblib"))


train=triage_segments[triage_segments['id'].isin(train_ids)]
test=triage_segments[triage_segments['id'].isin(test_ids)]


sfft_fun=partial(stft,fs=fs,nperseg=nperseg,noverlap=noverlap,spec_only=True)
rand_stft_fun=partial(rand_sfft,fs=fs,output_shape=(13,39))

#sample weights
class_sample_count = np.array(
    [len(np.where(train['admitted'] == t)[0]) for t in np.unique(train['admitted'])])
weight = 1. / class_sample_count
samples_weight = np.array([weight[int(t)] for t in train['admitted']])

samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))






aug_gausian=partial(gaus_noise,min_sd=1e-5,max_sd=1e-1)

train_transformation=nn.Sequential(
    torchaudio.transforms.TimeMasking(time_mask_param=5),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=3),
)

train_dataset=TriageDataset(train,labels="admitted",stft_fun=sfft_fun,
                            transforms=train_transformation,aug_raw=aug_gausian)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=50,
                        sampler=sampler
                        )

test_dataset=TriageDataset(test,labels="admitted",stft_fun=sfft_fun,aug_raw=None)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=15)


iter_=iter(train_loader)
batch_xraw,batch_x,batch_y=iter_.next()
print("Input shapes:", batch_xraw.shape, batch_x.shape)

if display:
    fig, axs = plt.subplots(4, 3, figsize=(12, 5))
    for r in range(2):
        for c in range(3):
            axs[r*2+1, c].imshow(batch_x.numpy()[c, r, :, :],
                             # vmin=batch_x.numpy()[:, r, :, :].min(),
                             # vmax=batch_x.numpy()[:, r, :, :].max(),
                             norm=colors.LogNorm()
                             )
            axs[r*2+1, c].invert_yaxis()
    for r in range(2):
        for c in range(3):
            axs[r*2, c].plot(batch_xraw.numpy()[c, r, :],
                             # vmin=batch_x.numpy()[:, r, :, :].min(),
                             # vmax=batch_x.numpy()[:, r, :, :].max(),
                             # norm=colors.LogNorm()
                             )

    plt.show()



device="cuda" if torch.cuda.is_available() else "cpu"
# model=Classifier(in_features=2,hid_dim=64,z_dim=64)

model_raw=resnet1d(num_classes=enc_representation_size)
model_stft=wideresnet50(num_classes=enc_representation_size)
enc_classifier=nn.Sequential(nn.ReLU(),nn.Linear(enc_representation_size,enc_output_size))
classifier=MLP(representation_size=enc_representation_size,dropout=dropout_,num_classes=1)
model=Encoder(raw_model=model_raw,stft_model=model_stft,representation_size=enc_representation_size,
              fc_layer=enc_classifier)


if initialize is not "random":
    enc_weights = torch.load(enc_weights_file)
    model.load_state_dict(enc_weights)
model.fc=classifier
model=model.to(device)
# optimizer=torch.optim.Adam(params=model.parameters(),lr=lr_ ,weight_decay=l2_,betas=(0.99,0.999))
# optimizer=torch.optim.SGD(params=model.parameters(),lr=lr_ ,weight_decay=l2_,momentum=0.99)
optimizer=torch.optim.SGD([
    {'params': model.stft_model.parameters(),'lr':lr_*conv_lr_factor},
    {'params': model.raw_model.parameters(), 'lr': lr_*conv_lr_factor},
    {'params':model.fc.parameters(),'lr':lr_},
            ],lr=lr_, momentum=0.99,weight_decay=l2_)

criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)).to(device)
# criterion=nn.BCEWithLogitsLoss().to(device)
epochs=300
scheduler=ReduceLROnPlateau(optimizer,factor=0.5,patience=10,verbose=True,min_lr=1e-5)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-1,epochs=epochs,steps_per_epoch=len(train_loader))

epoch=0
best_model={'loss':np.Inf,'params':None,'auc':0}
early_stop_counter=0
losses=[]
aucs=[]
for epoch in range(epoch,epochs):
    train_loss=0
    test_loss=0
    test_pred=np.array([])

    model.train()
    print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch,optimizer.param_groups[0]['lr']))
    for batch_xraw,batch_x,batch_y in tqdm(train_loader):
        batch_xraw,batch_x,batch_y=batch_xraw.to(device,dtype=torch.float),batch_x.to(device,dtype=torch.float),batch_y.to(device)
        logits=model(batch_xraw,batch_x)
        loss=criterion(logits,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()/len(train_loader)
        # scheduler.step()

    model.eval()
    with torch.no_grad():
        for batch_xraw,batch_x, batch_y in tqdm(test_loader):
            batch_xraw,batch_x, batch_y = batch_xraw.to(device,dtype=torch.float), batch_x.to(device, dtype=torch.float), batch_y.to(device)
            logits = model(batch_xraw,batch_x)
            loss = criterion(logits, batch_y)
            test_loss += loss.item() / len(test_loader)
            test_pred=np.concatenate([test_pred,logits.sigmoid().squeeze().cpu().numpy().reshape(-1)])
    test_auc=roc_auc_score(test['admitted'],test_pred)
    aucs.append(test_auc)
    print("Epoch: %d: loss %.3f, val_loss %.3f, auc %.3f" % (epoch, train_loss, test_loss,
                                                             test_auc))
    losses.append((train_loss, test_loss))
    scheduler.step(test_loss)
    if (test_loss < best_model['loss']) & (epoch>50):
        best_model['loss']=test_loss
        best_model['params']=copy.deepcopy(model.state_dict())
        best_model['auc']=test_auc
        best_model['epoch']=epoch
        early_stop_counter=0
    else:
        early_stop_counter+=1
        if (early_stop_counter>=50) & (epoch>70):
            print("Early stopping ...")
            break



torch.save({'best':best_model['params'],
            'last':model.state_dict(),
            'epoch':epoch,'best_epoch':best_model['epoch']},
           os.path.join(data_dir,f"results/weights/ppg_{initialize}_{details_}.pt"))

fig,ax=plt.subplots(1)
ax.plot([train for train, test in losses],label="train")
ax.plot([test for train, test in losses],label="test")
plt.legend()
plt.savefig(f"/home/pmwaniki/Dropbox/tmp/ppg_{initialize}_{os.uname()[1]}__lr{lr_}_l2{l2_}_drop{dropout_}.png" )
if display:
    plt.show()
else:
    plt.close()



# evaluate
if best_model['epoch']>50:
    model.load_state_dict(best_model['params'])

model.eval()
final_predictions=np.array([])
with torch.no_grad():
    for batch_xraw,batch_x, batch_y in tqdm(test_loader):
        batch_xraw,batch_x, batch_y = batch_xraw.to(device,dtype=torch.float), batch_x.to(device, dtype=torch.float), batch_y.to(device)
        logits = model(batch_xraw,batch_x)
        # test_loss += loss.item() / len(test_loader)
        final_predictions=np.concatenate([final_predictions,logits.sigmoid().squeeze().cpu().numpy().reshape(-1)])
final_predictions2=pd.DataFrame({'admitted':test['admitted'],
                                 'id':test['id'],
                                 'prediction':final_predictions})
final_predictions2=final_predictions2.groupby('id').agg('mean')
report=classification_report(final_predictions2['admitted'],(final_predictions2['prediction']>0.5)*1.0,output_dict=True)
recall=report['1.0']['recall']
precision=report['1.0']['precision']
f1=report['1.0']['f1-score']
specificity=report['0.0']['recall']
acc=report['accuracy']
auc=roc_auc_score(final_predictions2['admitted'],final_predictions2['prediction'])

print(classification_report(final_predictions2['admitted'],(final_predictions2['prediction']>0.5)*1.0))

print("AUC: %.2f" % auc)

save_table3(model=f"Supervised-{initialize}",precision=precision,recall=recall,specificity=specificity,
            auc=auc,details=details_,other=json.dumps({'host':os.uname()[1],'f1':f1,
                                                       'acc':acc,'batch_size':batch_size,
                                                       'encoder':None if initialize=="random" else enc_weights_file}))

# fpr, tpr, thresholds=roc_curve(final_predictions2['admitted'],final_predictions2['prediction'])
# plt.plot(fpr,tpr)
# plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()