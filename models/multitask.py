import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from settings import data_dir
import os
from models.cnn.networks import resnet34,groupNorm
from datasets.signals import stft
from sklearn.metrics import roc_auc_score,classification_report
from functools import partial

from datasets.loaders import TriageDataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
display=os.environ.get('DISPLAY',None) is not None


fs=128;nperseg=1024;noverlap=1000




triage_segments=pd.read_csv(os.path.join(data_dir,'triage/segments.csv'))

np.random.seed(123)
train_ids=np.random.choice(triage_segments['id'].unique(),
                           size=int(0.8*len(triage_segments['id'].unique())),
                           replace=False)
train=triage_segments[triage_segments['id'].isin(train_ids)]
test=triage_segments[~triage_segments['id'].isin(train_ids)]


sfft_fun=partial(stft,fs=fs,nperseg=nperseg,noverlap=noverlap)
train_dataset=TriageDataset(train,labels="admitted",stft_fun=sfft_fun)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=50)

test_dataset=TriageDataset(test,labels="admitted",stft_fun=sfft_fun)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=50)


iter_=iter(train_loader)
batch_x,batch_y=iter_.next()

if display:
    fig, axs = plt.subplots(2, 3, figsize=(12, 3))
    for r in range(2):
        for c in range(3):
            axs[r, c].imshow(batch_x[c, r, :, :], vmin=batch_x.numpy()[:, r, :, :].min(),
                             vmax=batch_x.numpy()[:, r, :, :].max(),
                             norm=colors.LogNorm()
                             )
            axs[r, c].invert_yaxis()

    plt.show()



device="cuda" if torch.cuda.is_available() else "cpu"
# model=Classifier(in_features=2,hid_dim=64,z_dim=64)
model=resnet34(num_classes=1,norm_layer=groupNorm(8))
model=model.to(device)
optimizer=torch.optim.AdamW(params=model.parameters(),lr=0.001 ,weight_decay=1e-3)
criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0)).to(device)
epochs=250
scheduler=ReduceLROnPlateau(optimizer,factor=0.5,patience=5,verbose=True,min_lr=1e-6)

epoch=0


losses=[]
for epoch in range(epoch,epochs):
    train_loss=0
    test_loss=0
    test_pred=np.array([])

    model.train()
    print("epoch: %d >> learning rate at begining of epoch: %.5f" % (epoch,optimizer.param_groups[0]['lr']))
    for batch_x,batch_y in tqdm(train_loader):
        batch_x,batch_y=batch_x.to(device,dtype=torch.float),batch_y.to(device)
        logits=model(batch_x)
        loss=criterion(logits,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()/len(train_loader)


    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader):
            batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            test_loss += loss.item() / len(test_loader)
            test_pred=np.concatenate([test_pred,logits.sigmoid().squeeze().cpu().numpy()])

    print("Epoch: %d: loss %.3f, val_loss %.3f, auc %.3f" % (epoch,train_loss,test_loss,
                                                             roc_auc_score(test['admitted'],test_pred)))
    losses.append((train_loss,test_loss))
    scheduler.step(test_loss)


if display:
    plt.plot(losses)
    plt.show()

print(classification_report(test['admitted'],(test_pred>0.5)*1.0))