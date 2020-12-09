import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from settings import data_dir
import os
from models.cnn.networks import EncoderRaw,MLP
from models.cnn.wavenet2 import WaveNetModel
from datasets.signals import gaus_noise,permute
from sklearn.metrics import roc_auc_score,classification_report
from functools import partial

from datasets.loaders import TriageDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib,copy
import json
from utils import save_table3


display=os.environ.get('DISPLAY',None) is not None

initialize='random'
batch_size=32
l2_=0.001
l2b_=0.001
lr_=0.001
dropout_=0.1
enc_representation_size=128
size_dense=128

details_=f"{initialize}_lr{lr_}_l2{l2_}_drop{dropout_}_bs{batch_size}"

enc_weights_file=None



triage_segments=pd.read_csv(os.path.join(data_dir,'triage/segments.csv'))

train_ids,test_ids=joblib.load(os.path.join(data_dir,"triage/ids.joblib"))


train=triage_segments[triage_segments['id'].isin(train_ids)]
test=triage_segments[triage_segments['id'].isin(test_ids)]

aug_raw=[
partial(gaus_noise,min_sd=1e-5,max_sd=1e-2),
partial(permute,n_segments=10,p=5.0),
]

#sample weights
class_sample_count = np.array(
    [len(np.where(train['admitted'] == t)[0]) for t in np.unique(train['admitted'])])
weight = 1. / class_sample_count
samples_weight = np.array([weight[int(t)] for t in train['admitted']])

samples_weight = torch.from_numpy(samples_weight)
samples_weigth = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))







train_dataset=TriageDataset(train,labels="admitted",stft_fun=None,
                            transforms=None,
                            aug_raw=aug_raw,
                            normalize=True)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=False,sampler=sampler,num_workers=50)

test_dataset=TriageDataset(test,labels="admitted",stft_fun=None,aug_raw=[],normalize=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=50)


iter_=iter(train_loader)
batch_xraw,batch_y=iter_.next()
print("Input shapes:", batch_xraw.shape)

if display:
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    for r in range(2):
        for c in range(3):
            axs[r, c].plot(batch_xraw.numpy()[c, r, :],
                             # vmin=batch_x.numpy()[:, r, :, :].min(),
                             # vmax=batch_x.numpy()[:, r, :, :].max(),
                             # norm=colors.LogNorm()
                             )

    plt.show()



device="cuda" if torch.cuda.is_available() else "cpu"

# model=Net(num_classes=1,n_convs=10)

classifier=MLP(enc_representation_size,num_classes=1)
# base_model=WaveNet(num_samples=800,num_channels=2,num_classes=size_dense,num_blocks=2,num_layers=7,dropout=dropout_)
# base_model=resnet1d(num_classes=1)
base_model=WaveNetModel(layers=6,blocks=6,dilation_channels=32,residual_channels=32,skip_channels=1024,
                        classes=1,kernel_size=3,input_length=800)
# base_model=WaveNet(num_samples=800,num_channels=2,num_classes=1,num_blocks=6,num_layers=7,skip_channels=32,res_channels=32,
#                    kernel_size=2,
#                    dropout=dropout_,bias=False)
model=EncoderRaw(base_model=base_model,representation_size=enc_representation_size,num_classes=1,dropout=dropout_)
# model=EncoderRaw(base_model=base_model,representation_size=size_dense,fc_layer=classifier)
# model.apply(init_fun)
model=model.to(device)
# optimizer=torch.optim.Adam(params=model.parameters(),lr=lr_ ,weight_decay=l2_)
optimizer=torch.optim.Adam([
    {'params':model.base_model.parameters()},
    {'params':model.bn0.parameters()},
    {'params':model.fc0.parameters(),'weight_decay':l2b_},
    {'params':model.fc.parameters()}
],lr=lr_,weight_decay=l2_)
criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5)).to(device)
epochs=400
scheduler=ReduceLROnPlateau(optimizer,factor=0.5,patience=5,verbose=True,min_lr=1e-6)

epoch=0


best_model={'loss':np.Inf,'params':None,'auc':0,'epoch':-1}
early_stop_counter=0
losses=[]
aucs=[]
for epoch in range(epoch,epochs):
    train_loss=0


    model.train()
    print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch,optimizer.param_groups[0]['lr']))
    for batch_xraw,batch_y in tqdm(train_loader):
        batch_xraw,batch_y=batch_xraw.to(device,dtype=torch.float),batch_y.to(device)
        logits=model(batch_xraw)
        loss=criterion(logits,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()/len(train_loader)


    model.eval()
    test_pred = np.array([])
    test_loss=0
    with torch.no_grad():
        for batch_xraw, batch_y in tqdm(test_loader):
            batch_xraw, batch_y = batch_xraw.to(device,dtype=torch.float), batch_y.to(device)
            logits = model(batch_xraw)
            loss = criterion(logits, batch_y)
            test_loss += loss.item() / len(test_loader)
            test_pred=np.concatenate([test_pred,logits.sigmoid().squeeze().cpu().numpy().reshape(-1)])

    print("Epoch: %d: loss %.3f, val_loss %.3f, auc %.3f" % (epoch,train_loss,test_loss,
                                                             roc_auc_score(test['admitted'],test_pred)))
    losses.append((train_loss,test_loss))
    scheduler.step(test_loss)
    test_auc = roc_auc_score(test['admitted'], test_pred)
    aucs.append(test_auc)
    if epoch>0:
        if test_loss < best_model['loss']:
            best_model['loss']=test_loss
            best_model['params']=copy.deepcopy(model.state_dict())
            best_model['auc']=test_auc
            best_model['epoch']=epoch
            early_stop_counter=0
        else:
            early_stop_counter+=1
            if early_stop_counter>=40:
                print("Early stopping ...")
                break

fig,ax=plt.subplots(1)
ax.plot([train for train, test in losses],label="train")
ax.plot([test for train, test in losses],label="test")
plt.legend()
plt.savefig(f"/home/pmwaniki/Dropbox/tmp/ppg_{os.uname()[1]}__lr{lr_}_l2{l2_}_drop{dropout_}.png" )
if display:
    plt.show()
else:
    plt.close()



# print(classification_report(test['admitted'],(test_pred>0.5)*1.0))
if best_model['epoch']>20:
    model.load_state_dict(best_model['params'])

model.eval()
final_predictions=np.array([])
with torch.no_grad():
    for batch_xraw, batch_y in tqdm(test_loader):
        batch_xraw, batch_y = batch_xraw.to(device,dtype=torch.float),  batch_y.to(device)
        logits = model(batch_xraw)
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