import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio
from settings import data_dir
import os
from models.cnn.networks import groupNorm,resnet1d,Encoder,MLP,wideresnet50
from models.selfsupervised.losses import NTXentLoss
from datasets.signals import stft, gaus_noise
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from functools import partial

from datasets.loaders import TriagePairs,TriageDataset
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from settings import Fs
import joblib,json
import copy
from settings import weights_dir
from utils import save_table3

display = os.environ.get('DISPLAY', None) is not None

lr_=0.001
l2_=5e-5
representation_size=64
output_size=128
temp=0.07
batch_size=64

weights_file=os.path.join(weights_dir,f"simclr_lr{lr_}_l2{l2_}_z{representation_size}_x{output_size}_temp{temp}_bs{batch_size}.pt")
details_=f"z{representation_size}_l2{l2_}_x{output_size}_lr{lr_}_temp{temp}_bs{batch_size}"


fs = Fs
slice_sec = 2.5
slide_sec = 0.2

nperseg = int(slice_sec * fs)
step = int(slide_sec * fs)
noverlap = nperseg - step

triage_segments = pd.read_csv(os.path.join(data_dir, 'triage/segments.csv'))


train_ids, test_ids = joblib.load(os.path.join(data_dir, "triage/ids.joblib"))
np.random.seed(123)
train_encoder_ids=np.random.choice(train_ids,size=660,replace=False)

train = triage_segments[triage_segments['id'].isin(train_ids)]
non_repeated_ids = [k for k, v in train['id'].value_counts().items() if v <= 15]
train = train[~train['id'].isin(non_repeated_ids)]  # remove non repeating ids
train_encoder = train[train['id'].isin(train_encoder_ids)]
train_classifier = train[~train['id'].isin(train_encoder_ids)]
test = triage_segments[triage_segments['id'].isin(test_ids)]


sfft_fun = partial(stft, fs=fs, nperseg=nperseg, noverlap=noverlap, spec_only=True)

train_transformation = nn.Sequential(
    torchaudio.transforms.TimeMasking(time_mask_param=5),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=3),
)

aug_gausian = partial(gaus_noise, min_sd=1e-5, max_sd=1e-1)

encoder_train_dataset = TriagePairs(train_encoder, id_var="id", stft_fun=sfft_fun,
                            transforms=train_transformation, aug_raw=aug_gausian)
encoder_train_loader = DataLoader(encoder_train_dataset, batch_size=batch_size, shuffle=True, num_workers=50)

encoder_test_dataset = TriagePairs(train_classifier, id_var="id", stft_fun=sfft_fun, aug_raw=None)
encoder_test_loader = DataLoader(encoder_test_dataset, batch_size=64, shuffle=False, num_workers=5)

classifier_train_dataset=TriageDataset(train_encoder,stft_fun=sfft_fun)
classifier_train_loader=DataLoader(classifier_train_dataset,batch_size=64,shuffle=False)

classifier_test_dataset=TriageDataset(test,stft_fun=sfft_fun)
classifier_test_loader=DataLoader(classifier_test_dataset,batch_size=64,shuffle=False)

iter_ = iter(encoder_train_loader)
x1_raw, x1_stft, x2_raw, x2_stft = iter_.next()
print("Input shape", x1_raw.shape, " and ", x1_stft.shape)

if display:
    fig, axs = plt.subplots(4, 3, figsize=(12, 5))
    for r in range(2):
        for c in range(3):
            axs[r * 2 + 1, c].imshow(x1_stft.numpy()[c, r, :, :],
                                     # vmin=batch_x.numpy()[:, r, :, :].min(),
                                     # vmax=batch_x.numpy()[:, r, :, :].max(),
                                     norm=colors.LogNorm()
                                     )
            axs[r * 2 + 1, c].invert_yaxis()
    for r in range(2):
        for c in range(3):
            axs[r * 2, c].plot(x1_raw.numpy()[c, r, :],
                               # vmin=batch_x.numpy()[:, r, :, :].min(),
                               # vmax=batch_x.numpy()[:, r, :, :].max(),
                               # norm=colors.LogNorm()
                               )

    plt.show()


device="cuda" if torch.cuda.is_available() else "cpu"
# model=Classifier(in_features=2,hid_dim=64,z_dim=64)

model_raw=resnet1d(num_classes=1,norm_layer=groupNorm(8))
model_stft=wideresnet50(num_classes=1,norm_layer=groupNorm(8))
classifier=MLP(representation_size=representation_size,dropout=0.0,num_classes=output_size)
model=Encoder(raw_model=model_raw,stft_model=model_stft,
          representation_size=representation_size)
model.fc=classifier
model=model.to(device)
# optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001 ,weight_decay=1e-4)
optimizer=torch.optim.Adam(params=model.parameters(),lr=lr_ ,weight_decay=l2_)
# optimizer=torch.optim.SGD(params=model.parameters(),lr=lr_ ,weight_decay=l2_,momentum=0.99)
criterion=NTXentLoss(device=device, temperature=temp, use_cosine_similarity=True).to(device)
epochs=500
scheduler=ReduceLROnPlateau(optimizer,factor=0.5,patience=10,verbose=True,min_lr=1e-5)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-1,epochs=epochs,steps_per_epoch=len(train_loader))

epoch=0
best_model={'loss':np.Inf,'params':None}
early_stop_counter=0
losses=[]
for epoch in range(epoch,epochs):
    train_loss=0
    test_loss=0

    model.train()
    print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch,optimizer.param_groups[0]['lr']))
    for x1_raw, x1_stft, x2_raw, x2_stft in tqdm(encoder_train_loader):
        x1_raw, x1_stft=x1_raw.to(device,dtype=torch.float), x1_stft.to(device,dtype=torch.float)
        xis=model(x1_raw, x1_stft)
        x2_raw, x2_stft=x2_raw.to(device,dtype=torch.float), x2_stft.to(device,dtype=torch.float)
        xjs = model(x2_raw, x2_stft)
        xis = F.normalize(xis, dim=1)
        xjs = F.normalize(xjs, dim=1)
        loss=criterion(xis,xjs,x1_raw.shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()/len(encoder_train_loader)
        # scheduler.step()

    model.eval()
    with torch.no_grad():
        for x1_raw, x1_stft, x2_raw, x2_stft in tqdm(encoder_test_loader):
            x1_raw, x1_stft = x1_raw.to(device, dtype=torch.float), x1_stft.to(device, dtype=torch.float)
            xis = model(x1_raw, x1_stft)
            x2_raw, x2_stft = x2_raw.to(device, dtype=torch.float), x2_stft.to(device, dtype=torch.float)
            xjs = model(x2_raw, x2_stft)

            xis = F.normalize(xis, dim=1)
            xjs = F.normalize(xjs, dim=1)

            loss = criterion(xis, xjs, x1_raw.shape[0])
            test_loss += loss.item() / len(encoder_test_loader)
    print("Epoch: %d: loss %.3f, val_loss %.3f" % (epoch, train_loss, test_loss))
    losses.append((train_loss, test_loss))
    scheduler.step(test_loss)

    if test_loss < best_model['loss']:
        best_model['loss']=test_loss
        best_model['params']=copy.deepcopy(model.state_dict())
        best_model['epoch']=epoch
        early_stop_counter=0
    else:
        early_stop_counter+=1
        if early_stop_counter>=100:
            print("Early stopping ... ")
            break


torch.save(best_model['params'],weights_file)

fig,ax=plt.subplots(1)
ax.plot([train for train, test in losses],label="train")
ax.plot([test for train, test in losses],label="test")
plt.legend()
plt.savefig("/home/pmwaniki/Dropbox/tmp/simclr_%s__lr%.5f_l2%.5f_temp%.3f_size%d.png" % (os.uname()[1],lr_,l2_,temp,representation_size))
if display:
    plt.show()
else:
    plt.close()


model.load_state_dict(best_model['params'])

model.fc=nn.Identity()
model.eval()

classifier_embedding=[]
with torch.no_grad():
    for x1_raw, x1_stft in tqdm(classifier_train_loader):
        x1_raw, x1_stft = x1_raw.to(device, dtype=torch.float), x1_stft.to(device, dtype=torch.float)
        xis = model(x1_raw, x1_stft)
        xis=F.normalize(xis,dim=1)
        classifier_embedding.append(xis.cpu().detach().numpy())

test_embedding=[]
with torch.no_grad():
    for x1_raw, x1_stft in tqdm(classifier_test_loader):
        x1_raw, x1_stft = x1_raw.to(device, dtype=torch.float), x1_stft.to(device, dtype=torch.float)
        xis = model(x1_raw, x1_stft)
        xis = F.normalize(xis, dim=1)
        test_embedding.append(xis.cpu().detach().numpy())

classifier_embedding=np.concatenate(classifier_embedding)
test_embedding=np.concatenate(test_embedding)
test_identifier=test['id']
subject_embeddings=[]
subject_ids=[]
subject_admitted=[]
for id in test_identifier.unique():
    temp_=test_embedding[test_identifier==id]
    subject_embeddings.append(temp_.mean(axis=0))
    subject_ids.append(id)
    subject_admitted.append(test.loc[test['id']==id,'admitted'].iloc[0])
subject_embeddings=np.stack(subject_embeddings)
scl=StandardScaler()
subject_scl=scl.fit_transform(subject_embeddings)
pca=PCA(n_components=6)
subject_pca=pca.fit_transform(subject_scl)
subject_admitted=np.array(subject_admitted)




fig,axs=plt.subplots(3,5,figsize=(15,10))
for ax,vals in zip(axs.flatten(),itertools.combinations(range(6),2)):
    r,c=vals
    ax.scatter(subject_pca[subject_admitted == 0, r], subject_pca[subject_admitted == 0, c],
                                          marker="o", label="No",
                                          alpha=0.5)
    ax.scatter(subject_pca[subject_admitted == 1, r], subject_pca[subject_admitted == 1, c],
                                          marker="o", label="Yes",
                                          alpha=0.5)
    ax.set_xlabel(f"PCA {r + 1}")
    ax.set_ylabel(f"PCA {c + 1}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig(f"/home/pmwaniki/Dropbox/tmp/simclr_embedding_lr{lr_}_l2{l2_}_temp{temp}_size{representation_size}.png")
if display:
    plt.show(block=False)
else:
    plt.close()


scl=StandardScaler()
scl_classifier_embedding=scl.fit_transform(classifier_embedding)
scl_test_embedding=scl.transform(test_embedding)

# pca=PCA(n_components=6)
# train_pca=pca.fit_transform(scl_classifier_embedding)
# test_pca=pca.transform(scl_test_embedding)




# base_clf=SVC(probability=True,class_weight='balanced')
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-2,1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

base_clf=LogisticRegression(class_weight='balanced',max_iter=10000,penalty='l2')
tuned_parameters = { 'C': [1e-4,1e-3,1e-2,1e-1,1, 10, 100, 1000,1e4],}

clf=GridSearchCV(base_clf,param_grid=tuned_parameters,cv=StratifiedKFold(10,),
                 verbose=1,n_jobs=10,
                 scoring=['f1','roc_auc','recall','precision'],refit='roc_auc')
clf.fit(scl_classifier_embedding,train_encoder['admitted'])

cv_results=pd.DataFrame({'params':clf.cv_results_['params'], 'auc':clf.cv_results_['mean_test_roc_auc'],
              'f1':clf.cv_results_['mean_test_f1'],'recall':clf.cv_results_['mean_test_recall'],
                          'precision':clf.cv_results_['mean_test_precision']})
print(cv_results)

test_pred=clf.predict_proba(scl_test_embedding)[:,1]

print(classification_report(test['admitted'],test_pred>0.5))
print("AUC: ",roc_auc_score(test['admitted'],test_pred))

final_predictions=pd.DataFrame({'admitted':test['admitted'],
                                 'id':test['id'],
                                 'prediction':test_pred})
final_predictions2=final_predictions.groupby('id').agg('mean')
print(classification_report(final_predictions2['admitted'],(final_predictions2['prediction']>0.5)*1.0))

print("AUC: %.2f" % roc_auc_score(final_predictions2['admitted'],final_predictions2['prediction']))


report=classification_report(final_predictions2['admitted'],(final_predictions2['prediction']>0.5)*1.0,output_dict=True)
recall=report['1.0']['recall']
precision=report['1.0']['precision']
f1=report['1.0']['f1-score']
specificity=report['0.0']['recall']
acc=report['accuracy']
auc=roc_auc_score(final_predictions2['admitted'],final_predictions2['prediction'])



save_table3(model="SimCLR",precision=precision,recall=recall,specificity=specificity,
            auc=auc,details=details_,other=json.dumps({'host':os.uname()[1],'f1':f1,
                                                       'acc':acc,'batch_size':batch_size}))

joblib.dump(clf,weights_file.replace(".pt",".joblib"))

if __name__=="__main__":
    pass