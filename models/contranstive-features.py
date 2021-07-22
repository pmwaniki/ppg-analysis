import os,json

import sys

from settings import checkpoint_dir as log_dir,data_dir,weights_dir,base_dir
import itertools
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
from ray.tune.analysis.experiment_analysis import Analysis
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsClassifier

from models.cnn.networks import resnet50_1d,EncoderRaw,wideresnet50_1d
from datasets.loaders import TriageDataset,TriagePairs
from utils import save_table3


# from models.contrastive_resnet import get_model,accuracy_fun

display=os.environ.get("DISPLAY",None)
experiment="Contrastive-original-sample-DotProduct32-sepsis"
res_type=experiment.split("-")[1]
trial_dir=os.path.join(log_dir,experiment)
weights_file=os.path.join(weights_dir,f"Contrastive_{experiment}.pt")

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
# non_repeated_ids = [k for k, v in train['id'].value_counts().items() if v <= 15]
# train = train[~train['id'].isin(non_repeated_ids)]  # remove non repeating ids
# train_encoder = train[train['id'].isin(train_encoder_ids)]


test=triage_segments[triage_segments['id'].isin(test_ids)]
test_full=test.copy()
non_repeated_test_ids = [k for k, v in test['id'].value_counts().items() if v <= 15]
test_encoder = test[~test['id'].isin(non_repeated_test_ids)]

encoder_test_dataset = TriagePairs(test_encoder, id_var="id", stft_fun=None, aug_raw=[],normalize=True)
encoder_test_loader = DataLoader(encoder_test_dataset, batch_size=16, shuffle=False, num_workers=5)
#
classifier_train_dataset=TriageDataset(train_full,normalize=True)
classifier_train_loader=DataLoader(classifier_train_dataset,batch_size=16,shuffle=False,num_workers=5)

classifier_test_dataset=TriageDataset(test_full,normalize=True)
classifier_test_loader=DataLoader(classifier_test_dataset,batch_size=16,shuffle=False,num_workers=5)



analysis=Analysis(trial_dir)
score="accuracy"
mode="max"
best_trial=analysis.get_best_logdir(score,mode)
best_dir=analysis.get_best_logdir(score,mode)
trial_data=analysis.dataframe(score,mode)
best_config=analysis.get_best_config(score,mode)







# device="cpu"
# best_config={'representation_size':64,'dropout':0.0,'enc_output_size':32}
# model_state=torch.load("/home/pmwaniki/data/ppg/results/weights/Contrastive_Contrastive-DotProduct.pt")
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

def accuracy_fun(xis_embeddings, xjs_embeddings, distance="minkowski"):
    knn = KNeighborsClassifier(n_neighbors=1,metric=distance)
    knn.fit(xis_embeddings, np.arange(xis_embeddings.shape[0]))
    knn_pred = knn.predict(xjs_embeddings)
    accuracy = np.mean(np.arange(xis_embeddings.shape[0]) == knn_pred)
    return accuracy

best_trials=trial_data.sort_values("accuracy",ascending=False).head(5).reset_index()

device="cuda" if torch.cuda.is_available() else "cpu"
trial_embeddings=[]
for i,row in best_trials.iterrows():
    with open(os.path.join(row['logdir'],'params.json'),'r') as f:
        config_=json.load(f)
    model_state_,_=torch.load(os.path.join(row['logdir'],"checkpoint_700","model.pth"),map_location=device)
    model_=get_model(config_)
    model_.load_state_dict(model_state_)
    model_.to(device)
    model_.eval()
    xis_embeddings = []
    xjs_embeddings = []
    with torch.no_grad():
        for x1_raw, x2_raw in encoder_test_loader:
            x1_raw = x1_raw.to(device, dtype=torch.float)
            xis = model_(x1_raw)
            x2_raw = x2_raw.to(device, dtype=torch.float)
            xjs = model_(x2_raw)
            xis = nn.functional.normalize(xis, dim=1)
            xjs = nn.functional.normalize(xjs, dim=1)
            xis_embeddings.append(xis.cpu().detach().numpy())
            xjs_embeddings.append(xjs.cpu().detach().numpy())

    xis_embeddings = np.concatenate(xis_embeddings)
    xjs_embeddings = np.concatenate(xjs_embeddings)
    accuracy = accuracy_fun(xis_embeddings, xjs_embeddings)
    print(f"Accuracy for trial {i}: {accuracy}")

    model_.fc = nn.Identity()
    model_.eval()

    classifier_embedding = []
    with torch.no_grad():
        for x1_raw in tqdm(classifier_train_loader):
            x1_raw = x1_raw.to(device, dtype=torch.float)
            xis = model_(x1_raw)
            xis = nn.functional.normalize(xis, dim=1)
            classifier_embedding.append(xis.cpu().detach().numpy())

    test_embedding = []
    with torch.no_grad():
        for x1_raw in tqdm(classifier_test_loader):
            x1_raw = x1_raw.to(device, dtype=torch.float)
            xis = model_(x1_raw)
            xis = nn.functional.normalize(xis, dim=1)
            test_embedding.append(xis.cpu().detach().numpy())

    classifier_embedding = np.concatenate(classifier_embedding)
    test_embedding = np.concatenate(test_embedding)
    trial_embeddings.append((classifier_embedding,test_embedding))

joblib.dump((trial_embeddings,train,test),
            os.path.join(data_dir,f"results/{experiment}_top5.joblib"))



# model_path=glob.glob1(best_dir,"checkpoint_*")[-1]
# model_state,optimizer_state=torch.load(os.path.join(best_dir,model_path,"model.pth"),map_location=device)
# torch.save(model_state,weights_file)
#
#
# model=get_model(best_config)
#
#
# model.load_state_dict(model_state)
# model.to(device)
# # Test model accuracy
# model.eval()
# xis_embeddings = []
# xjs_embeddings = []
# with torch.no_grad():
#     for x1_raw, x2_raw in encoder_test_loader:
#         x1_raw = x1_raw.to(device, dtype=torch.float)
#         xis = model(x1_raw)
#         x2_raw = x2_raw.to(device, dtype=torch.float)
#         xjs = model(x2_raw)
#         xis = nn.functional.normalize(xis, dim=1)
#         xjs = nn.functional.normalize(xjs, dim=1)
#         xis_embeddings.append(xis.cpu().detach().numpy())
#         xjs_embeddings.append(xjs.cpu().detach().numpy())
#
# xis_embeddings = np.concatenate(xis_embeddings)
# xjs_embeddings = np.concatenate(xjs_embeddings)
# accuracy = accuracy_fun(xis_embeddings, xjs_embeddings)
#
# model.fc=nn.Identity()
# model.eval()
#
# classifier_embedding=[]
# with torch.no_grad():
#     for x1_raw in tqdm(classifier_train_loader):
#         x1_raw = x1_raw.to(device, dtype=torch.float)
#         xis = model( x1_raw)
#         xis=nn.functional.normalize(xis,dim=1)
#         classifier_embedding.append(xis.cpu().detach().numpy())
#
# test_embedding=[]
# with torch.no_grad():
#     for x1_raw in tqdm(classifier_test_loader):
#         x1_raw = x1_raw.to(device, dtype=torch.float)
#         xis = model( x1_raw)
#         xis = nn.functional.normalize(xis, dim=1)
#         test_embedding.append(xis.cpu().detach().numpy())
#
# classifier_embedding=np.concatenate(classifier_embedding)
# test_embedding=np.concatenate(test_embedding)
#
# joblib.dump((classifier_embedding,test_embedding,train,test),
#             os.path.join(data_dir,f"results/{experiment}.joblib"))
#
# test_identifier=test['id']
# subject_embeddings=[]
# subject_ids=[]
# subject_admitted=[]
# subject_died=[]
# for id in test_identifier.unique():
#     temp_=test_embedding[test_identifier==id]
#     subject_embeddings.append(temp_.mean(axis=0))
#     subject_ids.append(id)
#     subject_admitted.append(test.loc[test['id']==id,'admitted'].iloc[0])
#     subject_died.append(test.loc[test['id'] == id, 'died'].iloc[0])
#
#
# subject_embeddings=np.stack(subject_embeddings)
# scl=StandardScaler()
# subject_scl=scl.fit_transform(subject_embeddings)
# pca=PCA(n_components=6)
# subject_pca=pca.fit_transform(subject_scl)
# subject_admitted=np.array(subject_admitted)
# subject_died=np.array(subject_died)
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
# fig.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_{os.uname()[1]}_{experiment}.png")
# if display:
#     plt.show(block=False)
# else:
#     plt.close()
#
#
# # scl=StandardScaler()
# # scl_classifier_embedding=scl.fit_transform(classifier_embedding)
# # scl_test_embedding=scl.transform(test_embedding)
#
# # pca=PCA(n_components=6)
# # train_pca=pca.fit_transform(scl_classifier_embedding)
# # test_pca=pca.transform(scl_test_embedding)
#
#
#
# ##CLASSIFICATION: ADMISSION
# base_clf=SVC(probability=True,class_weight='balanced')
# tuned_parameters = [
#     {'clf__kernel': ['rbf'], 'clf__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
# #                      'pca__n_components':[int(64/2**i) for i in range(5)],
#      'clf__C': [0.01,0.1,1, 10, 100, 1000,1e4]
#     },
#                     {'clf__kernel': ['linear'], 'clf__C': [1, 10, 100, 1000],
# #                      'pca__n_components':[int(64/2**i) for i in range(5)],
#                     }
#                    ]
#
# # base_clf=LogisticRegression(class_weight='balanced',max_iter=1000,penalty='elasticnet',solver='saga')
#
#
#
# # tuned_parameters = { 'clf__C': [1e-4,1e-3,1e-2,1e-1,1,],
# #                      'pca__n_components':[int(64/2**i) for i in range(5)],
# #                      'clf__l1_ratio':[0.2,0.5,0.8]}
#
# pipeline=Pipeline([
#     ('scl',RobustScaler()),
# #     ('pca',PCA()),
#     ('clf',base_clf),
# ])
#
# clf=GridSearchCV(pipeline,param_grid=tuned_parameters,cv=StratifiedKFold(10,),
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
# save_table3(model="Contrastive",precision=precision,recall=recall,specificity=specificity,
#             auc=auc,details=experiment,other=json.dumps({'host':os.uname()[1],'f1':f1,
#                                                        'acc':acc,'config':best_config}))
#
# joblib.dump(clf,weights_file.replace(".pt",".joblib"))
#
#
# #REGRESSION: RESPIRATORY RATE
# y_train=train.loc[:,'resp_rate'].values.reshape(-1,1)
# log_y_train=np.log(y_train)
# q_transformer=QuantileTransformer(output_distribution="normal",n_quantiles=1000)
# q_y_train=q_transformer.fit_transform(y_train)
#
# base_clf_resp=SVR()
# tuned_parameters_resp = [
#     {'clf__regressor__kernel': ['rbf'], 'clf__regressor__gamma': [0.00001,0.00005,0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,1.0],
# #                      'pca__n_components':[int(64/2**i) for i in range(5)],
#      'clf__regressor__C': [1, 1e1, 1e2, 1e3,1e4,1e5]
#     },
# #                     {'clf__regressor__kernel': ['linear'], 'clf__regressor__C': [1, 10, 100, 1000,1e4,1e5],
# # #                      'pca__n_components':[int(64/2**i) for i in range(5)],
# #                     }
#                    ]
#
#
# pipeline_resp=Pipeline([
#     ('scl',RobustScaler()),
# #     ('pca',PCA()),
#     ('clf',TransformedTargetRegressor(regressor=base_clf_resp,transformer=q_transformer)),
# ])
#
# clf_resp=RandomizedSearchCV(pipeline_resp,param_distributions=tuned_parameters_resp,
#                             cv=KFold(5,shuffle=False),
#                  verbose=1,n_jobs=10,n_iter=25,
#                  scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],refit='neg_root_mean_squared_error')
# # clf_resp.fit(classifier_embedding[~train['resp_rate'].isna()],train.loc[~train['resp_rate'].isna(),'resp_rate'])
# clf_resp.fit(classifier_embedding[~train['resp_rate'].isna()],y_train[~train['resp_rate'].isna()])
#
# cv_results_resp=pd.DataFrame({'params':clf_resp.cv_results_['params'],
#                               'rmse':clf_resp.cv_results_['mean_test_neg_root_mean_squared_error'],
#               'R2':clf_resp.cv_results_['mean_test_explained_variance'],
#                               'max_error':clf_resp.cv_results_['mean_test_max_error'],
#                           # 'precision':clf.cv_results_['mean_test_precision']
#                               })
# print(cv_results_resp)
#
# test_pred_resp=clf_resp.predict(test_embedding)
# r2=r2_score(test.loc[~test['resp_rate'].isna(),'resp_rate'],test_pred_resp[~test['resp_rate'].isna()])
# rmse=mean_squared_error(test.loc[~test['resp_rate'].isna(),'resp_rate'],test_pred_resp[~test['resp_rate'].isna()],squared=False)
#
# fig2,ax2=plt.subplots(1)
# ax2.scatter(test['resp_rate'],test_pred_resp)
# ax2.plot([20,100],[20,100])
# ax2.set_xlabel("Observed")
# ax2.set_ylabel("Predicted")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
# plt.show()