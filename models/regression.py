import matplotlib.pyplot as plt
import os
import json
import sys
import multiprocessing

import joblib
import numpy as np
import pandas as pd
import scipy

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,PolynomialFeatures
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.linear_model import Ridge,Lasso,LinearRegression,SGDRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor
from settings import data_dir,weights_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import save_table3

cores=multiprocessing.cpu_count()-2
experiment="Contrastive-DotProduct32"
weights_file=os.path.join(weights_dir,f"Contrastive_{experiment}_svm.joblib")
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")
dist_fun="euclidean" if "LpDistance" in experiment else scipy.spatial.distance.cosine

classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)

train_ids=train['id'].unique()
test_ids=test['id'].unique()
# classifier_embedding_reduced=np.stack(map(lambda id:classifier_embedding[train['id']==id,:].mean(axis=0) ,train_ids))
# test_embedding_reduced=np.stack(map(lambda id:test_embedding[test['id']==id,:].mean(axis=0) ,test_ids))
# admitted_train=np.stack(map(lambda id:train.loc[train['id']==id,'admitted'].iat[0],train_ids))
# admitted_test=np.stack(map(lambda id:test.loc[test['id']==id,'admitted'].iat[0],test_ids))
#
# hr_train=np.stack(map(lambda id:train.loc[train['id']==id,'hr'].iat[0],train_ids))
# hr_test=np.stack(map(lambda id:test.loc[test['id']==id,'hr'].iat[0],test_ids))

q_transformer=QuantileTransformer(output_distribution="normal",n_quantiles=1000)
# regressor=KNeighborsRegressor(weights='distance',metric=dist_fun,)
# hr_grid={'clf__regressor__n_neighbors':[1,3,5,10,15,30,50,100,300],
#          'clf__regressor__weights':['distance'],
#                 'pca__n_components':[2,4,8,16,32]}
regressor=LinearRegression()
hr_grid={
                'pca__n_components':[2,4,8,16,32]}
pipeline_hr=Pipeline([
    ('scl',StandardScaler()),
    ('pca',PCA()),
    ('clf',TransformedTargetRegressor(regressor=regressor,
                                      transformer=q_transformer)),
])

hr_clf=GridSearchCV(pipeline_hr,param_grid=hr_grid,cv=KFold(10),n_jobs=cores,
                    scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],
                    refit='neg_root_mean_squared_error')
hr_clf.fit(classifier_embedding[train.hr!=0,:],train.loc[train.hr!=0,'hr'])

cv_results_hr=pd.DataFrame({'params':hr_clf.cv_results_['params'],
                              'rmse':hr_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':hr_clf.cv_results_['mean_test_r2'],
                              'max_error':hr_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_hr)

test_pred_hr=hr_clf.predict(test_embedding)
r2=r2_score(test.loc[test['hr'] !=0,'hr'],test_pred_hr[test['hr'] !=0 ])
rmse=mean_squared_error(test.loc[test['hr'] !=0,'hr'],test_pred_hr[test['hr'] !=0 ],squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(test.loc[test['hr'] !=0,'hr'],test_pred_hr[test['hr'] !=0 ])
ax2.plot([50,225],[50,225],'r--')
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()


#spo2
pipeline_spo2=Pipeline([
    ('scl',RobustScaler()),
    ('pca',PCA()),
    ('clf',TransformedTargetRegressor(regressor=Lasso(max_iter=50000),
                                      func=lambda x:x, inverse_func=lambda x:x)),
])
spo2_grid={'clf__regressor__alpha':[0.00001,0.0001,0.001,0.01,0.1,1.0,],
                'pca__n_components':[2,4,8,16,32]}
spo2_clf=GridSearchCV(pipeline_spo2,param_grid=spo2_grid,cv=KFold(10),n_jobs=cores,
                    scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],
                    refit='r2')
spo2_clf.fit(classifier_embedding[train.spo2>=70,:],train.loc[train.spo2>=70,'spo2'])

cv_results_spo2=pd.DataFrame({'params':spo2_clf.cv_results_['params'],
                              'rmse':spo2_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':spo2_clf.cv_results_['mean_test_r2'],
                              'max_error':spo2_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_spo2)

test_pred_spo2=spo2_clf.predict(test_embedding)
r2=r2_score(test.loc[test['spo2'] !=0,'spo2'],test_pred_spo2[test['spo2'] !=0 ])
rmse=mean_squared_error(test.loc[test['spo2'] !=0,'spo2'],test_pred_spo2[test['spo2'] !=0 ],squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(test.loc[test['spo2'] >=70,'spo2'],test_pred_spo2[test['spo2'] >=70 ])
ax2.plot([75,100],[75,100],'r--')
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()

#spo2-PLS
pipeline_spo2=Pipeline([
    # ('scl',StandardScaler()),
    # ('pca',PCA()),
    ('clf',TransformedTargetRegressor(regressor=PLSRegression(),
                                      func=lambda x:x,inverse_func=lambda x:x)),
])
spo2_grid={
                'clf__regressor__n_components':[2,4,8,16,32]}
spo2_clf=GridSearchCV(pipeline_spo2,param_grid=spo2_grid,cv=KFold(10),n_jobs=cores,
                    scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],
                    refit='r2')
spo2_clf.fit(classifier_embedding[train.spo2>=70,:],train.loc[train.spo2>=70,'spo2'])

cv_results_spo2=pd.DataFrame({'params':spo2_clf.cv_results_['params'],
                              'rmse':spo2_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':spo2_clf.cv_results_['mean_test_r2'],
                              'max_error':spo2_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_spo2)

test_pred_spo2=spo2_clf.predict(test_embedding)
r2=r2_score(test.loc[test['spo2'] !=0,'spo2'],test_pred_spo2[test['spo2'] !=0 ])
rmse=mean_squared_error(test.loc[test['spo2'] !=0,'spo2'],test_pred_spo2[test['spo2'] !=0 ],squared=False)


#resp rate
regressor=SGDRegressor(loss='huber',max_iter=10000,early_stopping=True)
# regressor=Lasso(max_iter=50000)
pipeline_resp_rate=Pipeline([
    ('scl',RobustScaler()),
    # ('pca',PCA()),
    ('poly',PolynomialFeatures()),
# ('scl',RobustScaler()),
    ('clf',TransformedTargetRegressor(regressor=regressor,
                                      # transformer=QuantileTransformer(output_distribution="normal", n_quantiles=1000),
                                      func=lambda x:np.log(x),inverse_func=lambda x:np.exp(x),
                                      )),
])
resp_rate_grid={'clf__regressor__alpha':[1e-5,1e-4,1e-3,1e-2,1e-1,1.0],
                'clf__regressor__eta0':[0.001,0.01,0.1],
                # 'pca__n_components':[2,4,8,16,32],
                'poly__degree':[2,3,],
                }
resp_rate_clf=GridSearchCV(pipeline_resp_rate,param_grid=resp_rate_grid,cv=KFold(10),n_jobs=cores,
                    scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],
                    refit='neg_root_mean_squared_error')
resp_rate_clf.fit(classifier_embedding[~train.resp_rate.isna(),:],train.loc[~train.resp_rate.isna(),'resp_rate'])

cv_results_resp_rate=pd.DataFrame({'params':resp_rate_clf.cv_results_['params'],
                              'rmse':resp_rate_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':resp_rate_clf.cv_results_['mean_test_r2'],
                              'max_error':resp_rate_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_resp_rate)

test_pred_resp_rate=resp_rate_clf.predict(test_embedding)
r2=r2_score(test.loc[~test.resp_rate.isna(),'resp_rate'],test_pred_resp_rate[~test.resp_rate.isna() ])
rmse=mean_squared_error(test.loc[~test.resp_rate.isna(),'resp_rate'],test_pred_resp_rate[~test.resp_rate.isna() ],squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(test.loc[~test.resp_rate.isna(),'resp_rate'],test_pred_resp_rate[~test.resp_rate.isna() ])
ax2.plot([20,100],[20,100],'r--')
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()
