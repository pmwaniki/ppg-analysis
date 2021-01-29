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
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression,SelectPercentile
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor
from settings import data_dir,weights_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import save_table3

cores=multiprocessing.cpu_count()-2
experiment="Contrastive-sample-DotProduct32"
weights_file=os.path.join(weights_dir,f"Contrastive_{experiment}_svm.joblib")
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")
dist_fun="euclidean" if "LpDistance" in experiment else scipy.spatial.distance.cosine

classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)

train_ids=train['id'].unique()
test_ids=test['id'].unique()
classifier_embedding_reduced=np.stack(map(lambda id:classifier_embedding[train['id']==id,:].mean(axis=0) ,train_ids))
test_embedding_reduced=np.stack(map(lambda id:test_embedding[test['id']==id,:].mean(axis=0) ,test_ids))
# admitted_train=np.stack(map(lambda id:train.loc[train['id']==id,'admitted'].iat[0],train_ids))
# admitted_test=np.stack(map(lambda id:test.loc[test['id']==id,'admitted'].iat[0],test_ids))
#
hr_train=np.stack(map(lambda id:train.loc[train['id']==id,'hr'].median(skipna=True),train_ids))
hr_test=np.stack(map(lambda id:test.loc[test['id']==id,'hr'].median(skipna=True),test_ids))

resp_rate_train=np.stack(map(lambda id:train.loc[train['id']==id,'resp_rate'].median(skipna=True),train_ids))
resp_rate_test=np.stack(map(lambda id:test.loc[test['id']==id,'resp_rate'].median(skipna=True),test_ids))

spo2_train=np.stack(map(lambda id:train.loc[train['id']==id,'spo2'].median(skipna=True),train_ids))
spo2_test=np.stack(map(lambda id:test.loc[test['id']==id,'spo2'].median(skipna=True),test_ids))

#******************************************************************************************************************

regressor_hr=SGDRegressor(loss='squared_loss',max_iter=100000,early_stopping=True)
hr_grid={'clf__regressor__alpha':[1e-5,1e-4,1e-3,1e-2,1e-1,1.0,],
                'clf__regressor__eta0':[0.00001,0.0001,0.001,0.01,0.1,],
         'poly__degree':[2,],
         'select__percentile': [1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100],
         'select__score_func':[mutual_info_regression,]
        }
pipeline_hr = Pipeline([

    ('poly', PolynomialFeatures(interaction_only=True, include_bias=False)),
    ('select', SelectPercentile()),
    ('scl', StandardScaler()),
    ('clf', TransformedTargetRegressor(regressor=regressor_hr,
                                       #                                       transformer=QuantileTransformer(output_distribution="normal",n_quantiles=1000),
                                       func=lambda x: x, inverse_func=lambda x: x
                                       )),
])

hr_clf=GridSearchCV(pipeline_hr,param_grid=hr_grid,cv=KFold(10),n_jobs=cores,
                    scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],
                    refit='r2')
hr_clf.fit(classifier_embedding_reduced[hr_train!=0,:],hr_train[hr_train!=0])

cv_results_hr=pd.DataFrame({'params':hr_clf.cv_results_['params'],
                              'rmse':hr_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':hr_clf.cv_results_['mean_test_r2'],
                              'max_error':hr_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_hr)

test_pred_hr=hr_clf.predict(test_embedding_reduced)
r2=r2_score(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ])
rmse=mean_squared_error(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ],squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ])
ax2.plot([50,225],[50,225],'r--')
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()

joblib.dump(hr_clf,os.path.join(data_dir,f"results/Regression_hr_{experiment}.joblib"))

##*******************************************************************************************************************
#resp rate
regressor=SGDRegressor(loss='squared_loss',max_iter=100000,early_stopping=True)
# regressor=Lasso(max_iter=50000)
pipeline_resp_rate=Pipeline([

    ('poly',PolynomialFeatures(interaction_only=False,include_bias=False)),
    ('select',SelectPercentile()),
    ('scl', StandardScaler()),
    ('clf',TransformedTargetRegressor(regressor=regressor,
#                                       transformer=QuantileTransformer(output_distribution="normal", n_quantiles=500),
                                      func=lambda x:x,inverse_func=lambda x:x,
                                      )),
])
resp_rate_grid = {'clf__regressor__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, ],
                  'clf__regressor__eta0': [0.00001, 0.0001, 0.001, 0.01, 0.1, ],
                  #                 'clf__transformer__n_quantiles':[200,300,500,700,900],
                  # 'pca__n_components':[2,4,8,16,32],
                  'poly__degree': [2, ],
                  'select__percentile': [1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100],
                  'select__score_func': [ mutual_info_regression,]
                  }
resp_rate_clf=GridSearchCV(pipeline_resp_rate,param_grid=resp_rate_grid,cv=KFold(10),n_jobs=cores,
                    scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],
                    refit='r2')
resp_rate_clf.fit(classifier_embedding_reduced[~np.isnan(resp_rate_train),:],resp_rate_train[~np.isnan(resp_rate_train)])

cv_results_resp_rate=pd.DataFrame({'params':resp_rate_clf.cv_results_['params'],
                              'rmse':resp_rate_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':resp_rate_clf.cv_results_['mean_test_r2'],
                              'max_error':resp_rate_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_resp_rate)

test_pred_resp_rate=resp_rate_clf.predict(test_embedding_reduced)
r2=r2_score(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)])
rmse=mean_squared_error(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)],squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)])
ax2.plot([20,100],[20,100],'r--')
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()


#spo2*****************************************************************************************************************
plt.hist(spo2_train[spo2_train>70])
plt.show()


regressor_spo2=SGDRegressor(loss='squared_loss',max_iter=500000,early_stopping=True,random_state=123)
# regressor=Lasso(max_iter=50000)
pipeline_spo2 = Pipeline([

    # ('pca',PCA()),
    ('poly', PolynomialFeatures(interaction_only=True, include_bias=False)),
    ('select', SelectPercentile()),
    ('scl', StandardScaler()),
    ('clf', TransformedTargetRegressor(regressor=regressor_spo2,
                                       # transformer=QuantileTransformer(output_distribution="normal", n_quantiles=500),
                                       func=lambda x: x, inverse_func=lambda x: x,
                                       )),
])
spo2_grid = {'clf__regressor__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, ],
             'clf__regressor__eta0': [0.00001, 0.0001, 0.001, 0.01, 0.1, ],
             # 'clf__transformer__n_quantiles':[5,20,200,300,500,700,900],
             # 'pca__n_components':[2,4,8,16,32],
             'poly__degree': [2,  ],
             'select__percentile': [1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100],
             'select__score_func': [mutual_info_regression,],
             }
spo2_clf=GridSearchCV(pipeline_spo2,param_grid=spo2_grid,cv=KFold(10),n_jobs=cores,
                    scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],
                    refit='r2')
spo2_clf.fit(classifier_embedding_reduced[spo2_train>70,:],spo2_train[spo2_train>70])

cv_results_spo2=pd.DataFrame({'params':spo2_clf.cv_results_['params'],
                              'rmse':spo2_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':spo2_clf.cv_results_['mean_test_r2'],
                              'max_error':spo2_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_spo2)

test_pred_spo2=spo2_clf.predict(test_embedding_reduced)
r2=r2_score(spo2_test[spo2_test>70],test_pred_spo2[spo2_test>70])
rmse=mean_squared_error(spo2_test[spo2_test>70],test_pred_spo2[spo2_test>70],squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(spo2_test[spo2_test>70],test_pred_spo2[spo2_test>70])
ax2.plot([70,100],[70,100],'r--')
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
ax2.set_title("SPO2")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()



#combined plot ******************************************************************************************************************************************
fig,axs=plt.subplots(1,3)

axs[0].scatter(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ])
axs[0].plot([50,225],[50,225],'r--')
axs[0].set_xlabel("Observed")
axs[0].set_ylabel("Predicted")
axs[0].set_title("Respiratory rate")



plt.show()



