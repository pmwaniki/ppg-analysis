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
from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,PolynomialFeatures,PowerTransformer
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.linear_model import Ridge,Lasso,LinearRegression,SGDRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression,SelectPercentile,VarianceThreshold
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor
from settings import data_dir,weights_dir,output_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import save_regression

cores=multiprocessing.cpu_count()-2
# experiment="Contrastive-original-sample-DotProduct32-sepsis"
experiment='PCA-32'
# weights_file=os.path.join(weights_dir,f"Contrastive_{experiment}_svm.joblib")
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

def regressor():
    base_regressor = SGDRegressor(loss='squared_loss', penalty='l2', max_iter=5000, early_stopping=False,
                                random_state=123)
    bagging_regressor = BaggingRegressor(base_estimator=base_regressor, n_estimators=10, random_state=123)
    grid = {'clf__base_estimator__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, ],
               'clf__base_estimator__eta0': [0.00001, 0.0001, 0.001, 0.01, 0.1, ],
               'clf__base_estimator__max_iter': [100, 500, 1000, 5000],
               # 'poly__degree':[2,],
               #          'poly__interaction_only':[False],
               'select__percentile': [10, 15, 20, 30, 40, 60, 100],
               'select__score_func': [mutual_info_regression, ]
               }
    pipeline = Pipeline([
        ('variance_threshold', VarianceThreshold()),
        ('select', SelectPercentile()),
        ('poly', PolynomialFeatures(interaction_only=False, include_bias=False)),

        ('scl', StandardScaler()),
        ('clf', bagging_regressor),
    ])

    estimator = GridSearchCV(pipeline, param_grid=grid, cv=KFold(10, random_state=123, shuffle=True), n_jobs=cores,
                          scoring=['explained_variance', 'neg_root_mean_squared_error', 'max_error', 'r2'],
                          refit='r2', verbose=1)
    return estimator


#******************************************************************************************************************



hr_clf=regressor()
hr_clf.fit(classifier_embedding_reduced[hr_train!=0,:],hr_train[hr_train!=0])

cv_results_hr=pd.DataFrame({'params':hr_clf.cv_results_['params'],
                              'rmse':hr_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':hr_clf.cv_results_['mean_test_r2'],
                              'max_error':hr_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_hr)

test_pred_hr=hr_clf.predict(test_embedding_reduced)
r2_hr=r2_score(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ])
rmse_hr=mean_squared_error(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ],squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ])
ax2.plot([50,225],[50,225],'r--')
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()

joblib.dump(hr_clf,os.path.join(data_dir,f"results/weights/Regression_hr_{experiment}.joblib"))

##*******************************************************************************************************************
#resp rate

plt.hist(resp_rate_train[~np.isnan(resp_rate_train)])
plt.show()

pw_transformer=PowerTransformer(method='box-cox')
plt.hist(pw_transformer.fit_transform(resp_rate_train[~np.isnan(resp_rate_train)].reshape(-1,1)))
plt.show()



resp_rate_clf=regressor()
resp_rate_clf.fit(classifier_embedding_reduced[~np.isnan(resp_rate_train),:],resp_rate_train[~np.isnan(resp_rate_train)])

cv_results_resp_rate=pd.DataFrame({'params':resp_rate_clf.cv_results_['params'],
                              'rmse':resp_rate_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':resp_rate_clf.cv_results_['mean_test_r2'],
                              'max_error':resp_rate_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_resp_rate)

test_pred_resp_rate=resp_rate_clf.predict(test_embedding_reduced)
r2_rest_rate=r2_score(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)])
rmse_rest_rate=mean_squared_error(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)],squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)])
ax2.plot([20,100],[20,100],'r--')
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
ax2.set_title("Respiratory rate")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()

joblib.dump(resp_rate_clf,os.path.join(data_dir,f"results/weights/Regression_resp_rate_{experiment}.joblib"))
#spo2*****************************************************************************************************************
plt.hist(spo2_train[spo2_train>70])
plt.show()

pw_transformer=PowerTransformer(method='box-cox',standardize=True)
plt.hist(pw_transformer.fit_transform(spo2_train[spo2_train>70].reshape(-1,1)))
plt.show()

q_transformer=QuantileTransformer(n_quantiles=20)
plt.hist(q_transformer.fit_transform(spo2_train[spo2_train>70].reshape(-1,1)))
plt.show()


spo2_clf=regressor()
spo2_clf.fit(classifier_embedding_reduced[spo2_train>80,:],spo2_train[spo2_train>80])

cv_results_spo2=pd.DataFrame({'params':spo2_clf.cv_results_['params'],
                              'rmse':spo2_clf.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':spo2_clf.cv_results_['mean_test_r2'],
                              'max_error':spo2_clf.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_spo2)

test_pred_spo2=spo2_clf.predict(test_embedding_reduced)
r2_spo2=r2_score(spo2_test[spo2_test>80],test_pred_spo2[spo2_test>80])
rmse_spo2=mean_squared_error(spo2_test[spo2_test>80],test_pred_spo2[spo2_test>80],squared=False)

fig2,ax2=plt.subplots(1)
ax2.scatter(spo2_test[spo2_test>80],test_pred_spo2[spo2_test>80])
ax2.plot([80,100],[80,100],'r--')
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
ax2.set_title("SPO2")
# fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()

joblib.dump(spo2_clf,os.path.join(data_dir,f"results/weights/Regression_spo2_{experiment}.joblib"))

#combined plot ******************************************************************************************************************************************

hr_clf=joblib.load(os.path.join(data_dir,f"results/weights/Regression_hr_{experiment}.joblib"))
test_pred_hr=hr_clf.predict(test_embedding_reduced)
r2_hr=r2_score(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ])
rmse_hr=mean_squared_error(hr_test[hr_test!=0],test_pred_hr[hr_test !=0 ],squared=False)

resp_rate_clf=joblib.load(os.path.join(data_dir,f"results/weights/Regression_resp_rate_{experiment}.joblib"))
test_pred_resp_rate=resp_rate_clf.predict(test_embedding_reduced)
r2_rest_rate=r2_score(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)])
rmse_rest_rate=mean_squared_error(resp_rate_test[~np.isnan(resp_rate_test)],test_pred_resp_rate[~np.isnan(resp_rate_test)],squared=False)

spo2_clf=joblib.load(os.path.join(data_dir,f"results/weights/Regression_spo2_{experiment}.joblib"))
test_pred_spo2=spo2_clf.predict(test_embedding_reduced)
r2_spo2=r2_score(spo2_test[spo2_test>80],test_pred_spo2[spo2_test>80])
rmse_spo2=mean_squared_error(spo2_test[spo2_test>80],test_pred_spo2[spo2_test>80],squared=False)


fig,axs=plt.subplots(1,3,figsize=(12,4))

axs[0].scatter(test_pred_hr[hr_test !=0 ],hr_test[hr_test!=0],)
axs[0].plot([50,225],[50,225],'r--')
axs[0].text(0.05,0.90,f"r2={r2_hr:.2f}\nrmse={rmse_hr:.1f}",transform=axs[0].transAxes,
            bbox={'boxstyle':"round",'facecolor':"wheat",'alpha':0.5})
axs[0].set_ylabel("Observed")
axs[0].set_xlabel("Predicted")
axs[0].set_title("Heart rate")

axs[1].scatter(test_pred_resp_rate[~np.isnan(resp_rate_test)],resp_rate_test[~np.isnan(resp_rate_test)],)
axs[1].plot([20,100],[20,100],'r--')
axs[1].text(0.05,0.9,f"r2={r2_rest_rate:.2f}\nrmse={rmse_rest_rate:.1f}",transform=axs[1].transAxes,
            bbox={'boxstyle':"round",'facecolor':"wheat",'alpha':0.5})
axs[1].set_ylabel("Observed")
axs[1].set_xlabel("Predicted")
axs[1].set_title("Respiratory rate")

axs[2].scatter(test_pred_spo2[spo2_test>70],spo2_test[spo2_test>70],)
axs[2].plot([80,100],[80,100],'r--')
axs[2].text(0.05,0.9,f"r2={r2_spo2:.2f}\nrmse={rmse_spo2:.1f}",transform=axs[2].transAxes,
            bbox={'boxstyle':"round",'facecolor':"wheat",'alpha':0.5})
axs[2].set_ylabel("Observed")
axs[2].set_xlabel("Predicted")
axs[2].set_title("SPO2")
plt.savefig(os.path.join(output_dir,f"Regression plots - {experiment}.png"))
plt.show()


save_regression(model=experiment,rmse=rmse_hr,r2=r2_hr,details='heart rate',
                other=json.dumps({k:v for k,v in hr_clf.best_params_.items() if k != "select__score_func"}))

save_regression(model=experiment,rmse=rmse_rest_rate,r2=r2_rest_rate,details='respiratory rate',
                other=json.dumps({k:v for k,v in resp_rate_clf.best_params_.items() if k != "select__score_func"}))

save_regression(model=experiment,rmse=rmse_spo2,r2=r2_spo2,details='SpO2',
                other=json.dumps({k:v for k,v in spo2_clf.best_params_.items() if k != "select__score_func"}))
