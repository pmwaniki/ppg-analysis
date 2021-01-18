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
from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV,RepeatedStratifiedKFold
from settings import data_dir,weights_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import save_table3

cores=multiprocessing.cpu_count()-2
experiment="Contrastive-DotProduct32"
weights_file=os.path.join(weights_dir,f"Contrastive_{experiment}_svm.joblib")
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")


classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)

train_ids=train['id'].unique()
test_ids=test['id'].unique()
classifier_embedding_reduced=np.stack(map(lambda id:classifier_embedding[train['id']==id,:].mean(axis=0) ,train_ids))
test_embedding_reduced=np.stack(map(lambda id:test_embedding[test['id']==id,:].mean(axis=0) ,test_ids))
admitted_train=np.stack(map(lambda id:train.loc[train['id']==id,'admitted'].iat[0],train_ids))
admitted_test=np.stack(map(lambda id:test.loc[test['id']==id,'admitted'].iat[0],test_ids))


base_clf=SGDClassifier(loss='modified_huber',
                       # class_weight='balanced',
                       penalty='l2',
                       early_stopping=True,n_iter_no_change=10,max_iter=10000)


tuned_parameters = {
    # 'clf__alpha': (1e-5, 1e-1, 'loguniform'),
    'clf__alpha': scipy.stats.loguniform(1e-5, 1e-1),
    'clf__eta0': scipy.stats.loguniform(1e-5, 1e-1),
    'clf__learning_rate': ['constant', 'adaptive', 'invscaling'],
    'clf__class_weight':['balanced'],#'[{0:1,1:2},{0:1,1:3},{0:1,1:5},{0:1,1:10},{0:1,1:100}]
    # 'clf__l1_ratio': [0.1, 0.3, 0.5, 0.8, 1.0],

}

#full data set
pipeline_full=Pipeline([
    ('scl',RobustScaler()),
#     ('pca',PCA()),
    ('clf',base_clf),
])

clf_full = RandomizedSearchCV(pipeline_full, param_distributions=tuned_parameters, cv=RepeatedStratifiedKFold(10,10 ),
                   verbose=1, n_jobs=cores,n_iter=5000,
                   scoring=[ 'balanced_accuracy','roc_auc','f1', 'recall', 'precision'], refit='roc_auc',
                   return_train_score=True,
                   )
#
clf_full.fit(classifier_embedding_reduced,admitted_train)

cv_results=pd.DataFrame({'params':clf_full.cv_results_['params'], 'auc':clf_full.cv_results_['mean_test_roc_auc'],
              'acc':clf_full.cv_results_['mean_test_balanced_accuracy'],'recall':clf_full.cv_results_['mean_test_recall'],
                          'precision':clf_full.cv_results_['mean_test_precision'],
                         'f1':clf_full.cv_results_['mean_test_f1']})
print(cv_results)

test_pred=clf_full.predict_proba(test_embedding)[:,1]

print(classification_report(test['admitted'],test_pred>0.5))
print("AUC: ",roc_auc_score(test['admitted'],test_pred))

final_predictions=pd.DataFrame({'admitted':test['admitted'],
                                 'id':test['id'],
                                 'prediction':test_pred})
final_predictions2=final_predictions.groupby('id').agg('median')
print(classification_report(final_predictions2['admitted'],(final_predictions2['prediction']>0.5)*1.0))

print("AUC: %.2f" % roc_auc_score(final_predictions2['admitted'],final_predictions2['prediction']))


lr=RandomForestClassifier(class_weight='balanced')
# lr=LogisticRegression(class_weight='balanced',max_iter=10000)
rfe=Pipeline([
    ('scl',StandardScaler()),
#     ('pca',PCA()),
    ('clf',RFECV(lr,step=1,cv=10,scoring='roc_auc',n_jobs=1)),
])

# grid={'clf__estimator__C':[1,10,100,1000]}
grid = {"clf__estimator__max_depth": [3,5,8, None],
        "clf__estimator__n_estimators": [300,500,700],
        "clf__estimator__max_features": [0.5,0.8,1.0],
        # "clf__estimator__min_samples_leaf": [2, 3, 5, 10],
        # "clf__estimator__bootstrap": [True, False],
        # "clf__estimator__criterion": ["gini", "entropy"]
        }
clf_lr=GridSearchCV(rfe,param_grid=grid,cv=10,n_jobs=-1,scoring=[ 'balanced_accuracy','roc_auc','f1', 'recall', 'precision'], refit='roc_auc',)
clf_lr.fit(classifier_embedding_reduced,admitted_train)

test_pred=clf_lr.predict_proba(test_embedding)[:,1]

print(classification_report(test['admitted'],test_pred>0.5))
print("AUC: ",roc_auc_score(test['admitted'],test_pred))

