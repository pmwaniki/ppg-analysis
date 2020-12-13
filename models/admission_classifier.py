import matplotlib.pyplot as plt
import os
import sys
import multiprocessing

import joblib
import numpy as np
import pandas as pd
import scipy
import torch

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV
from settings import data_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

cores=multiprocessing.cpu_count()-2
experiment="Contrastive-LpDistance32"
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")


classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)

# base_clf=SGDClassifier(loss='log',class_weight='balanced',penalty='l2',
#                        early_stopping=True,n_iter_no_change=10,max_iter=100000)
#
#
# tuned_parameters = {
#     # 'clf__alpha': (1e-5, 1e-1, 'loguniform'),
#     'clf__alpha': scipy.stats.loguniform(1e-5, 1e-1),
#     'clf__eta0': scipy.stats.loguniform(1e-5, 1e-1),
#     'clf__learning_rate': ['constant', 'adaptive', 'invscaling'],
#     # 'clf__l1_ratio': [0.1, 0.3, 0.5, 0.8, 1.0],
#
# }

#
# base_clf=MLPClassifier(hidden_layer_sizes=(100,100,100,100,100),
#                        early_stopping=True,n_iter_no_change=10,max_iter=100000)
#
# tuned_parameters = {
#     # 'clf__alpha': (1e-5, 1e-1, 'loguniform'),
#     'clf__alpha': scipy.stats.loguniform(1e-5, 1e-1),
#     'clf__learning_rate_init': scipy.stats.loguniform(1e-5, 1e-1),
#     'clf__learning_rate': ['constant', 'adaptive', 'invscaling'],
#     # 'clf__l1_ratio': [0.1, 0.3, 0.5, 0.8, 1.0],
#
# }

base_clf=SVC(probability=False,class_weight='balanced')
tuned_parameters = [
    {'clf__kernel': ['rbf'], 'clf__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
#                      'pca__n_components':[int(64/2**i) for i in range(5)],
     'clf__C': [1, 10, 100, 1000,1e4]
    },
                    {'clf__kernel': ['linear'], 'clf__C': [1, 10, 100, 1000],
#                      'pca__n_components':[int(64/2**i) for i in range(5)],
                    },
                   ]

pipeline=Pipeline([
    ('scl',RobustScaler()),
#     ('pca',PCA()),
    ('clf',base_clf),
])

clf = RandomizedSearchCV(pipeline, param_distributions=tuned_parameters, cv=StratifiedKFold(10, ),
                   verbose=1, n_jobs=cores,n_iter=50,
                   scoring=[ 'balanced_accuracy','roc_auc','f1', 'recall', 'precision'], refit='balanced_accuracy',
                   return_train_score=True,
                   )
results=clf.fit(classifier_embedding,train['admitted'])



cv_results=pd.DataFrame({'params':clf.cv_results_['params'], 'auc':clf.cv_results_['mean_test_roc_auc'],
              'f1':clf.cv_results_['mean_test_f1'],'recall':clf.cv_results_['mean_test_recall'],
                          'precision':clf.cv_results_['mean_test_precision']})
print(cv_results)

test_pred=clf.predict_proba(test_embedding)[:,1]

print(classification_report(test['admitted'],test_pred>0.5))
print("AUC: ",roc_auc_score(test['admitted'],test_pred))

final_predictions=pd.DataFrame({'admitted':test['admitted'],
                                 'id':test['id'],
                                 'prediction':test_pred})
final_predictions2=final_predictions.groupby('id').agg('mean')
print(classification_report(final_predictions2['admitted'],(final_predictions2['prediction']>0.5)*1.0))

print("AUC: %.2f" % roc_auc_score(final_predictions2['admitted'],final_predictions2['prediction']))


