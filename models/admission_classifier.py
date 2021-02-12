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
from sklearn.feature_selection import SelectKBest,f_classif,mutual_info_classif,SelectPercentile,VarianceThreshold,RFECV
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV,RepeatedStratifiedKFold
from settings import data_dir,weights_dir
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import save_table3


# rng=np.random.RandomState(123)
cores=multiprocessing.cpu_count()-2
experiment="Contrastive-sample-DotProduct32b"
weights_file=os.path.join(weights_dir,f"Classification_{experiment}.joblib")
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")


classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)

train_ids=train['id'].unique()
test_ids=test['id'].unique()
classifier_embedding_reduced=np.stack(map(lambda id:classifier_embedding[train['id']==id,:].mean(axis=0) ,train_ids))
test_embedding_reduced=np.stack(map(lambda id:test_embedding[test['id']==id,:].mean(axis=0) ,test_ids))
admitted_train=np.stack(map(lambda id:train.loc[train['id']==id,'admitted'].iat[0],train_ids))
admitted_test=np.stack(map(lambda id:test.loc[test['id']==id,'admitted'].iat[0],test_ids))


base_clf=SGDClassifier(loss='modified_huber',
                       class_weight='balanced',
                       penalty='l2',
                       early_stopping=False,
#                        validation_fraction=0.05,n_iter_no_change=20,
                       max_iter=100,random_state=123)
                       # n_iter_no_change=20,
# base_clf=LogisticRegression(
#     # penalty='elasticnet',
#     max_iter=500000,
#     random_state=56,
#     # solver='saga',
#     class_weight='balanced')




grid_parameters = {
    # 'clf__C': [1.0,5e-1,1e-1,5e-2,1e-2,1e-3,1e-4],
    # 'clf__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],


    'clf__alpha': [1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0],
    'clf__eta0': [0.00001,0.0001,0.001,0.01,.1,1.0],
#     'clf__max_iter':[5,10,50,100,200,500],
#     'clf__loss': ['modified_huber'],
#     'clf__learning_rate': [ 'adaptive',],
    'poly__degree': [2, ],

    'poly__interaction_only': [False,],
    'select__percentile': [10, 15, 20, 30, 40, 60, 70,100],
#     'select__score_func': [mutual_info_classif, ],
    # 'clf__l1_ratio': [0.1, 0.3, 0.5, 0.8, 1.0],

}

pipeline = Pipeline([
    ('variance_threshold',VarianceThreshold()),
    ('poly', PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)),
    ('select', SelectPercentile(mutual_info_classif)),
    ('scl', StandardScaler()),
    ('clf', base_clf),
])

clf = GridSearchCV(pipeline, param_grid=grid_parameters, cv=StratifiedKFold(50 ,random_state=123),
                   verbose=1, n_jobs=cores,#n_iter=500,
                   scoring=[ 'balanced_accuracy','roc_auc','f1', 'recall', 'precision'], refit='roc_auc',
                   return_train_score=True,
                   )
#  
clf.fit(classifier_embedding_reduced,admitted_train)



cv_results=pd.DataFrame({'params':clf.cv_results_['params'], 'auc':clf.cv_results_['mean_test_roc_auc'],
              'acc':clf.cv_results_['mean_test_balanced_accuracy'],'recall':clf.cv_results_['mean_test_recall'],
                          'precision':clf.cv_results_['mean_test_precision'],
                         'f1':clf.cv_results_['mean_test_f1']})
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

report=classification_report(final_predictions2['admitted'],(final_predictions2['prediction']>0.5)*1.0,output_dict=True)
recall=report['1.0']['recall']
precision=report['1.0']['precision']
f1=report['1.0']['f1-score']
specificity=report['0.0']['recall']
acc=report['accuracy']
auc=roc_auc_score(final_predictions2['admitted'],final_predictions2['prediction'])
save_table3(model="Contrastive",precision=precision,recall=recall,specificity=specificity,
            auc=auc,details=experiment,other=json.dumps({'host':os.uname()[1],'f1':f1,
                                                       'acc':acc}))

joblib.dump(clf,weights_file)
