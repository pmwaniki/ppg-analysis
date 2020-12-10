import matplotlib.pyplot as plt
import os
import sys
# sys.setrecursionlimit(100000)
import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from settings import data_dir


experiment="Contrastive-DotProduct32"
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")






classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)





#REGRESSION: RESPIRATORY RATE
y_train=train.loc[:,'resp_rate'].values.reshape(-1,1)
log_y_train=np.log(y_train)
q_transformer=QuantileTransformer(output_distribution="normal",n_quantiles=1000)
q_y_train=q_transformer.fit_transform(y_train)

base_clf_resp=SVR()
tuned_parameters_resp = [
    {'clf__regressor__kernel': ['rbf'], 'clf__regressor__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,1.0,10.0],
#                      'pca__n_components':[int(64/2**i) for i in range(5)],
     'clf__regressor__C': [1, 1e1, 1e2, 1e3,1e4,1e5]
    },
#                     {'clf__regressor__kernel': ['linear'], 'clf__regressor__C': [1, 10, 100, 1000,1e4,1e5],
# #                      'pca__n_components':[int(64/2**i) for i in range(5)],
#                     }
                   ]


pipeline_resp=Pipeline([
    ('scl',StandardScaler()),
#     ('pca',PCA()),
    ('clf',TransformedTargetRegressor(regressor=base_clf_resp,transformer=q_transformer)),
])

clf_resp=RandomizedSearchCV(pipeline_resp,param_distributions=tuned_parameters_resp,
                            cv=KFold(10,shuffle=False),
                 verbose=1,n_jobs=60,n_iter=10,
                 scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],refit='neg_root_mean_squared_error')
# clf_resp.fit(classifier_embedding[~train['resp_rate'].isna()],train.loc[~train['resp_rate'].isna(),'resp_rate'])
clf_resp.fit(classifier_embedding[~train['resp_rate'].isna()],y_train[~train['resp_rate'].isna()])

cv_results_resp=pd.DataFrame({'params':clf_resp.cv_results_['params'],
                              'rmse':clf_resp.cv_results_['mean_test_neg_root_mean_squared_error'],
              'R2':clf_resp.cv_results_['mean_test_explained_variance'],
                              'max_error':clf_resp.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_resp)

test_pred_resp=clf_resp.predict(test_embedding)
r2=r2_score(test.loc[~test['resp_rate'].isna(),'resp_rate'],test_pred_resp[~test['resp_rate'].isna()])
rmse=mean_squared_error(test['resp_rate'],test_pred_resp,squared=False)
