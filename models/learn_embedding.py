import matplotlib.pyplot as plt
import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.metrics import roc_auc_score, classification_report,r2_score,mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold,RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.compose import TransformedTargetRegressor
from settings import data_dir


experiment="Contrastive-LpDistance32"
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")


classifier_embedding,test_embedding,train,test=joblib.load(experiment_file)





#REGRESSION: RESPIRATORY RATE
y_train=train.loc[:,'resp_rate'].values.reshape(-1,1)
log_y_train=np.log(y_train)
q_transformer=QuantileTransformer(output_distribution="normal",n_quantiles=1000)
q_y_train=q_transformer.fit_transform(y_train)

# base_clf_resp=SVR()
# tuned_parameters_resp = [
# #     {'clf__regressor__kernel': ['rbf'], 'clf__regressor__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1,1.0,10.0],
# # #                      'pca__n_components':[int(64/2**i) for i in range(5)],
# #      'clf__regressor__C': [1, 1e1, 1e2, 1e3,1e4,1e5]
# #     },
#                     {'clf__regressor__kernel': ['linear'], 'clf__regressor__C': [1, 10, 100, 1000,1e4,1e5],
#                      # 'pca__n_components':[int(32/2**i) for i in range(4)],
#                     }
#                    ]

base_clf_resp=GradientBoostingRegressor()
tuned_parameters_resp = {
    # 'pca__n_components':[int(32/2**i) for i in range(5)],
    'clf__regressor__n_estimators':[1000,],
    'clf__regressor__learning_rate':[0.5,0.1,0.01,0.05,0.001],
    'clf__regressor__subsample':[0.8,0.9,1.0],
    'clf__regressor__max_depth':[2,3,5,8]
}

pipeline_resp=Pipeline([
    ('scl',StandardScaler()),
    # ('pca',PCA()),
    ('clf',TransformedTargetRegressor(regressor=base_clf_resp,
                                      transformer=q_transformer,
                                      # func=np.log,
                                      # inverse_func=np.exp
                                      )),
])

clf_resp=RandomizedSearchCV(pipeline_resp,param_distributions=tuned_parameters_resp,
                            cv=KFold(10,shuffle=False),
                 verbose=1,n_jobs=10,n_iter=100,
                 scoring=['explained_variance','neg_root_mean_squared_error','max_error','r2'],
                            refit='neg_root_mean_squared_error',
                            return_train_score=True)
# clf_resp.fit(classifier_embedding[~train['resp_rate'].isna()],train.loc[~train['resp_rate'].isna(),'resp_rate'])
clf_resp.fit(classifier_embedding[~train['resp_rate'].isna()],y_train[~train['resp_rate'].isna()])

cv_results_resp=pd.DataFrame({'params':clf_resp.cv_results_['params'],
                              'rmse':clf_resp.cv_results_['mean_test_neg_root_mean_squared_error'],
'rmse_train':clf_resp.cv_results_['mean_train_neg_root_mean_squared_error'],
              'R2':clf_resp.cv_results_['mean_test_explained_variance'],
                              'max_error':clf_resp.cv_results_['mean_test_max_error'],
                          # 'precision':clf.cv_results_['mean_test_precision']
                              })
print(cv_results_resp)

test_pred_resp=clf_resp.predict(test_embedding)
r2=r2_score(test.loc[~test['resp_rate'].isna(),'resp_rate'],test_pred_resp[~test['resp_rate'].isna()])
rmse=mean_squared_error(test.loc[~test['resp_rate'].isna(),'resp_rate'],test_pred_resp[~test['resp_rate'].isna()],squared=False)



fig2,ax2=plt.subplots(1)
ax2.scatter(test['resp_rate'],test_pred_resp)
plt.show()

fig2,ax2=plt.subplots(1)
ax2.scatter(test['resp_rate'],test_pred_resp)
ax2.plot([20,100],[20,100])
ax2.set_xlabel("Observed")
ax2.set_ylabel("Predicted")
fig2.savefig(f"/home/pmwaniki/Dropbox/tmp/contrastive_resp_rate_{os.uname()[1]}_{experiment}.png")
plt.show()