import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectPercentile
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

from settings import data_dir
import os
import matplotlib.pyplot as plt

experiment="Contrastive-original-sample-DotProduct32-sepsis"
experiment_file=os.path.join(data_dir,f"results/{experiment}.joblib")
classifier_embedding,test_embedding,seg_train,seg_test=joblib.load(experiment_file)
train_ids=seg_train['id'].unique()
test_ids=seg_test['id'].unique()

classifier_embedding_reduced=np.stack(map(lambda id:classifier_embedding[seg_train['id']==id,:].mean(axis=0) ,train_ids))
test_embedding_reduced=np.stack(map(lambda id:test_embedding[seg_test['id']==id,:].mean(axis=0) ,test_ids))
admitted_train=np.stack(map(lambda id:seg_train.loc[seg_train['id']==id,'admitted'].iat[0],train_ids))
admitted_test=np.stack(map(lambda id:seg_test.loc[seg_test['id']==id,'admitted'].iat[0],test_ids))



data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
# feature engineering
data['spo_transformed'] = 4.314 * np.log10(103.711-data['Oxygen saturation']) - 37.315

train=data.iloc[[np.where(data['Study No']==id)[0][0] for id in train_ids],:].copy()
test=data.iloc[[np.where(data['Study No']==id)[0][0] for id in test_ids],:].copy()





#Imputation
predictors=['Weight (kgs)','Irritable/restlessness ','MUAC (Mid-upper arm circumference) cm',
            'Can drink / breastfeed?','spo_transformed','Temperature (degrees celsius)',
            'Difficulty breathing','Heart rate(HR) ']

predictors_oximeter=['Heart rate(HR) ','spo_transformed',]

for p in predictors:
    if data[p].dtype==np.float:
        median=train[p].median()
        train[p]=train[p].fillna(median)
        test[p] = test[p].fillna(median)
    else:
        majority=train[p].value_counts().index[0]
        train[p]=train[p].fillna(majority)
        test[p] = test[p].fillna(majority)

train['Can drink / breastfeed?']=train['Can drink / breastfeed?'].map({"Yes":"No","No":"Yes"})

train_x=pd.get_dummies(train[predictors],drop_first=True)
test_x=pd.get_dummies(test[predictors],drop_first=True)

# logit_model=smf.logit()
sm_train=train_x.copy()
sm_test=test_x.copy()
sm_train['constant']=1
sm_test['constant']=1
logit_mod = sm.Logit( admitted_train,sm_train,).fit()
print(logit_mod.summary())
pred_test=logit_mod.predict(sm_test)
roc_auc_score(admitted_test,pred_test)
def pred_published(row):
    logit=-3.45 - 0.006 * row['Weight (kgs)']+1.51 * row['Irritable/restlessness _Yes'] -\
          0.03 * row['MUAC (Mid-upper arm circumference) cm']+\
    1.19 * row['Can drink / breastfeed?_Yes'] -0.004 * row['spo_transformed']+\
    0.05* row['Temperature (degrees celsius)']+ 1.15* row['Difficulty breathing_Yes']+0.006 * row['Heart rate(HR) ']
    return 1/(1+np.exp(-logit))

pred_train_published=[pred_published(r) for i,r in train_x.iterrows()]
pred_test_published=[pred_published(r) for i,r in test_x.iterrows()]

published_auc=roc_auc_score(admitted_test,pred_test_published)

# clf=LogisticRegression(max_iter=10000)
# pipeline=Pipeline([
#     # ('scl',StandardScaler()),
#     ('clf',clf)
# ])

def classifier(select_percentile=False):
    base_clf=LogisticRegression(
        penalty='l2',
        max_iter=1000,
        random_state=123,
        solver='lbfgs',
        class_weight='balanced'
    )
    # bagging=BaggingClassifier(base_estimator=base_clf,n_estimators=10,n_jobs=1,random_state=123)
    grid_parameters = {
        'clf__C': [1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6],
        'select__percentile': [20, 30, 40, 60, 70, 100] if select_percentile else [100,],

    }
    pipeline = Pipeline([
        ('variance_threshold', VarianceThreshold()),
        ('select', SelectPercentile(mutual_info_classif)),
        #     ('poly', PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)),

        # ('scl', StandardScaler()),
        ('clf', base_clf),
    ])

    clf = GridSearchCV(pipeline, param_grid=grid_parameters, cv=StratifiedKFold(10, random_state=123, shuffle=True),
                       verbose=1, n_jobs=-1,  # n_iter=500,
                       scoring=['balanced_accuracy', 'roc_auc', 'f1', 'recall', 'precision'],
                       refit='roc_auc',
                       return_train_score=True,
                       )
    return clf

clf=classifier()
clf.fit(train_x,admitted_train)

test_pred=clf.predict_proba(test_x)
test_auc=roc_auc_score(admitted_test,test_pred[:,1])