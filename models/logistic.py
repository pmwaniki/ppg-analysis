from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
from settings import data_dir
import os

train_ids, test_ids = joblib.load(os.path.join(data_dir, "triage/ids.joblib"))

data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
train = data[data['Study No'].isin(train_ids)]
test = data[data['Study No'].isin(test_ids)]


clf=LogisticRegression()

