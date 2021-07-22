import os

import warnings
import json
import pandas as pd
import numpy as np

import sqlite3
from settings import base_dir,output_dir

database_file=os.path.join(base_dir,"results.db")

con=sqlite3.connect(database_file)
cursor=con.cursor()

cursor.execute("SELECT * FROM table3")
rows=cursor.fetchall()

col_names = ['model','precision','recall','specificity','auc','details','other']

table=pd.DataFrame(rows,columns=col_names)

table['model']=table['model'].map(lambda x: "Contrastive learning & Logistic Regression" if x=="Contrastive" else x)
table['init']=""

for i,row in table.iterrows():
    if "Logistic" in row['model']:
        if 'sepsis' in row['details']:
            table.loc[i,'init']="Contrastive: Labeled & Unlabeled"
        else:
            table.loc[i,'init']="Contrastive: Labeled"
        continue
    if "End to end" in row['model']:
        details=json.loads(row['details'])
        if details['init'] is None:
            table.loc[i,'init']="Random"
        elif 'sepsis' in details['init']:
            table.loc[i,'init']="Contrastive: Labeled & Unlabeled"
        else:
            table.loc[i, 'init'] = "Contrastive: Labeled"

table=table.rename(columns={'model':"Model",'precision':"Precision",'recall':"Sensitivity",
              'specificity':"Specificity",'init':"Initialization",'auc':'AUC'})

table[["Model","Initialization","Precision","Sensitivity","Specificity",'AUC']]\
    .to_csv(os.path.join(output_dir,"Table 1.csv"),index=False,float_format=lambda x:np.round(x,2))
    # .melt(id_vars=["Model","Initialization"],var_name="Metric")


