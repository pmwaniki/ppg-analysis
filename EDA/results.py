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
table2=table.loc[table['model'] != "Contrastive",:].copy()


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

#######################################################################################################################
#Revised with bayesian logistic regression

#####################################################################################################################
table2=table2.loc[~((table2['model']=="Contrastive-Bayesian_clinical") & (table2['details']=="Contrastive-original-sample-DotProduct32")),:].copy()
table2=table2.loc[~((table2['model']=="Contrastive-Bayesian_oximeter") & (table2['details']=="Contrastive-original-sample-DotProduct32")),:].copy()
table2['deep_learning']=table2['model']=="End to end"
table2['type']=table2['deep_learning'].map(lambda x: "Initialization" if x else "Features")

table2['Model']=table2['model'].map(lambda x: "Logistic regression" if x != "End to end" else "Deep learning")

# table2['init']=""
table2['features_init']=np.nan
for i,row in table2.iterrows():
    includes_sepsis="sepsis" in row['details']
    if "Logistic" in row['Model']:
        features=row['model'].replace("Contrastive-Bayesian_","").lower()
        if features == "clinical":
            table2.loc[i,'features_init']="Clinical"
        elif features=="concat_clinical":
            if includes_sepsis:
                table2.loc[i, 'features_init'] = "Clinical & SSL(Labelled & Unlabelled)"
            else:
                table2.loc[i, 'features_init'] = "Clinical & SSL(Labelled)"
        elif features=="ppg":
            if includes_sepsis:
                table2.loc[i, 'features_init'] = "SSL(Labelled & Unlabelled)"
            else:
                table2.loc[i, 'features_init'] = "SSL(Labelled)"
        elif features=="oximeter":
            table2.loc[i, 'features_init'] = "SPO2 & heart rate"
        elif features=="concat_oximeter":
            if includes_sepsis:
                table2.loc[i, 'features_init'] = "SPO2, heart rate & SSL(Labelled & Unlabelled)"
            else:
                table2.loc[i, 'features_init'] = "SPO2, heart rate & SSL(Labelled)"


    elif "End to end" in row['model']:
        details=json.loads(row['details'])
        if details['init'] is None:
            table2.loc[i,'features_init']="Random"
        elif 'sepsis' in details['init']:
            table2.loc[i,'features_init']="SSL(Labelled & Unlabelled)"
        else:
            table2.loc[i, 'features_init'] = "SSL(Labelled)"

table2=table2.rename(columns={'precision':"Precision",'recall':"Sensitivity",
              'specificity':"Specificity",'features_init':"Initialization",'auc':'AUC'})

table2=table2.sort_values(by=["Model",'type',"Initialization"])
table2b=table2[["Model",'type',"Initialization","Precision","Sensitivity","Specificity",'AUC']].set_index(["Model",'type',"Initialization"])

writer = pd.ExcelWriter(os.path.join(output_dir,"Table 1-revised.xlsx"), engine='xlsxwriter')
table2b.to_excel(writer)

workbook  = writer.book
worksheet = writer.sheets['Sheet1']
format = workbook.add_format({'num_format': '0.00'})

worksheet.set_column(0, 0, 19)
worksheet.set_column(1, 1, 15)
worksheet.set_column(2, 2, 35)
worksheet.set_column(3, 3, 10, format)
worksheet.set_column(4, 4, 10, format)
worksheet.set_column(5, 4, 10, format)
worksheet.set_column(6, 6, 10, format)
writer.save()



####################################################################################################################
#
#
#
###################################################################################################################
cursor.execute("SELECT * FROM regression")
rows=cursor.fetchall()

col_names = ['model','rmse','r2','details','other']

regression=pd.DataFrame(rows,columns=col_names)
regression['include']=regression['model'].map(lambda x:"Labelled & unlabelled" if "sepsis" in x else "Labelled")
regression['model_unsupervised']=regression['model'].map(lambda x: "Contrastive learning" if "Contrastive" in x else "PCA")

regression['rmse']=regression['rmse'].map(lambda x:f"{x:.1f}")
regression['r2']=regression['r2'].map(lambda x:f"{x:.2f}")

regression_long=pd.melt(regression,id_vars=['model_unsupervised','details','include'],value_vars=['rmse','r2'])

regression_table=pd.pivot_table(regression_long,values='value',index=['details','variable'],
                                columns=['model_unsupervised','include'],
                                aggfunc=lambda x: ' '.join(x))

regression_table.to_csv(os.path.join(output_dir,"Regression table.csv"))