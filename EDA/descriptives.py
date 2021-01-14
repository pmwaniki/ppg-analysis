from settings import data_dir
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np




triage_data=pd.read_csv(os.path.join(data_dir,"triage/data.csv"))
train_ids,test_ids=joblib.load(os.path.join(data_dir,"triage/ids.joblib"))
triage_data=triage_data[triage_data['Study No'].isin(np.concatenate([test_ids,train_ids]))]

sns.scatterplot(x='Oxygen saturation',y='Heart rate(HR) ',
                hue='Was child admitted (this illness)?', alpha=.5,data=triage_data)
plt.show()

triage_data2=triage_data[['Oxygen saturation','Heart rate(HR) ','Was child admitted (this illness)?']].copy()
triage_data2=triage_data2.dropna()


def marginal_boxplot(a, vertical=False, **kws):
    c=kws['hue']
    if vertical:
        sns.boxplot(y=a,x=c, palette="Set1",order=kws['order'])
    else:
        sns.boxplot(x=a,y=c,palette="Set1",order=kws['order'])

g=sns.JointGrid(x='Oxygen saturation',y='Heart rate(HR) ',data=triage_data2)
g = g.plot_joint(sns.scatterplot, hue=triage_data2['Was child admitted (this illness)?'],
                 palette="Set1",alpha=.5)
g = g.plot_marginals(marginal_boxplot,hue=triage_data2['Was child admitted (this illness)?'],
                     order=None)
_=g.ax_marg_x.set_ylabel("")
_=g.ax_marg_y.set_xlabel("")
plt.show()


sns.boxplot(x="Oxygen saturation",y='Was child admitted (this illness)?',
orient='h',data=triage_data2)
plt.show()


temp_data=triage_data[['Study No','Sex','Age (years)','Fever']].head()
temp_data=temp_data.to_latex(index=False)
with open("/tmp/triage_data.tex","w") as f:
    f.write(temp_data)