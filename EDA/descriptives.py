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


# AGE IN MONTHS
def cal_age_months(age_years,age_months,age_days):
    if np.isnan(age_years) & np.isnan(age_months) & np.isnan(age_days):
        return np.nan
    age_years=age_years if not np.isnan(age_years) else 0.0
    age_months = age_months if not np.isnan(age_months) else 0.0
    age_days = age_days if not np.isnan(age_days) else 0.0
    return age_years*12.0 + age_months + age_days/30


triage_data['age_months']=triage_data.apply(lambda row: cal_age_months(row['Age (years)'],row['Age (months)'],row['Age (Days)']),axis=1)
# triage_data[['Age (years)','Age (months)','Age (Days)','age_months']].head(10)

np.quantile(triage_data['age_months'],q=(0.25,0.5,0.75))


#SQI

triage_data['SQI'].quantile(q=(0.25,0.5,0.75))