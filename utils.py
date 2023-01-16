import os

import warnings
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sqlite3

from sklearn.metrics import confusion_matrix

from settings import base_dir

database_file=os.path.join(base_dir,"results.db")





# Function for saving results


def create_table3():
    conn=sqlite3.connect(database_file)
    c=conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS table3 (
        model TEXT NOT NULL,
        precision NUMERIC,
        recall NUMERIC ,
        specificity NUMERIC ,
        auc NUMERIC,
        details TEXT NOT NULL ,
        other TEXT,
        PRIMARY KEY (model,details)
    )
    """)
    conn.commit()
    conn.close()


def save_table3(model,precision,recall,specificity,auc,details,other=None):
    create_table3()

    conn = sqlite3.connect(database_file)
    c = conn.cursor()
    c.execute("""
                    INSERT OR REPLACE INTO table3 (

                        model,
                        precision,
                        recall,
                        specificity,
                        auc,
                        details,
                        other
                    ) VALUES(?,?,?,?,?,?,?)
                    """, (model, precision, recall,specificity, auc, details, other))

    conn.commit()
    conn.close()



def create_regrssion_table():
    conn=sqlite3.connect(database_file)
    c=conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS regression (
        model TEXT NOT NULL,
        rmse NUMERIC,
        r2 NUMERIC ,
        details TEXT NOT NULL ,
        other TEXT,
        PRIMARY KEY (model,details)
    )
    """)
    conn.commit()
    conn.close()

def save_regression(model,rmse,r2,details,other=None):
    create_regrssion_table()

    conn = sqlite3.connect(database_file)
    c = conn.cursor()
    c.execute("""
                    INSERT OR REPLACE INTO regression (

                        model,
                        rmse,
                        r2,
                        details,
                        other
                    ) VALUES(?,?,?,?,?)
                    """, (model, rmse, r2, details, other))

    conn.commit()
    conn.close()

def admission_confusion_matrix(ytrue,ypred,labels=['Not admitted','Admitted']):
    cf_matrix=confusion_matrix(ytrue,ypred,labels=[0,1])
    cf_matrix_norm=cf_matrix/np.reshape(cf_matrix.sum(axis=1),[-1,1])

    fig,ax=plt.subplots(1,1,figsize=(10,8))
    img=ax.matshow(cf_matrix_norm,cmap=plt.cm.get_cmap("Greys"),vmin=0,vmax=1)
    for i in [0,1,]:
        for j in [0,1,]:
            ax.text(j,i,"%d(%.2f)" % (cf_matrix[i,j],cf_matrix_norm[i,j]),ha='center',fontsize=16,weight='bold',color='blue')
    ax.set_xticks([0,1,])
    ax.set_xticklabels(labels,rotation=45,ha="left",rotation_mode="anchor",fontsize=12)
    # ax.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #      rotation_mode="anchor")
    ax.set_yticks([0, 1,])
    ax.set_yticklabels(labels,rotation=45,fontsize=12)
    # ax.set_ylim(4.5,-0.5)
    ax.set_xlabel("Predicted Class",fontsize=12,fontweight='bold')
    ax.set_ylabel("Target Class",fontsize=12,fontweight='bold')
    fig.colorbar(img)
    fig.tight_layout()
    plt.show()

    return fig

def admission_distplot(samples,ytrue,ypred):
    indices,axes,labels={},{},{}
    indices['tp']=np.where((ytrue==1) & (ypred==1))[0]
    indices['tn'] = np.where((ytrue == 0) & (ypred == 0))[0]
    indices['fp'] = np.where((ytrue == 0) & (ypred == 1))[0]
    indices['fn'] = np.where((ytrue == 1) & (ypred == 0))[0]
    axes['tn']=0
    axes['fp']=1
    axes['fn']=2
    axes['tp']=3
    labels['tn']="True negative"
    labels['fp'] = "False positive"
    labels['fn'] = "False negative"
    labels['tp'] = "True positive"
    fig,axs=plt.subplots(1,4,sharex=True,figsize=(12,4))
    for p in ['tn','fp','fn','tp']:
        axs[axes[p]].set_xlabel("P")
        axs[axes[p]].set_ylabel("Density")
        axs[axes[p]].set_ylim((0,6))
        axs[axes[p]].set_title(labels[p])
        for i in indices[p]:
            sns.kdeplot(samples[:,i].reshape(-1),ax=axs[axes[p]])

    plt.show()
    return fig