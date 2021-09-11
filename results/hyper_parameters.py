import matplotlib.pyplot as plt
import os,shutil
import sqlite3
import json

import pandas as pd
from ray.tune.analysis.experiment_analysis import Analysis
from settings import checkpoint_dir,output_dir,base_dir

display=os.environ.get("DISPLAY",None)

db=sqlite3.connect(os.path.join(base_dir,"results.db"))
cur=db.cursor()
cur.execute("SELECT * FROM table3")

rows=cur.fetchall()

table3=pd.DataFrame(rows,columns=['model','precision','recall','specificity','auc','details','others'])

end_to_end=table3.loc[table3['model']=="End to end",:].reset_index().copy()
end_to_end['init']=end_to_end['details'].map(lambda x:json.loads(x)['init'])


experiment=f"Supervised2-{'original'}-{32}"
experiments=end_to_end['init'].map(lambda x:experiment if x is None else f"{experiment}__{x}" )




log_dirs=os.listdir(os.path.join(checkpoint_dir, "Supervised"))
all_experiments={}
for exp in experiments:
    analysis=Analysis(os.path.join(checkpoint_dir, "Supervised",exp))
    dfs=analysis.trial_dataframes
    incomplete=[k for k,v in dfs.items() if "loss" not in v]
    for i in incomplete:shutil.rmtree(i,ignore_errors=True) #delete trials without data
    analysis = Analysis(os.path.join(checkpoint_dir, "Supervised", exp))
    score = "auc"
    mode = "max"
    best_trial = analysis.get_best_logdir(score, mode)
    best_dir = analysis.get_best_logdir(score, mode)
    best_data = analysis.trial_dataframes[best_trial]
    best_config = analysis.get_best_config(score, mode)
    all_experiments[exp]=best_data


experiments_labels={'Supervised2-original-32__Contrastive-original-sample-DotProduct32-sepsis': 'Self-supervised: Labelled and Unlabelled',
 'Supervised2-original-32__Contrastive-original-sample-DotProduct32': 'Self-supervised: Labelled only',
 'Supervised2-original-32': 'Random initialization'}

experiments_color={'Supervised2-original-32__Contrastive-original-sample-DotProduct32-sepsis': 'blue',
 'Supervised2-original-32__Contrastive-original-sample-DotProduct32': 'green',
 'Supervised2-original-32': 'black'}



fig,ax=plt.subplots(1,2,figsize=(12,10))

for exp,dat in all_experiments.items():
    if dat is None: continue
    ax[0].plot(dat['training_iteration'],dat['auc'],
               c=experiments_color[exp],
               # c="black" if "ens" in exp else "blue",
               label=experiments_labels[exp],
               # linestyle="-" if "ens" in exp else ":",
               linewidth=1)
    ax[0].set_title("AUC")
    ax[0].set_xlabel("Epoch")

    ax[1].plot(dat['training_iteration'], dat['loss'],
               c=experiments_color[exp],
               label=experiments_labels[exp],
               # linestyle="-" if "ens" in exp else ":",
               linewidth=1)
    ax[1].set_title("loss")
    ax[1].set_xlabel("Epoch")

handles, labels = ax[0].get_legend_handles_labels()
unique_labels=[labels.index(x) for x in set(labels)]
plt.legend([handles[i] for i in unique_labels],[labels[i] for i in unique_labels],bbox_to_anchor=(0.5, -0.05),
           ncol=3, fancybox=True, shadow=True,loc="upper right")
plt.legend(handles,labels,bbox_to_anchor=(0.5, -0.05),
           ncol=3, fancybox=True, shadow=True,loc="upper right")
# fig.legend()
plt.savefig(os.path.join(output_dir,"loss and AUC trend.png"))
if display: fig.show()


# self-supervised***************************************************************************************************


#*******************************************************************************************************************
experiments=end_to_end['init'].values
experiments=[e for e in experiments if e]

# log_dirs=os.listdir(os.path.join(checkpoint_dir,))
all_experiments={}
best_configs={}
for exp in experiments:
    analysis=Analysis(os.path.join(checkpoint_dir,exp))
    dfs=analysis.trial_dataframes
    incomplete=[k for k,v in dfs.items() if "loss" not in v]
    for i in incomplete:shutil.rmtree(i,ignore_errors=True) #delete trials without data
    analysis = Analysis(os.path.join(checkpoint_dir, exp))
    score = "accuracy"
    mode = "max"
    best_trial = analysis.get_best_logdir(score, mode)
    best_dir = analysis.get_best_logdir(score, mode)
    best_data = analysis.trial_dataframes[best_trial]
    best_config = analysis.get_best_config(score, mode)
    all_experiments[exp]=best_data
    best_configs[exp]=best_config


experiments_labels={'Contrastive-original-sample-DotProduct32-sepsis': 'Self-supervised: Labelled and Unlabelled',
 'Contrastive-original-sample-DotProduct32': 'Self-supervised: Labelled only'}

experiments_color={'Contrastive-original-sample-DotProduct32-sepsis': 'blue',
 'Contrastive-original-sample-DotProduct32': 'green',
 }



fig,ax=plt.subplots(1,2,figsize=(12,10))

for exp,dat in all_experiments.items():
    if dat is None: continue
    ax[0].plot(dat['training_iteration'],dat['accuracy'],
               c=experiments_color[exp],
               # c="black" if "ens" in exp else "blue",
               label=experiments_labels[exp],
               # linestyle="-" if "ens" in exp else ":",
               linewidth=1)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")

    ax[1].plot(dat['training_iteration'], dat['loss'],
               c=experiments_color[exp],
               label=experiments_labels[exp],
               # linestyle="-" if "ens" in exp else ":",
               linewidth=1)
    ax[1].set_title("loss")
    ax[1].set_xlabel("Epoch")

handles, labels = ax[0].get_legend_handles_labels()
unique_labels=[labels.index(x) for x in set(labels)]
plt.legend([handles[i] for i in unique_labels],[labels[i] for i in unique_labels],bbox_to_anchor=(0.5, -0.05),
           ncol=3, fancybox=True, shadow=True,loc="upper right")
plt.legend(handles,labels,bbox_to_anchor=(0.5, -0.05),
           ncol=3, fancybox=True, shadow=True,loc="upper right")
# fig.legend()
plt.savefig(os.path.join(output_dir,"SSL loss and accuracy trend.png"))
if display: fig.show()

ssl_hyperparameters=pd.DataFrame(best_configs).rename(columns=experiments_labels,
                                                      index={'aug_gaus':"Proportion of signals with Gaussian noise augmentation",
                                                             'aug_num_seg':"Number of slices",
                                                             'aug_prop_seg':"Proportion of signals with signal slicing and permutation augmentation",
                                                             'batch_size':"Batch size",
                                                             'dropout':"Dropout proportion",
                                                             'enc_output_size':"Encoder output size",
                                                             'enc_temp':"NCE temperature",
                                                             'l2':"L2 regularization parameter for convolutional layers",
                                                             'l2_fc':"L2 regularization parameter for fully connected layers",
                                                             'lr':"Learning rate for convolutional layers",
                                                             'lr_fc':"Learning rate for fully connected layers",
                                                             'representation_size':"Representation size"})

ssl_hyperparameters.to_csv(os.path.join(output_dir,"SSL optimal hyper-parameters.csv"))

supervised_hyperparameters=end_to_end.copy()
supervised_hyperparameters['init2']=supervised_hyperparameters['init'].map({
    'Contrastive-original-sample-DotProduct32-sepsis':"SSL: Labelled & Unlabelled",
    'Contrastive-original-sample-DotProduct32':'SSL: Labelled',None:'Random'})
supervised_hyperparameters={row['init2']:json.loads(row['others'])['config'] for i,row in supervised_hyperparameters.iterrows()}
for key in ['lr','dropout','l2','l2_fc',]:
    supervised_hyperparameters={k:{param:value for param,value in v} for k,v in supervised_hyperparameters.items()}

supervised_hyperparameters=supervised_hyperparameters.rename(index={'dropout':"Dropout proportion",
                                         'representation_size':'Representation size',
                                         'batch_size':"Batch size",
                                         'smoothing':"Label smothething parameter",
                                         'lr':"Learning rate for convolutional layers",
                                         'l2':"Weight decay parameter for convolutional layers",
                                         'lr_fc':"Learning rate for fully connected layer",
                                         'l2_fc':"Weight decay for fully connected layer",
                                         'aug_gaus':"Proportion of segments with Gaussian augmentation",
                                         'aug_num_seg':"Number of slices for signal slicing and permutation augmentation",
                                         'aug_prop_seg':"Proportion of signals with slicing and permutation augmentation",
                                         'max_iter':"Number of training iterations"})
#supervised_hyperparameters.to_clipboard()
