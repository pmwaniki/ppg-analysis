# Analysis of PPG Signals using Self-Supervised Learning

Self-supervised learning is used to extract features from raw PPG signals and initializing end-to-end deep learning models to predict hospitalization given raw PPG signals. Extracted features are used to predict hospitalization using logistic regression. We also relate extracted features to known physiological parameters such as respiratory rate, heart rate and oxygen saturation (SpO2) using linear regression.


The analysis is done using pytorch==1.7.1, pytorch-metric-learning==0.9.96, and hyper-parameter optimization using ray[tune]==1.2.0.
Install packages in *requirements.txt*

## Preparing the data
- Install packages in *requirements.txt* 
- Create directories for data and results as contained in *setting.py*
- Prepare data by runing *datasets/segments.py*. The datasets can be requested from https://doi.org/10.7910/DVN/KQ4DNK, https://doi.org/10.5683/SP2/PHX4C5, and https://doi.org/10.5683/SP2/ZDDFZL

## Self supervised learning
Self-supervised learning model are trained using contrastive learning (*contrastive_resnet.py*) using Noise Contrastive Estimation loss and dot product as the distance metric.

## Regression models
Features extracted using contrastive learning are used as predictors of heart rate, respiratory rate and oxygen saturation (SpO2) in *regression.py*.

## Classification models
Classification of hospitalization using features extracted using self-supervised learning: *admission_classifier.py*

Classification of hospitalization using end-to-end deep learning:*end_to_end.py*


End to end models can either be initialized randomly or using weights of the self-supervised model
