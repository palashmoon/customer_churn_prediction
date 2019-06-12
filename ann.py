# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#create database
dataset = pd.read_csv('Churn_Modelling.csv')

# create the matrix which all the componemts are imp
X = dataset.iloc[: , 3:13].values
y = dataset.iloc[: , 13].values

#categarize karna hai
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:,1] = labelEncoder_X_1.fit_transform(X[: , 1])
labelEncoder_X_2 = LabelEncoder()
X[:,2] = labelEncoder_X_2.fit_transform(X[: , 2])
print(X)
oneHotEncoder = OneHotEncoder(categorical_features=[1])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[: , 1:]    
#split the dataset
#feature scaling
