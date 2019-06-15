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
from  sklearn.model_selection import train_test_split
X_train  , X_test , y_train , y_test = train_test_split(X,y , test_size = 0.20 , random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#part 2 - ANN
import keras
from keras.models import Sequential #this is to intialize our nn
from keras.layers import Dense # this to add layers to our nn

#initlixe the ANN
classifier = Sequential()

#create input lauers and hidden layers
classifier.add(Dense(input_dim = 11 , output_dim = 6 , activation = 'relu'  , init  = 'uniform'))

#create a second hidden layer
classifier.add(Dense(output_dim = 6 , activation='relu' , init = 'uniform'))

#create a 3rd layer which is a output layer
classifier.add(Dense(output_dim = 1 , activation='sigmoid' , init = 'uniform'))

#fitting
classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

#fitting our dataset
classifier.fit(X_train , y_train , batch_size=10 , epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#create a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred , y_test)

#predicting for a particulAr test example
'''
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
'''

new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred > 0.5)

#find accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test , y_pred)
print(accuracy*100)