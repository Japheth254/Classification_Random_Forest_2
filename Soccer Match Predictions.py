# Predicting home team outcome in all international football matches - A calssification problem
#The dataset comprises of internationalmatches of various countries and their home matches prediction
#The objective of this analysis is to predict the outcome of thir home matches

#Import libraries
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt

#Import dataset
dataset = pd.read_csv('soccer_history.csv')
x = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 8].values

#Encoding categorical data
#Encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
x[:, 0] = labelencoder_X_1.fit_transform(x[:, 0])
labelencoder_X_2 = LabelEncoder()
x[:, 1] = labelencoder_X_2.fit_transform(x[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 0:]
#Encoding the dependednt variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting data into training data and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 0)

"""#feature scaling-hatmonizing data to prevent one from dominating the other
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#Fitting the Random Forest classification to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(x_test)

#Making the confusion matrix - tp predict accuracy of model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
