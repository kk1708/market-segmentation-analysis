# loading the required libraries
import pandas as pd
import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split, GridSearchCV


# loading the dataset into a pandas dataframe
dataframe = pd.read_csv('../dataset/Train_After.csv')

# creating a dummy dataframe to convert categorical data into numerical data
dummy_dataframe = pd.get_dummies(dataframe)

# creating the labels and features
y = dataframe['Segmentation'].map({'A':0,'B':1,'C':2,'D':3}).values
X = dummy_dataframe.drop(['Segmentation_A', 'Segmentation_B', 'Segmentation_C', 'Segmentation_D','ID'], axis=1)


# splitting the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# random forest classification
from sklearn.ensemble import RandomForestClassifier

# building the random forest classifier model and saving it into a file

# parameters = {'n_estimators': [10, 50, 100, 150, 200, 250, 500], 'max_features': ['auto','sqrt', 'log2'],'max_depth' : [4,5,6,7,8, None], 'criterion' :['gini', 'entropy']}
# rfc=RandomForestClassifier(random_state=42)
# rfc_t = GridSearchCV(rfc, parameters, return_train_score=False)
# rfc_t.fit(X_train, y_train)
# jb.dump(rfc_t, 'models/RFC')


# SVM classification
from sklearn import svm

# building the svm classifier and saving it into a file

# clf = svm.SVC()
# parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[0.1, 1, 10, 20, 100], 'gamma': [1,0.1,0.01]}
# svc = GridSearchCV(clf, parameters, return_train_score=False)
# svc.fit(X_train, y_train)
# jb.dump(svc, 'models/SVM')

svc = jb.load('models/SVM')
rfc = jb.load('models/RFC')