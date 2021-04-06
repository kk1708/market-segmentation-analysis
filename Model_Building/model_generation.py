# loading the required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# loading the dataset into a pandas dataframe
dataframe = pd.read_csv('../dataset/Train_After.csv')

# creating a dummy dataframe to convert categorical data into numerical data
dummy_dataframe = pd.get_dummies(dataframe)

# creating the labels and features
y = dataframe['Segmentation'].values
X = dataframe.drop(['Segmentation'], axis=1)

# splitting the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)