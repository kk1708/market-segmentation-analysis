# importing the required libraries
import numpy 
import pandas as pd 
from pandas_profiling import ProfileReport

# loading the training and test dataset into the pandas variable
train_data = pd.read_csv('../dataset/Train.csv')
test_data = pd.read_csv('../dataset/Test.csv')

# generating the exploratory analysis page for the training dataset
profile = ProfileReport(train_data, minimal=True, title='Training dataset BEFORE analysis')
profile.to_file("training_BEFORE.html")

# generating the exploratory analysis page for the test dataset
profile = ProfileReport(test_data, minimal=True, title='Test dataset analysis')
profile.to_file("test.html")