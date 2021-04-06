# importing the required libraries
import numpy 
import pandas as pd 
from pandas_profiling import ProfileReport

# loading the training dataset into the pandas variable
train_data = pd.read_csv('../dataset/Train.csv')

# generating the exploratory analysis page for the training dataset
profile = ProfileReport(train_data, minimal=True, title='Training dataset BEFORE analysis')
profile.to_file("training_BEFORE.html")