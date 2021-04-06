# importing the required libraries
import numpy as np
import pandas as pd 
from pandas_profiling import ProfileReport

# loading the training dataset into the pandas variable
train_data = pd.read_csv('../dataset/Train.csv')

# removing all rows that have more than 3 missing values
train_data.dropna(thresh=8, inplace=True)


# removing all elements with missing Profession, Graduated and Var_1 values
train_data.dropna(subset = ['Profession','Graduated', 'Var_1'], inplace=True)


# resetting the index after removing missing elements
train_data.reset_index(drop=True, inplace=True)


# replacing the missing values in Family_Size with the mode of Family_Size
mode_family = train_data['Family_Size'].mode()
values = {'Family_Size': mode_family.iloc[0]}
train_data.fillna(value=values, inplace=True)


# setting the average age of marriage for men and women
mAge = 26
wAge = 22

# creating numpy arrays for these column attributes
gender = train_data['Gender'].values
age = train_data['Age'].values
married = train_data['Ever_Married'].values

# assigning the missing values
for i in range(0, len(married)):
    if married[i] == 'Yes' or married[i] == 'No':
        continue
    else:
        if gender[i] == 'Male':
            if age[i] > mAge:
                married[i] = 'Yes'
            else:
                married[i] = 'No'
        else:
            if age[i] > wAge:
                married[i] = 'Yes'
            else:
                married[i] = 'No'

train_data['Ever_Married'] = married.tolist()

# deleting the intermediates
del gender, married, mAge, wAge



# importing the random function
import random

work = train_data['Work_Experience'].values

# finding the mix and max values for work experience
minWork = train_data['Work_Experience'].min()
maxWork = train_data['Work_Experience'].max()

for i in range(0,len(work)):
    if minWork <= work[i] <= maxWork:
        continue
    else:
        if age[i] > 50:
            work[i] = maxWork
        elif 40 < age[i] < 50:
            work[i] = random.randint(10,14)
        elif 30 < age[i] < 40:
            work[i] = random.randint(5,9)
        elif 23 < age[i] < 30:
            work[i] = random.randint(0,4)
        else:
            work[i] = minWork

train_data['Work_Experience'] = work.tolist()

# deleting the intermediates
del work, minWork, maxWork, age, random



# setting the attribute ID as the main index
train_data.set_index('ID', inplace=True)

# generating the exploratory analysis page for the training dataset
profile = ProfileReport(train_data, minimal=True, title='Training dataset AFTER analysis')
profile.to_file("training_AFTER.html")

# exporting the dataframe to a new dataset
train_data.to_csv(path_or_buf='../Dataset/Train_After.csv')