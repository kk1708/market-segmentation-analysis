# Market Segmentation
An analysis of customers and predicting the class to which they belong too

## Overview
- Created a machine learning model that helps a company classify their customers into 4 segments - (A, B, C, D)
- Performed exploratory analysis on the dataset provided by the company to observe the relationship between each attributes
- By extracting features and assigning labels, the model was trained using the following algorithms
  - `K-Nearest Neighbours`
  - `Gradient Boosting Classifier`
  - `Random Forest Classifier`
- Using GridSearchCV the models were optimised
- A classification report for each model was also generated

## Resources
**Dataset:** https://www.kaggle.com/vetrirah/customer

**Python Version:** 3.91

**Python Libraries Used:** `numpy`, `pandas`, `sci-kit learn`, `pandas-profiling`, `joblib`, `seaborn`

## Data Cleaning

The dataset contained quite a few missing values that had to be taken care of. The following processes were done on the dataset to fill up missing values:
- Since certain attributes contained very few missing data values, those rows containing those values were removed
- The missing values in the family size attribute was replaced by the mode of the family size attribute
- The married attribute was filled based on the [average marriage age](https://en.wikipedia.org/wiki/List_of_countries_by_age_at_first_marriage)
- The work experience attribute was filled based on a conditional statement of their respective ages

## Exploratory Analysis
An exploratory analysis was done on the dataset to gather some preliminary observations regarding the attributes and how they interact with each other. The following are a few observations:

<img src="https://github.com/kk1708/market-segmentation-analysis/blob/main/images/overview.png" height="390" width="1149">
<img src="https://github.com/kk1708/market-segmentation-analysis/blob/main/images/segmentation%20details.png" height="390" width="1149">
<img src="https://github.com/kk1708/market-segmentation-analysis/blob/main/images/work_experience%20histogram.png" height="417" width="650">


[Click here for the complete EDA](https://github.com/kk1708/market-segmentation-analysis/blob/main/Exploratory_Analysis/training_AFTER.html)

## Model Building
