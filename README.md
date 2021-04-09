# Market Segmentation
An analysis of customers and predicting the class to which they belong too

## Overview
- Created a machine learning model that helps a company classify their customers into 4 segments - (A, B, C, D)
- Performed exploratory analysis on the dataset provided by the company to observe the relationship between each attributes
- By extracting features and assigning labels, the model was trained using the following algorithms
  - K-Nearest Neighbours
  - Gradient Boosting Classifier
  - Random Forest Classifier
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

<img src="https://github.com/kk1708/market-segmentation-analysis/blob/main/images/overview.png" height="234" width="690">
<img src="https://github.com/kk1708/market-segmentation-analysis/blob/main/images/segmentation%20details.png" height="234" width="690">
<img src="https://github.com/kk1708/market-segmentation-analysis/blob/main/images/work_experience%20histogram.png" height="333" width="520">


[Click here for the complete EDA](https://github.com/kk1708/market-segmentation-analysis/blob/main/Exploratory_Analysis/training_AFTER.html)

## Model Building

All the categorical variables are converted into numerical variables by using dummy data. The dataset is now split into training set and test set with the test set having a size of 30%.

**The dataset is trained for accuracy.** We want to find the learning model with the highest accuracy. As mentioned above, the learning models used are:
- K-Nearest Neighbours
- Gradient Boosting Classifier
- Random Forest Classifier

The algorithms are selected by observing the pair plot between each attributes
<img src="https://github.com/kk1708/market-segmentation-analysis/blob/main/images/pair%20plot.png">

The complete model selection process can be [found here](https://github.com/kk1708/market-segmentation-analysis/blob/main/Model_Building/model_selection.ipynb)

## Model Performance
The Gradient Boosting algorithm performed the best out of the three on both the training and test dataset. It is found to have a **mean accuracy of 53%.**

The algorithm also performs better for `Segmentation - D`. This is because this label had the highest number of elements in the dataset.

**_Note:_** _The model can be further tuned for better results. As of now, the only tuning done is using GridSearchCV_

The performance of each algorithm along with their classification report can be [found here](https://github.com/kk1708/market-segmentation-analysis/blob/main/Model_Building/model_analysis.ipynb)

## Conclusion
We've managed to train our model to predict the segment - with a decent accuracy - where a potential customer might belong. Based on this information, the company will now have more command over their customer base.
