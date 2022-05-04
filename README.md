# 1. Project Title: House Price Prediction

## Professor: Carlos Rojas

## Term: Spring 2022

## Team Number: 02

## Team Members:avprashanth Aksh-Narayana pdabre12

1. Ghanashri Mariyanna (https://github.com/Ghanashri-m) <br />
2. Akshara Narayana (https://github.com/Aksh-Narayana) <br />
3. Pratik Dabre (https://github.com/pdabre12) <br />
4. Prashanth Adapa (https://github.com/avprashanth) <br />

## 2. Dataset from Ongoing Kaggle Competition: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## 3. Project Idea Description:

Build a Machine Learning model that, given the various attributes of the house, can predict the price of the house. We’re going to use the Kaggle competition Dataset. It contains the relevant information and covers numerous attributes.
Essentially, the project can be divided into two parts: First part, a deep dive into the data, exploring the dataset we’re going to use and feature engineering. It includes tasks such as cleaning up the data, dealing with the missing values or unknown values, observing on the categorical/non-numerical columns and encoding them with relevant methods, analyzing the size of the dataset and if required, discarding some of the features of the dataset based on their impact, scaling the selected features to be in the similar range to each other so that it does not create any kind of bias towards any feature. It also includes, splitting the dataset into relevant train-test ratio. Having preprocessed data as an output from the first part of the project, the second part is all about training different Machine Learning models, evaluating the trained models with the testing data and few unknown data points. Based on the evaluation metrics, finalizing an algorithmic approach from the set of Machine Learning algorithms we used to create different models. This model can be used to predict the price of any new house based on the relevant input feature it possesses. Data Visualization will be done using various suitable charts/graphs throughout the development phase of the project.

## 4. Potential Approach

We will be using advanced regression models (supervised methods) like <br />

1. Random Forest Model
2. Gradient Boosting Regressor

Will be comparing the results accuracy and apply the best of the 2 models, on the final test data.


## 5. Preliminary Analysis

### 5.1. Extracting, Preprocessing and Curating the dataset: 

Datasets

Train Dataset: This dataset is a collection of data curated from Ames Housing Dataset. This dataset has data with 81 features including the target variable, SalePrice.
Test Dataset: This is the test dataset which has exact same features as in train dataset except for the target variable.

Initially, we merged train dataset and test dataset to attain enough data for us to perform furether processes such as analysis and training the model.

Later, we analyzed both numerical and categorical columns present in our dataset to identify null values. We used dendogram to obtain relationships between columns and to understand their significance with respect to null values. We went on to handle the features which comprised of missing values greater than 80% of their data.

Imputed missing values in categorical columns which contained fewer null values, converted few numerical coulumns as categorical columns. Implemented variance plotting for categorical columns and dropped columns that showed high variance.

### 5.2. Analysis and Feature Addition

We plotted relationships between numerical columns with target variable and we determined that combining few features to create new features would lead us with qualitative data. Example, as we know that a Sale Price of a house is proportional to age of the house, we generated a new feature based on house-built year and house sold year to determine the age of the house. We used scatter plots to analyse few features with respect to Target variable – SalePrice, to detect the outliers. It was observed that few datapoints are very abnormal and would be of now use to our model, hence, we dropped such outliers by defining a meaningful range of values to our features. Plotted graphs for data outlier visualization before and after handling outliers. Plotted a correlation map to analyse how the given features are related to our target value. As part of Feature Engineering, we performed Logarithm transformation of skewed target variable - 'SalePrice'. This method helps to handle skewed data and after transformation, the distribution becomes more approximate to normal. Log-Tranform method is majorly used to decreases the effect of the outliers, due to the normalization of magnitude differences so that the model becomes more robust.

![image](https://user-images.githubusercontent.com/99863530/166649613-f39ffe39-5853-4217-881a-01b0ac4745f9.png)

