---
Title : House Price Prediction
Date : "Spring 2022"
Authors (San Jose State University) : Prashanth Adapa, Ghanashri Mariyanna, Akshara Narayana, Pratik Dabre
Guidance (San José State University) : Prof. Carlos Rojas
---


# Abstract

# Introduction

## Background
Dean De Cock produced the Ames Housing dataset for use in data science courses. It's a fantastic alternative for data scientists looking for an updated and enhanced version of the well-known Boston Housing dataset. Dataset used can be found here https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The training dataset comprises 1460 observations and 79 explanatory variables, whereas the test dataset has one less variable because the predicted variable, SalePrice, has been removed. To maintain uniformity in pre-processing, both datasets are combined.

## Initial Data Analysis
- Examining the data is one of the first and most crucial processes in data analysis. We can observe that there are missing values in the dataset based on the count. We can also see that the mean and standard deviation differences between variables is large.

<img width="1028" alt="Screen Shot 2022-05-13 at 12 02 07 PM" src="https://user-images.githubusercontent.com/39545809/168371039-11ac42b9-8d22-4f72-b422-7b790a714a44.png">

For clear visualization of the missing data, we were also able to identify count of null values in columns by plotting a Missingno graph. 

![download](https://user-images.githubusercontent.com/75163512/168749122-99e5dfed-6a2e-4a7a-9bba-7d8152f65c9c.png)

- To better understand the correlation between columns we plotted a Dendogram which helped us generate a tree-like graph through hierarchical clustering and group together the columns that have strong correlations in nullity.

![download](https://user-images.githubusercontent.com/75163512/168749678-a42faf23-6e9b-4cd9-b2e3-a9e9e3106d6e.png)

- We performed column wise variance plotting and eliminated few columns in which the distribution of data is very uneven.

- We have split our data to both numerical and categorical columns and **imputed** data for selected colummns after checking unique values in each of the columns and replacing with those which we felt most accurate. We later extracted data for columns which contained less than or equal to 9 unique values and extracted to a csv file to read and understand the data better for encoding.

### Label Encoding
- The extracted unique_vals dataset contains information on the categorical columns and the count of unique values in each of these columns. Using this dataset we were able to plot bar charts and analyse the data : 

![image](https://user-images.githubusercontent.com/75163512/168752086-dd612b59-3cbd-44c6-9563-18ee0ec47ac2.png)

- We concluded that few of these categorical columns (which contained less than 9 unique values) can be encoded by label encoder and we converted these columns to numeric.

### Feature Addition

- We plotted relationships between numerical columns with target variable and we determined that combining few features to create new features would lead us with qualitative data. We understood that a Sale Price of a house is proportional to age of the house, so we generated a new feature based on house-built year and house sold year to determine the age of the house. We used scatter plots to analyse few features with respect to Target variable – SalePrice, to detect the outliers. It was observed that few datapoints are very abnormal and would be of now use to our model, hence, we dropped such outliers by defining a meaningful range of values to our features. Plotted graphs for data outlier visualization before and after handling outliers. Plotted a correlation map to analyse how the given features are related to our target value. As part of Feature Engineering, we performed Logarithm transformation of skewed target variable - 'SalePrice'. This method helps to handle skewed data and after transformation, the distribution becomes more approximate to normal. Log-Tranform method is majorly used to decreases the effect of the outliers, due to the normalization of magnitude differences so that the model becomes more robust.

![166649613-f39ffe39-5853-4217-881a-01b0ac4745f9](https://user-images.githubusercontent.com/75163512/168747937-9d5d2045-c4dd-4c81-b048-a2166e050696.png)



# Methods

# Comparisons

# Example Analysis

# Conclusions


# References
