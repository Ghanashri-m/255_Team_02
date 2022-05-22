---
Title : House Price Prediction
Date : "Spring 2022"
Authors (San Jose State University) : Prashanth Adapa, Ghanashri Mariyanna, Akshara Narayana, Pratik Dabre
Guidance (San José State University) : Prof. Carlos Rojas
---


# Abstract

## Introduction
Every day, thousands of homes are sold. There are some questions that every buyer asks himself, such as: What is the true value of this home? Is the price I'm paying reasonable? A machine learning model is proposed in this research to forecast a property price based on data about the house (size, year it was built, etc.). We will show the code used for each stage followed by its results during the construction and evaluation of our model. Our work will be more reproducible as a result of this. The Python programming language will be utilized in this study, along with a variety of Python packages.

## Background
Dean De Cock produced the Ames Housing dataset for use in data science courses. It's a fantastic alternative for data scientists looking for an updated and enhanced version of the well-known Boston Housing dataset. Dataset used can be found here https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The training dataset comprises 1460 observations and 79 explanatory variables, whereas the test dataset has one less variable because the predicted variable, SalePrice, has been removed. To maintain uniformity in pre-processing, both datasets are combined.

## Goals of the Study
The following are the study's primary goals: <br />
• Build machine learning models capable of predicting property price based on house features  <br />
• Analyze and compare model performance to select the best model

## Methods
- The Dataset created has multiple features ranging from the Garage area, the basement condition, the living area to the Fireplace quality for the houses. 
- Examining the data is one of the first and most crucial processes in data analysis. We can observe that there are missing values in the dataset based on the count. We can also see that the mean and standard deviation differences between variables is large.

<img width="1028" alt="Screen Shot 2022-05-13 at 12 02 07 PM" src="https://user-images.githubusercontent.com/39545809/168371039-11ac42b9-8d22-4f72-b422-7b790a714a44.png">

For clear visualization of the missing data, we were also able to identify count of null values in columns by plotting a Missingno graph. 

![download](https://user-images.githubusercontent.com/75163512/168749122-99e5dfed-6a2e-4a7a-9bba-7d8152f65c9c.png)

- To better understand the correlation between columns we plotted a Dendogram which is essentially a tree-like graph, that group together the columns that have strong correlations in nullity through hierarchical clustering.

![download](https://user-images.githubusercontent.com/75163512/168749678-a42faf23-6e9b-4cd9-b2e3-a9e9e3106d6e.png)


### Outliers

### Logarithmic Transformation

As part of Feature Engineering we are performing 'Logarithm transformation' of skewed target variable - 'SalePrice'

Log-transformation is a technique used to perform Feature Transformation. It is one of the many techniques that can be used to transform the features so that they are treated equally. This method helps to handle skewed data and after transformation, the distribution becomes more approximate to normal. Log-Tranform method is majorly used to decreases the effect of the outliers, due to the normalization of magnitude differences so that the model becomes more robust.

Why do we want models to treat them equally? It is because when we input these features to the model, there is a posibillity that a larger value in an imbalance feature will influence the result more and further affect the model performance. This is not something we will want as each and every row of data are equally important as a predictor.

We wouldn't want the model to prioritize predicting only data with higher sale prices. Hence, scaling and transforming is important for algorithms where distance between the data points is important.

We picked log-transformation here as it has the power to alter the skewness of a distribution towards normality. We were able to observe how log-transformation of a feature could transform the data distribution and scale.

We first plotted a distribution plot where we compared the distribution of our target variable with a normal distribution. We were ble to observe that it was right-skewed. After the log transformation, we again plotted a graph with respect to the quantiles of our target feature against the quantiles of a normal distribution. We've just applied a log transformation to the 'SalePrice' variable, reducing its skew and resulting in a more or less regularly distributed variable.

Notice how it changes after we apply log transformation onto our feature.

### Variance

- We performed column wise variance plotting and eliminated few columns in which the distribution of data is very uneven.

### Imputation of Null values
- The housing data comprised of both Numerical and Categorical columns and there were null values that needed to be cleaned and imputed in both of them.

- For the ease of handling the data, we first split it in to both numerical and categorical columns.
- We then verified if the data was split appropriately by displaying the data type of each of these columns.
- We wrote a script to loop through the data set individually for categorical and Numerical columns, and displayed the unique values for each of the columns.
- Whichever column contained Nan or null in the Categorical split of the data was imputed with 'None'
- Those columns that contained nan or null in hte Numerical split was either replaced with 0, Mean or Median based on the characteristic of the column, whichever was more appropriate.
- For example: Because the area of each street connected to the house property is most likely similar to the area of other houses in the neighborhood, we may fill in missing numbers by using the community's median LotFrontage. Replacing missing data with 0 in case of GarageType, GarageFinish, GarageQual and GarageCond (Since No garage = no cars in such garage.) etc.
-  Thus we **imputed** the data for selected columns after checking unique values in each of the columns and replacing with those which we felt most accurate. We later extracted data for columns which contained less than or equal to 9 unique values and extracted to a csv file to read and understand the data better for encoding.

### Label Encoding
- The extracted unique_vals dataset contains information on the categorical columns and the count of unique values in each of these columns. Using this dataset we were able to plot bar charts and analyse the data : 

![image](https://user-images.githubusercontent.com/75163512/168752086-dd612b59-3cbd-44c6-9563-18ee0ec47ac2.png)

- We concluded that few of these categorical columns (which contained less than 9 unique values) can be encoded by label encoder and we converted these columns to numeric.

### Feature Addition

- We plotted relationships between numerical columns with target variable and we determined that combining few features to create new features would lead us with qualitative data. We understood that a Sale Price of a house is proportional to age of the house, so we generated a new feature based on house-built year and house sold year to determine the age of the house. We used scatter plots to analyse few features with respect to Target variable – SalePrice, to detect the outliers. It was observed that few datapoints are very abnormal and would be of now use to our model, hence, we dropped such outliers by defining a meaningful range of values to our features. Plotted graphs for data outlier visualization before and after handling outliers. Plotted a correlation map to analyse how the given features are related to our target value. As part of Feature Engineering, we performed Logarithm transformation of skewed target variable - 'SalePrice'. This method helps to handle skewed data and after transformation, the distribution becomes more approximate to normal. Log-Tranform method is majorly used to decreases the effect of the outliers, due to the normalization of magnitude differences so that the model becomes more robust.

![166649613-f39ffe39-5853-4217-881a-01b0ac4745f9](https://user-images.githubusercontent.com/75163512/168747937-9d5d2045-c4dd-4c81-b048-a2166e050696.png)


# Comparisons

# Example Analysis

# Conclusions


# References
