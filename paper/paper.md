---
Title : House Price Prediction
Date : "Spring 2022"
Authors (San Jose State University) : Prashanth Adapa, Ghanashri Mariyanna, Akshara Narayana, Pratik Dabre
Guidance (San José State University) : Prof. Carlos Rojas
---


## Abstract
Building a machine learning model that can predict the house prices based on various describing attributes. The dataset used is the Ames Housing dataset. It contains of 79 explanatory variables which describe every aspect of residential homes in Ames, Iowa. The data is first cleaned and then imputed in different machine learning models for comparison of their performance. Through the data analysis and observations summarized in this paper, a machine learning model which can effectively predict the housing prices is finalized and used for house price prediction, with the understanding that the algorithm can still be improved using advanced machine learning algorithms.

## Introduction
Every day, thousands of houses are sold but there are questions that every buyer asks, such as: *What is the true value of this house? Is the price I'm paying reasonable?* A machine learning model is proposed in this research to forecast a property price based on data about the house (size, construction details, age, amenities etc.) We present the code used at each stage followed by the results obtained during the construction and evaluation of our model to make our work more reproducible. Python programming language has been utilized in this study, along with a variety of Python packages and interesting libraries.

Our team project focuses on building an effective machine learning model which can reasonably predict the house prices in Ames, Iowa. With the increase and rise of property and real estate prices not only in United States but everywhere in the world, it is vital to conduct this analysis. According to our research, the house prices cannot be dependent on just a few factors but there can be several different features which may overall affect the price of the house. Also, house prices almost never can be predicted effectively using just a few variables. Houses have a variant number of features that may not have the same cost due to its location. For instance, a big house may have a higher price if it is located in desirable rich area than being placed in a poor neighborhood. We strived to identify these significant factors that house in Iowa are dependent on and create a model which is reasonably able to predict the house prices. 
The data used in the experiment will be handled by using a combination of pre-processing methods to improve the prediction accuracy. In addition, some factors will be added to the local dataset in order to study the relationship between these factors and the sale price in Ames.
Different types of machine learning models were used and compared with each other to find the best performing model for this dataset. We were able to identify a model which was capable of identifying the house prices with reasonable accuracy.

## Background
Dean De Cock produced the Ames Housing dataset for use in data science courses. It's a fantastic alternative for data scientists looking for an updated and enhanced version of the well-known Boston Housing dataset. Dataset used can be found here https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The training dataset comprises 1460 observations and 79 explanatory variables, whereas the test dataset has one less variable because the predicted variable, SalePrice, has been removed. To maintain uniformity in pre-processing, both datasets are combined.

## Goals of the Study
The following are the study's primary goals: <br />
- Thoroughly understand the dataset and identify how features interact with each other.
- Build machine learning models capable of predicting property price based on house features.
- Analyze and compare model performance to select the best model

## Methods

### Preliminary findings

- At first glance, we noticed that the dataset has about 76 features ranging from the Garage area, the basement condition, location of the house, amenities, the living area to the Fireplace etc. which determine the salesprice of houses. 
- Examining the data is one of the first and most crucial processes in data analysis. We observed that there are missing values in the dataset based on the count. We have also noticed that the mean and standard deviation differences between variables is large.
- For clear visualization of the missing data, we were also able to identify count of null values in columns by plotting a Missingno graph. Missingno is an excellent tool for quickly visualizing missing values. The tool has great filtering functions to select and arrange the variables we want to plot and it allowed us to customize aspects of the chart.
- To better understand the correlation between columns, we plotted a Dendrogram which is essentially a tree-like graph, that group together the columns that have strong correlations in nullity through hierarchical clustering. So it’s like figuring out which fields are highly related to each other in matters of nullity, then testing how those groups of variables relate to themselves and so on. 
- Again we can notice the discontinued fields, but we can get a more unobstructed view of which variables may be more reliable. The chart illustrates how the groups connect, where connections farther from zero represent combinations of variables that are less similar in nullity.


### Outliers
Outliers are cases that are unusual because they fall outside the distribution that is considered normal for the data. The distance from the center of a normal distribution indicates how typical a given point is with respect to the distribution of the data. The presence of outliers can have a deleterious effect on many forms of data mining. Hence, We are trying to find the outliers or abnormal values of two features ‘GrLivArea’ and ‘LotArea’ with respect to the target variable of our dataset i.e., ‘SalePrice’. We considered analyzing these two features because the sale price of a house is majorly dependent on the area of the house and our dataset consists of these two variables that define the area. We used a scatter plot to visualize the data points and observed the distribution. After careful observation and analysis we defined a range to our features, where all the data points that fall within the range are considered useful to our further analysis and all the points that fall outside the range can be considered outliers and can be safely removed from our dataset. 

### Logarithmic Transformation

- As part of Feature Engineering we are performing 'Logarithm transformation' of skewed target variable - 'SalePrice'
- Log-transformation is a technique used to perform Feature Transformation. It is one of the many techniques that can be used to transform the features so that they are treated equally. This method helps to handle skewed data and after transformation, the distribution becomes more approximate to normal. Log-Tranform method is majorly used to decreases the effect of the outliers, due to the normalization of magnitude differences so that the model becomes more robust.
- Why do we want models to treat them equally? It is because when we input these features to the model, there is a posibillity that a larger value in an imbalance feature will influence the result more and further affect the model performance. This is not something we will want as each and every row of data are equally important as a predictor.
- We wouldn't want the model to prioritize predicting only data with higher sale prices. Hence, scaling and transforming is important for algorithms where distance between the data points is important.
- We picked log-transformation here as it has the power to alter the skewness of a distribution towards normality. We were able to observe how log-transformation of a feature could transform the data distribution and scale.
- We first plotted a distribution plot where we compared the distribution of our target variable with a normal distribution. We were ble to observe that it was right-skewed. After the log transformation, we again plotted a graph with respect to the quantiles of our target feature against the quantiles of a normal distribution. We've just applied a log transformation to the 'SalePrice' variable, reducing its skew and resulting in a more or less regularly distributed variable. 

### Imputation of Null values
- The housing data comprised of both Numerical and Categorical columns and there were null values that needed to be cleaned and imputed in both of them.
- For the ease of handling the data, we first split it in to both numerical and categorical columns.
- We then verified if the data was split appropriately by displaying the data type of each of these columns.
- We wrote a script to loop through the data set individually for categorical and Numerical columns, and displayed the unique values for each of the columns.
- Whichever column contained Nan or null in the Categorical split of the data was imputed with 'None'
- Those columns that contained nan or null in hte Numerical split was either replaced with 0, Mean or Median based on the characteristic of the column, whichever was more appropriate.
- For example: Because the area of each street connected to the house property is most likely similar to the area of other houses in the neighborhood, we may fill in missing numbers by using the community's median LotFrontage. Replacing missing data with 0 in case of GarageType, GarageFinish, GarageQual and GarageCond (Since No garage = no cars in such garage.) etc.
-  Thus we **imputed** the data for selected columns after checking unique values in each of the columns and replacing with those which we felt most accurate. We later extracted data for columns which contained less than or equal to 9 unique values and extracted to a csv file to read and understand the data better for encoding.

### Column-Wise Variance Plotting
- Whenever there are columns in a data frame with only one distinct value, those columns will have zero variance. In fact the reverse is true too; a zero variance column will always have exactly one distinct value. The proof of the former statement follows directly from the definition of variance. The proof of the reverse, however, is based on measure theory - specifically that if the expectation of a non-negative random variable is zero then the random variable is equal to zero.
- The existance of zero variance columns in a data frame seemed benign in predicting house prices.
- We performed variance plotting for all categorical columns to indentify any uneven distribution of data.
- Constant features show similar/single values in all the observations in the dataset. We concluded the features which provide no information that allows ML models to predict the target and dropped them from our dataset.

### Label Encoding
- The extracted unique_vals dataset contains information on the categorical columns and the count of unique values in each of these columns. Using this dataset we were able to plot bar charts and analyse the data. 
- We concluded that few of these categorical columns (which contained less than 9 unique values) can be encoded by label encoder and we converted these columns to numeric.

### Feature Addition
We determined that combining few features to create new features would lead us with qualitative data. We understood that a Sale Price of a house is proportional to age of the house, so we generated a new feature based on house-built year and house sold year to determine the age of the house.

### Data Correlation

### Exploratory Data Analysis
- We used visuals to explore the data in this part. This helped us better comprehend the data and the relationships between variables, allowing us to develop a more accurate model.
Our data set contains 80 columns. It would take a long time to visualize all of the data. Hence we looked in to variables that are strongly associated (both positively and negatively) with our goal variable, "SalePrice." We generated a heatmap of connected data to help us think visually.
From the heat map we were able to conclude that, the garage space, general living area, and overall quality metric are all substantially connected with our goal variable. We then went ahead and visualized each of these column data with our target variable to better our understanding of the relation between them. 

#### From the data visualizations, we were able to conclude that :
- 'SalePrice' and 'GarageArea' have a linear connection so does OverallQual and SalePrice.
- The distribution of 'YearBuilt' is biased towards the year 2000 and has a lengthy tail that stretches till 1900, according to the univariate plot. In the case of newly constructed residences, the linear link between the factors is more obvious.
- GarageYrBlt is similarly substantially negatively linked with the target variable, therefore there was no discernible trend in the data.
- YearRemodAdd has a linear relationship with SalePrice.
- In case of Full bath, there is no trend in data. It is highly negatively correlated with our target variable. But this helps the model to predict better.
- TotalBsmtSF has a strong linear trend and is substantially linked with our objective variable SalePrice
- There's strong link between 'SalePrice' and 'GrLivArea.'

# Comparisons

# Example Analysis

# Conclusions


# References
