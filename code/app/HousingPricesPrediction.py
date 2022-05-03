# -*- coding: utf-8 -*-
"""Housing_Saleprice_Analysis_Ghanashri-checkpoint.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Ghanashri-m/CMPE255/blob/main/code/app/.ipynb_checkpoints/Housing_Saleprice_Analysis_Ghanashri-checkpoint.ipynb

# Import all the necessary libraries
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import catboost as cb

from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from scipy.stats import skew, norm, probplot
from xgboost import XGBRegressor
from mlxtend.regressor import StackingCVRegressor

from scipy import stats
from scipy.stats import norm, skew #for some statistics

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set_style('darkgrid')

"""# Load the dataset"""

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Creating a copy of the dataframe to avoid indices conflicts in future
copy_train = train_data.copy()
copy_test = test_data.copy()

copy_train.head(10)

copy_test.head(10)

sample_submission= pd.read_csv("sample_submission.csv")

sample_submission.head(5)

"""This means that we have to keep the id. Because it will be included in the finals submission and the target variable is the **Salesprice**."""

housing_dataset = copy_train
test_dataset = copy_test

"""# Data Outliers Visualization"""

#Identifying the outliers by comparing the feature "LotArea" with SalePrice.

fig, ax = plt.subplots(figsize=(15, 12))
ax.scatter(x = housing_dataset['LotArea'], y = housing_dataset['SalePrice'], marker = "*", edgecolors = "Green")
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('LotArea', fontsize=13)
plt.show()

#Deleting outliers by defining the range
housing_dataset = housing_dataset.drop(housing_dataset[(housing_dataset['LotArea']>100000) & (housing_dataset['SalePrice']<300000)].index)

#Lets plot the graph again to see if the outliers are removed.
fig, ax = plt.subplots(figsize=(15, 12))
ax.scatter(housing_dataset['LotArea'], housing_dataset['SalePrice'], marker = "*", edgecolors = "Green")
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('LotArea', fontsize=13)
plt.show()

"""# Clean the Data"""

Current_Saleprice = housing_dataset['SalePrice']

# Save the saleprice in a variable for later comparision,
# and since we need to predict it, drop it from the current housing_dataset
housing_dataset = housing_dataset.drop(['SalePrice'],axis=1)

# set index to ID
housing_dataset = housing_dataset.set_index('Id')
test_dataset = test_dataset.set_index('Id')

housing_dataset.shape

list_of_null_values = []

for col in housing_dataset.columns:
    null = housing_dataset[col].isnull().sum()
    test_null = test_dataset[col].isnull().sum()
    if null != 0 or test_null != 0:
        list_of_null_values.append([col,null,test_null])
        
null_dataframe = pd.DataFrame(list_of_null_values,columns=['Feature_Name','Null','Test_Null'])
null_dataframe.set_index('Feature_Name')
null_dataframe['Total_Null'] = null_dataframe['Null'] + null_dataframe['Test_Null']

"""## Missing value visualization"""

print("-------------------------")
print("Total number of columns with null values:")
print(len(null_dataframe))
print("-------------------------")
print("Total number of null values:")
print(null_dataframe['Total_Null'].sum(axis=0))
print("-------------------------")

f, ax = plt.subplots(figsize=(15, 12))
sns.set_palette(sns.color_palette("pastel"))
sns.barplot(data=null_dataframe.sort_values(by='Total_Null',ascending = False), x='Feature_Name',y='Total_Null')

plt.xticks(rotation = 90)
plt.title("Total Nulls in the column")
plt.show()

# Sort the null values
null_dataframe = null_dataframe.sort_values(by=['Total_Null'], ascending = False)

# drop the rows with very minor number of null values for visualization purpose (the rows are kept in the main housing_dataset) 
sorted_null_df = null_dataframe.drop(null_dataframe[null_dataframe['Total_Null'] <= 20].index)

# set figure size
plt.figure(figsize=(15, 12))

# plot polar axis
ax = plt.subplot(111, polar=True)

# remove grid
plt.axis('off')
plt.title("The 18 null columns above that have non negligible values") 

# Set the coordinates limits
upperLimit = 100
lowerLimit = 30

# Compute max and min in the dataset
max = sorted_null_df['Total_Null'].max()

# Let's compute heights: they are a conversion of each item value (Total_Null column value) in those new coordinates
# In our example, 0 in the dataset will be converted to the lowerLimit (10)
# The maximum will be converted to the upperLimit (100)
slope = (max - lowerLimit) / max
heights = slope * sorted_null_df.Total_Null + lowerLimit

# Compute the width of each bar. In total we have 2*Pi = 360°
width = 2*np.pi / len(sorted_null_df.index)

# Compute the angle each bar is centered on:
indexes = list(range(1, len(sorted_null_df.index)+1))
angles = [element * width for element in indexes]
angles

ax.tick_params(axis='x', labelrotation=10)

# Draw bars
bars = ax.bar(
    x=angles, 
    height=heights, 
    width=width, 
    bottom=lowerLimit,
    linewidth=1, 
    edgecolor="white",
    color="#61a4b2"
)

# little space between the bar and the label
labelPadding = 10

# Add labels
for bar, angle, height, label in zip(bars,angles, heights, sorted_null_df["Feature_Name"]):

    # Labels are rotated. Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle)

    # Flip some labels upside down
    alignment = ""
    if angle >= np.pi/2 and angle < 3*np.pi/2:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"

    # Finally add the labels
    ax.text(
        x=angle, 
        y=lowerLimit + bar.get_height() + labelPadding, 
        s=label, 
        ha=alignment, 
        va='center', 
        rotation=rotation, 
        rotation_mode="anchor")

#Correlation map to see how features are correlated with SalePrice
corrmat = housing_dataset.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corrmat, vmax=0.9, square=True)

"""## Imputing Missing Values"""

# Let's combine our test and training dataset together for further processing
Complete_Test_and_Train_Data = pd.concat([housing_dataset,test_dataset],axis=0).reset_index(drop=True)

"""### MSZoning (the zoning classification in general): 
The value 'RL' is by far the most popular (MODE). As a result, we can use 'RL' to fill in any missing data in this column.
"""

Complete_Test_and_Train_Data['MSZoning'] = Complete_Test_and_Train_Data['MSZoning'].fillna(Complete_Test_and_Train_Data['MSZoning'].mode()[0])

"""### LotFrontage : 
Because the area of each street connected to the house property is most likely similar to the area of other houses in the neighborhood, we may fill in missing numbers by using the community's median LotFrontage.
"""

# Sort by neighborhood, then use the median Lot value to fill up the blanks. All of the neighborhood's frontage
Complete_Test_and_Train_Data["LotFrontage"] = Complete_Test_and_Train_Data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

"""### Alley : 
Data description says NA means "no alley access"
"""

Complete_Test_and_Train_Data["Alley"] = Complete_Test_and_Train_Data["Alley"].fillna("None")

"""### Utilities : 
For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
"""

Complete_Test_and_Train_Data = Complete_Test_and_Train_Data.drop(['Utilities'], axis=1)

"""### PoolQC : 
Data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
"""

Complete_Test_and_Train_Data["PoolQC"] = Complete_Test_and_Train_Data["PoolQC"].fillna("None")

"""### MiscFeature : 
Data description says NA means "no misc feature"
"""

Complete_Test_and_Train_Data["MiscFeature"] = Complete_Test_and_Train_Data["MiscFeature"].fillna("None")

"""### Fence : 
Data description says NA means "no fence"
"""

Complete_Test_and_Train_Data["Fence"] = Complete_Test_and_Train_Data["Fence"].fillna("None")

"""### FireplaceQu : 
Data description says NA means "no fireplace"
"""

Complete_Test_and_Train_Data["FireplaceQu"] = Complete_Test_and_Train_Data["FireplaceQu"].fillna("None")

"""### GarageType, GarageFinish, GarageQual and GarageCond : 
Replacing missing data with None
"""

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    Complete_Test_and_Train_Data[col] = Complete_Test_and_Train_Data[col].fillna('None')

"""### GarageYrBlt, GarageArea and GarageCars : 
Replacing missing data with 0 (Since No garage = no cars in such garage.)
"""

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    Complete_Test_and_Train_Data[col] = Complete_Test_and_Train_Data[col].fillna(0)

"""### BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : 
missing values are likely zero for having no basement
"""

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    Complete_Test_and_Train_Data[col] = Complete_Test_and_Train_Data[col].fillna(0)

"""### BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : 
For all these categorical basement-related features, NaN means that there is no basement.
"""

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    Complete_Test_and_Train_Data[col] = Complete_Test_and_Train_Data[col].fillna('None')

"""### MasVnrArea and MasVnrType : 
NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
"""

Complete_Test_and_Train_Data["MasVnrType"] = Complete_Test_and_Train_Data["MasVnrType"].fillna("None")
Complete_Test_and_Train_Data["MasVnrArea"] = Complete_Test_and_Train_Data["MasVnrArea"].fillna(0)

"""### Functional : 
data description says NA means typical
"""

Complete_Test_and_Train_Data["Functional"] = Complete_Test_and_Train_Data["Functional"].fillna("Typ")

"""### Electrical : 
It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
"""

Complete_Test_and_Train_Data['Electrical'] = Complete_Test_and_Train_Data['Electrical'].fillna(Complete_Test_and_Train_Data['Electrical'].mode()[0])

"""### KitchenQual: 
Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
"""

Complete_Test_and_Train_Data['KitchenQual'] = Complete_Test_and_Train_Data['KitchenQual'].fillna(Complete_Test_and_Train_Data['KitchenQual'].mode()[0])

"""### Exterior1st and Exterior2nd :
Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
"""

Complete_Test_and_Train_Data['Exterior1st'] = Complete_Test_and_Train_Data['Exterior1st'].fillna(Complete_Test_and_Train_Data['Exterior1st'].mode()[0])
Complete_Test_and_Train_Data['Exterior2nd'] = Complete_Test_and_Train_Data['Exterior2nd'].fillna(Complete_Test_and_Train_Data['Exterior2nd'].mode()[0])

"""### SaleType : 
Fill in again with most frequent which is "WD"
"""

Complete_Test_and_Train_Data['SaleType'] = Complete_Test_and_Train_Data['SaleType'].fillna(Complete_Test_and_Train_Data['SaleType'].mode()[0])

"""### MSSubClass : 
Na most likely means No building class. We can replace missing values with None
"""

Complete_Test_and_Train_Data['MSSubClass'] = Complete_Test_and_Train_Data['MSSubClass'].fillna("None")

"""### Check if there are any more null values in the data set"""

list_of_null_values = []

for col in Complete_Test_and_Train_Data.columns:
    null = Complete_Test_and_Train_Data[col].isnull().sum()
    if null != 0:
        list_of_null_values.append([col,null])
        
null_dataframe = pd.DataFrame(list_of_null_values,columns=['Feature_Name','Total_Null'])
null_dataframe.set_index('Feature_Name')

print("-------------------------")
print("Total number of columns with null values:")
print(len(null_dataframe))
print("-------------------------")
print("Total number of null values:")
print(null_dataframe['Total_Null'].sum(axis=0))
print("-------------------------")

Complete_Test_and_Train_Data.shape

Complete_Test_and_Train_Data.isnull().sum().sum()

Complete_Test_and_Train_Data.index = Complete_Test_and_Train_Data.index - 1

"""# Feature Engineering
Log-transformation of skewed target variable

Log-transformation is a technique used to perform Feature Transformation. It is one of the many techniques that can be used to transform the features so that they are treated equally.

Why do we want models to treat them equally? It is because when we input these features to the model, there is a posibillity that an larger value in an imbalance feature will influence the result more and further affect the model performance. This is not something we will want as each and every row of data are equally important as a predictor.

We wouldn't want the model to prioritize predicting only data with higher sale prices. Hence, scaling and transforming is important for algorithms where distance between the data points is important.

We picked log-transformation here as it has the power to alter the skewness of a distribution towards normality. You can observe how log-transformation of a feature can transform its distribution and scale.
"""

# Distribution plot
y = copy_train['SalePrice']
fig = plt.figure(figsize=(15,12))
sns.distplot(y , fit=norm);

(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# QQ-plot
fig = plt.figure(figsize=(15,12))
res = probplot(y, plot=plt)
plt.show()

"""The first plot is a distribution plot where we compare the distribution of our target variable with a normal distribution.
We can easily see it is right-skewed.

The Q-Q plot below plots the quantiles of our target feature against the quantiles of a normal distribution.
We can also easily see the skewness in the target feature.

Notice how it changes after we apply log transformation onto our feature.
"""

y = np.log( copy_train['SalePrice'])

fig = plt.figure(figsize=(15,12))
sns.distplot(y , fit=norm);
(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure(figsize=(15,12))
res = probplot(y, plot=plt)
plt.show()

"""#### Transforming some numerical variables that are really categorical"""

#MSSubClass=The building class
Complete_Test_and_Train_Data['MSSubClass'] = Complete_Test_and_Train_Data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
Complete_Test_and_Train_Data['OverallCond'] = Complete_Test_and_Train_Data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
Complete_Test_and_Train_Data['YrSold'] = Complete_Test_and_Train_Data['YrSold'].astype(str)
Complete_Test_and_Train_Data['MoSold'] = Complete_Test_and_Train_Data['MoSold'].astype(str)

"""#### Label Encoding some categorical variables that may contain information in their ordering set"""

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(Complete_Test_and_Train_Data[c].values)) 
    Complete_Test_and_Train_Data[c] = lbl.transform(list(Complete_Test_and_Train_Data[c].values))

# shape        
print('Shape all_data: {}'.format(Complete_Test_and_Train_Data.shape))

"""#### Adding one more important feature
Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house
"""

# Adding total sqfootage feature 
Complete_Test_and_Train_Data['TotalSF'] = Complete_Test_and_Train_Data['TotalBsmtSF'] + Complete_Test_and_Train_Data['1stFlrSF'] + Complete_Test_and_Train_Data['2ndFlrSF']

"""#### Skewed features"""

numeric_feats = Complete_Test_and_Train_Data.dtypes[Complete_Test_and_Train_Data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = Complete_Test_and_Train_Data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

"""#### Box Cox Transformation of (highly) skewed features
We use the scipy function boxcox1p which computes the Box-Cox transformation of  1+x .

Note that setting  λ=0  is equivalent to log1p used above for the target variable.


"""

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    Complete_Test_and_Train_Data[feat] = boxcox1p(Complete_Test_and_Train_Data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])

"""#### Getting dummy categorical features"""

Complete_Test_and_Train_Data = pd.get_dummies(Complete_Test_and_Train_Data)
print(Complete_Test_and_Train_Data.shape)

"""#### Getting the new train and test sets.


"""

ntrain = housing_dataset.shape[0]
ntest = test_dataset.shape[0]
copy_train = Complete_Test_and_Train_Data[:ntrain]
copy_test = Complete_Test_and_Train_Data[ntrain:]

"""# Modeling"""

