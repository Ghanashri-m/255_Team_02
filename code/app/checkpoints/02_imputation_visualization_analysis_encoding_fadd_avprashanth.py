# -*- coding: utf-8 -*-
"""02_imputation_visualization_analysis_encoding_fadd_avprashanth.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10h_kmMHlC0oMWNAQT2suiujNwCIXoCLv

# Import all the necessary libraries
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

"""# Load the dataset"""

train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# Creating a copy of the dataframe to avoid indices conflicts in future
copy_train = train_data.copy()
copy_test = test_data.copy()
test_dataset = copy_test

copy_train.head(10)

copy_test.head(10)

sample_submission= pd.read_csv("sample_submission.csv")

sample_submission.head(5)

"""This means that we have to keep the id. Because it will be included in the finals submission and the target variable is the **Salesprice**."""

housing_dataset  = pd.concat([copy_train, copy_test], axis=0,sort=False)
housing_dataset

msno.bar(housing_dataset)

"""The above plot gives us an estimation on the number of null values in each column"""

msno.dendrogram(housing_dataset)

"""The dendrogram plot provides a tree-like graph generated through hierarchical clustering and groups together columns that have strong correlations in nullity.

# Clean Data
"""

def plot_nas(df: pd.DataFrame):
    if df.isnull().sum().sum() != 0:
        plot_width, plot_height = (20,15)
        plt.rcParams['figure.figsize'] = (plot_width,plot_height)
        na_df = (df.isnull().sum() / len(df)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
        missing_data.plot(kind = "bar")
        plt.show()
    else:
        print('No NANs found')
plot_nas(housing_dataset)

"""Percentage of missing values"""

NAN = [(col, housing_dataset[col].isna().mean()*100) for col in housing_dataset]
NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])

NAN = NAN[NAN.percentage > 80]
NAN.sort_values("percentage", ascending=False)

housing_dataset.shape

#From above we can identify that the columns PoolQC, MiscFeature, Alley and Fence contain significant amount of null values and we can get rid of these colums
housing_dataset = housing_dataset.drop(axis = 1, columns="PoolQC")
housing_dataset = housing_dataset.drop(axis = 1, columns="MiscFeature")
housing_dataset = housing_dataset.drop(axis = 1, columns="Alley")
housing_dataset = housing_dataset.drop(axis = 1, columns="Fence")

housing_dataset.shape

"""Reduced shape of the dataframe after removing the columns which has more than 80 percent of null values

# Splitting into numerical and categorical data
"""

object_columns = housing_dataset.select_dtypes(include=['object'])
numeric_columns =housing_dataset.select_dtypes(exclude=['object'])

"""Splitting the dataframe into the columns which has numerical and categorical features into object_columns and numeric_columns respectively."""

len(object_columns.columns)

len(numeric_columns.columns)

"""In total it can be seen that there are in total 39 and 38 columns which are categorical and numeric respectively."""

object_columns.dtypes

numeric_columns.dtypes

null_categorical_counts = object_columns.isnull().sum()
print("Null values in each column:\n{}".format(null_categorical_counts))

"""## Imputing Missing Values"""

print(f"Unique values in MSZoning columns: {housing_dataset['MSZoning'].unique()}, Total unique value count:{len(housing_dataset['MSZoning'].unique())} ")
print(f"Unique values in Utilities columns: {housing_dataset['Utilities'].unique()}, Total unique value count:{len(housing_dataset['Utilities'].unique())} ")
print(f"Unique values in Exterior1st columns: {housing_dataset['Exterior1st'].unique()}, Total unique value count:{len(housing_dataset['Exterior1st'].unique())} ")
print(f"Unique values in Exterior2nd columns: {housing_dataset['Exterior2nd'].unique()}, Total unique value count:{len(housing_dataset['Exterior2nd'].unique())} ")
print(f"Unique values in KitchenQual columns: {housing_dataset['KitchenQual'].unique()}, Total unique value count:{len(housing_dataset['KitchenQual'].unique())} ")
print(f"Unique values in Functional columns: {housing_dataset['Functional'].unique()}, Total unique value count:{len(housing_dataset['Functional'].unique())} ")
print(f"Unique values in SaleType columns: {housing_dataset['SaleType'].unique()}, Total unique value count:{len(housing_dataset['SaleType'].unique())} ")
print(f"Unique values in MasVnrType columns: {housing_dataset['MasVnrType'].unique()}, Total unique value count:{len(housing_dataset['MasVnrType'].unique())} ")
print(f"Unique values in BsmtQual columns: {housing_dataset['BsmtQual'].unique()}, Total unique value count:{len(housing_dataset['BsmtQual'].unique())} ")
print(f"Unique values in BsmtCond columns: {housing_dataset['BsmtCond'].unique()}, Total unique value count:{len(housing_dataset['BsmtCond'].unique())} ")
print(f"Unique values in BsmtExposure columns: {housing_dataset['BsmtExposure'].unique()}, Total unique value count:{len(housing_dataset['BsmtExposure'].unique())} ")
print(f"Unique values in BsmtFinType1 columns: {housing_dataset['BsmtFinType1'].unique()}, Total unique value count:{len(housing_dataset['BsmtFinType1'].unique())} ")
print(f"Unique values in BsmtFinType2 columns: {housing_dataset['BsmtFinType2'].unique()}, Total unique value count:{len(housing_dataset['BsmtFinType2'].unique())} ")
print(f"Unique values in GarageType columns: {housing_dataset['GarageType'].unique()}, Total unique value count:{len(housing_dataset['GarageType'].unique())} ")
print(f"Unique values in GarageFinish columns: {housing_dataset['GarageFinish'].unique()}, Total unique value count:{len(housing_dataset['GarageFinish'].unique())} ")
print(f"Unique values in GarageQual columns: {housing_dataset['GarageQual'].unique()}, Total unique value count:{len(housing_dataset['GarageQual'].unique())} ")
print(f"Unique values in GarageCond columns: {housing_dataset['GarageCond'].unique()}, Total unique value count:{len(housing_dataset['GarageCond'].unique())} ")
print(f"Unique values in Electrical columns: {housing_dataset['Electrical'].unique()}, Total unique value count:{len(housing_dataset['Electrical'].unique())} ")
print(f"Unique values in FireplaceQu columns: {housing_dataset['FireplaceQu'].unique()}, Total unique value count:{len(housing_dataset['FireplaceQu'].unique())} ")

none_categorical_columns = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']
object_columns[none_categorical_columns] = object_columns[none_categorical_columns].fillna('None')

fewmiss_categorical_columns = ['MasVnrType', 'Electrical', 'MSZoning','Utilities','Exterior1st','Exterior2nd','SaleType', 'KitchenQual','Functional']
object_columns[fewmiss_categorical_columns] = object_columns[fewmiss_categorical_columns].fillna(object_columns.mode().iloc[0])

null_categorical_counts = object_columns.isnull().sum()
print("Null values in each column:\n{}".format(null_categorical_counts))

count_unique = object_columns.nunique()
unique_val = pd.DataFrame(count_unique, columns=["Unique_vals"])

unique_val

unique_df = pd.read_csv("unique_vals.csv")
unique_df.head(10)

sns.set_theme(style="whitegrid")
ax = sns.barplot(x="index", y="Unique_vals",data=unique_df)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)

unique_df = unique_df.loc[unique_df['Unique_vals']<=9]
unique_df

sns.set_theme(style="whitegrid")
ax = sns.barplot(x="index", y="Unique_vals",data=unique_df)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)

null_numeric_counts = numeric_columns.isnull().sum()
print("Null values in each column:\n{}".format(null_numeric_counts))

"""LotFrontage: Linear feet of street connected to property"""

print(f"Unique values in MasVnrArea columns: {housing_dataset['MasVnrArea'].unique()}, Total unique value count:{len(housing_dataset['MasVnrArea'].unique())} ")
print(f"Unique values in BsmtFinSF1 columns: {housing_dataset['BsmtFinSF1'].unique()}, Total unique value count:{len(housing_dataset['BsmtFinSF1'].unique())} ")
print(f"Unique values in BsmtFinSF2 columns: {housing_dataset['BsmtFinSF2'].unique()}, Total unique value count:{len(housing_dataset['BsmtFinSF2'].unique())} ")
print(f"Unique values in TotalBsmtSF columns: {housing_dataset['TotalBsmtSF'].unique()}, Total unique value count:{len(housing_dataset['TotalBsmtSF'].unique())} ")
print(f"Unique values in BsmtFullBath columns: {housing_dataset['BsmtFullBath'].unique()}, Total unique value count:{len(housing_dataset['BsmtFullBath'].unique())} ")
print(f"Unique values in BsmtHalfBath columns: {housing_dataset['BsmtHalfBath'].unique()}, Total unique value count:{len(housing_dataset['BsmtHalfBath'].unique())} ")
print(f"Unique values in GarageCars columns: {housing_dataset['GarageCars'].unique()}, Total unique value count:{len(housing_dataset['GarageCars'].unique())} ")
print(f"Unique values in GarageArea columns: {housing_dataset['GarageArea'].unique()}, Total unique value count:{len(housing_dataset['GarageArea'].unique())} ")

numeric_columns['GarageCars'] = numeric_columns['GarageCars'].fillna(0)

print(numeric_columns["LotFrontage"].mean())

numeric_columns['LotFrontage'] = numeric_columns['LotFrontage'].fillna(numeric_columns["LotFrontage"].mean())

print(numeric_columns['GarageYrBlt'])

print((numeric_columns["YrSold"]-numeric_columns["YearBuilt"]).median())

numeric_columns['GarageYrBlt'] = numeric_columns['GarageYrBlt'].fillna(numeric_columns['YrSold']-35)

null_numeric_counts = numeric_columns.isnull().sum()
print("Null values in each column:\n{}".format(null_numeric_counts))

fewmiss_numeric_columns = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']
object_columns[fewmiss_numeric_columns] = numeric_columns[fewmiss_numeric_columns].fillna(numeric_columns.mode().iloc[0])

object_columns.columns



"""# Column-wise Variance Plotting"""

object_columns['MSZoning'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['MSZoning'].value_counts()

object_columns['Street'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Street'].value_counts()

object_columns['LotShape'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['LotShape'].value_counts()

object_columns['LandContour'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['LandContour'].value_counts()

object_columns['Utilities'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Utilities'].value_counts()

object_columns['LotConfig'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['LotConfig'].value_counts()

object_columns['LandSlope'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['LandSlope'].value_counts()

object_columns['Neighborhood'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Neighborhood'].value_counts()

object_columns['Condition1'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Condition1'].value_counts()

object_columns['Condition2'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Condition2'].value_counts()

object_columns['BldgType'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['BldgType'].value_counts()

object_columns['HouseStyle'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['HouseStyle'].value_counts()

object_columns['RoofStyle'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['RoofStyle'].value_counts()

object_columns['RoofMatl'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['RoofMatl'].value_counts()

object_columns['Exterior1st'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Exterior1st'].value_counts()

object_columns['Exterior2nd'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Exterior2nd'].value_counts()

object_columns['MasVnrType'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['MasVnrType'].value_counts()

object_columns['ExterQual'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['ExterQual'].value_counts()

object_columns['ExterCond'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['ExterCond'].value_counts()

object_columns['Foundation'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Foundation'].value_counts()

object_columns['BsmtQual'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['BsmtQual'].value_counts()

object_columns['BsmtCond'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['BsmtCond'].value_counts()

object_columns['BsmtExposure'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['BsmtExposure'].value_counts()

object_columns['BsmtFinType1'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['BsmtFinType1'].value_counts()

object_columns['Heating'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Heating'].value_counts()

object_columns['HeatingQC'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['HeatingQC'].value_counts()

object_columns['CentralAir'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['CentralAir'].value_counts()

object_columns['Electrical'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Electrical'].value_counts()

object_columns['KitchenQual'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['KitchenQual'].value_counts()

object_columns['Functional'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['Functional'].value_counts()

object_columns['FireplaceQu'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['FireplaceQu'].value_counts()

object_columns['GarageType'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['GarageType'].value_counts()

object_columns['GarageFinish'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['GarageFinish'].value_counts()

object_columns['GarageQual'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['GarageQual'].value_counts()

object_columns['GarageCond'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['GarageCond'].value_counts()

object_columns['PavedDrive'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['PavedDrive'].value_counts()

object_columns['SaleType'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['SaleType'].value_counts()

object_columns['SaleCondition'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['SaleCondition'].value_counts()

object_columns['BsmtFullBath'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['BsmtFullBath'].value_counts()

object_columns['BsmtHalfBath'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['BsmtHalfBath'].value_counts()

object_columns['GarageCars'].value_counts().plot(kind='bar',figsize=[7,7])
object_columns['GarageCars'].value_counts()



"""After looking at the above variance for all the categorical columns. I have decided to remove RoofMatl,Condition2, Utilities, Street and Heating column becuase the distribution of data is very uneven in those columns."""

object_columns = object_columns.drop(['Utilities','Condition2','Heating','Utilities','RoofMatl','Street'], axis=1)



"""# Feature Addition"""

numeric_columns["Age_of_house"] = numeric_columns['YrSold'] - numeric_columns['YearBuilt']

numeric_columns['Age_of_house'].describe()

"""Negative minimum value is identified for age of house, which is not appropriate. Hence we need to look where the negative value is found."""

find_neg = numeric_columns[numeric_columns['Age_of_house'] < 0]

find_neg

"""The remodelling date of the house is found to be 2009. So we will change the year sold date to 2009"""

numeric_columns.loc[numeric_columns['YrSold']< numeric_columns['YearBuilt'], 'YrSold'] = 2009
numeric_columns['Age_of_house'] = numeric_columns['YrSold'] - numeric_columns['YearBuilt']
numeric_columns['Age_of_house'].describe()

"""##Identifying the outliers by comparing the features "LotArea" & "GrLivArea" with SalePrice."""

def outlier_visualization(feature):
    plt.figure(figsize=(10, 10))
    plt.scatter(x = housing_dataset[feature], y = housing_dataset['SalePrice'], marker = "*", edgecolors = "Red")
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel(feature, fontsize=13)
    plt.show()

outlier_visualization('LotArea')
outlier_visualization('GrLivArea')

housing_dataset = housing_dataset.drop(housing_dataset[(housing_dataset['LotArea']>100000) & (housing_dataset['SalePrice']<400000)].index)

outlier_visualization('LotArea')

housing_dataset = housing_dataset.drop(housing_dataset[(housing_dataset['GrLivArea']>4000) & (housing_dataset['SalePrice']<300000)].index)
outlier_visualization('GrLivArea')

"""## Data Correleation

Correlation map to see how features are correlated with SalePrice
"""

def correlation_matrix():
    corr_matrix = housing_dataset.corr()
    plt.subplots(figsize=(15,12))
    sns.color_palette("bright")
    sns.heatmap(corr_matrix, vmax=0.9, square=True, cmap="Blues")
correlation_matrix()

housing_dataset = housing_dataset.drop(['SalePrice'],axis=1)
housing_dataset = housing_dataset.set_index('Id') # Settting to id because it is required in the submission format
test_dataset = test_dataset.set_index('Id')

housing_dataset.shape

housing_dataset.iloc[:1,2]

transformed_dataset = pd.concat([object_columns, numeric_columns], axis=1,sort=False)

transformed_dataset

transformed_dataset.isna().sum()

"""## Logarithmic Transformation"""

from scipy.stats import skew, norm, probplot
y = copy_train['SalePrice']
fig = plt.figure(figsize=(10,10))
sns.distplot(y , fit=norm);

(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# QQ-plot
fig = plt.figure(figsize=(10,10))
res = probplot(y, plot=plt)
plt.show()

"""# Plotting displot for the Saleprice"""

y = np.log( copy_train['SalePrice'])

fig = plt.figure(figsize=(10,10))
sns.distplot(y , fit=norm);
(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure(figsize=(10,10))
res = probplot(y, plot=plt)
plt.show()

"""## Transforming some numerical variables that are categorical"""

#MSSubClass=The building class
transformed_dataset['MSSubClass'] = transformed_dataset['MSSubClass'].apply(str)

#Changing 'OverallCond' into a categorical variable
transformed_dataset['OverallCond'] = transformed_dataset['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
transformed_dataset['YrSold'] = transformed_dataset['YrSold'].astype(str)
transformed_dataset['MoSold'] = transformed_dataset['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# Below we process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(transformed_dataset[c].values)) 
    transformed_dataset[c] = lbl.transform(list(transformed_dataset[c].values))
      
print('Shape all_data: {}'.format(transformed_dataset.shape))

"""#Getting dummy categorical features"""

transformed_dataset = pd.get_dummies(transformed_dataset)
print(transformed_dataset.shape)

ntrain = housing_dataset.shape[0]
ntest = test_dataset.shape[0]
copy_train = transformed_dataset[:ntrain]
copy_test = transformed_dataset[ntrain:]