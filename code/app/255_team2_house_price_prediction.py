# -*- coding: utf-8 -*-
"""255_Team2_House_Price_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bOCee2G5lL9H7BY4sBLp8wG18m6fxnyY

<a href="https://colab.research.google.com/github/Ghanashri-m/CMPE255/blob/prashanth-branch/02_imputation_visualization_analysis_encoding_fadd_avprashanth.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Import all the necessary libraries
"""

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

warnings.filterwarnings("ignore")
# pd.set_option('display.max_columns', None)
sns.set_style('darkgrid')

"""# Load the dataset"""

train_data = pd.read_csv("./data/train_data.csv")
test_data = pd.read_csv("./data/test_data.csv")

# Creating a copy of the dataframe to avoid indices conflicts in future
copy_train = train_data.copy()
copy_test = test_data.copy()
test_dataset = copy_test

copy_train

copy_test

sample_submission= pd.read_csv("./data/sample_submission.csv")

sample_submission.head(5)

"""This means that we have to keep the id. Because it will be included in the finals submission and the target variable is the **Salesprice**."""

# Combining the train and test data for data cleaning purpose

housing_dataset = pd.concat([copy_train, copy_test], axis=0,sort=False)
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

## Identifying the outliers by comparing the features "LotArea" & "GrLivArea" with SalePrice.
"""

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

"""## Logarithmic Transformation"""

from scipy.stats import skew, norm, probplot
y = copy_train['SalePrice']
fig = plt.figure(figsize=(5,5))
sns.distplot(y , fit=norm, color='r');

(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

# QQ-plot
fig = plt.figure(figsize=(5,5))
res = probplot(y, plot=plt)
plt.show()

"""## Plotting displot for the Saleprice"""

y = np.log( copy_train['SalePrice'])

fig = plt.figure(figsize=(5,5))
sns.distplot(y , fit=norm, color='r');
(mu, sigma) = norm.fit(y)
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure(figsize=(5,5))
res = probplot(y, plot=plt)
plt.show()

housing_dataset = pd.concat([copy_train, copy_test], axis=0,sort=False)
housing_dataset = housing_dataset.drop(axis = 1, columns="PoolQC")
housing_dataset = housing_dataset.drop(axis = 1, columns="MiscFeature")
housing_dataset = housing_dataset.drop(axis = 1, columns="Alley")
housing_dataset = housing_dataset.drop(axis = 1, columns="Fence")
housing_dataset

"""## Splitting into numerical and categorical data

"""

categorical_columns = housing_dataset.select_dtypes(include=['object'])
numeric_columns = housing_dataset.select_dtypes(exclude=['object'])

"""Splitting the dataframe into the columns which has numerical and categorical features into object_columns and numeric_columns respectively."""

len(categorical_columns.columns)

len(numeric_columns.columns)

"""In total it can be seen that there are in total 39 and 38 columns which are categorical and numeric respectively."""

categorical_columns.dtypes

numeric_columns.dtypes

null_categorical_counts = categorical_columns.isnull().sum()
print("Null values in each column:\n{}".format(null_categorical_counts))

"""## Imputing Missing Values"""

for (columnName, columnData) in categorical_columns.iteritems():
    print('Column Name : ', columnName)
    print('Column Contents : ', columnData.unique())
    print('Total unique value counts : ', len(columnData.unique()))

"""From the above list of unique values for each column, we pick the columns that have nan values present in them. Essentially these -> (MSZoning, Utilities, Exterior1st, Exterior2nd, MasVnrType, BsmtQual, BsmtExposure, BsmtFinType1, BsmtFinType2, Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, SaleType, BsmtCond)"""

none_categorical_columns = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','FireplaceQu','GarageCond']
categorical_columns[none_categorical_columns] = categorical_columns[none_categorical_columns].fillna('None')

null_categorical_counts = categorical_columns.isnull().sum()
print("Null values in each column:\n{}".format(null_categorical_counts))

fewmiss_categorical_columns = ['MasVnrType', 'Electrical', 'MSZoning','Utilities','Exterior1st','Exterior2nd','SaleType', 'KitchenQual','Functional']
categorical_columns[fewmiss_categorical_columns] = categorical_columns[fewmiss_categorical_columns].fillna(categorical_columns.mode().iloc[0])

null_categorical_counts = categorical_columns.isnull().sum()
print("Null values in each column:\n{}".format(null_categorical_counts))

count_unique = categorical_columns.nunique()
unique_val = pd.DataFrame(count_unique, columns=["Unique_vals"])

unique_val

unique_df = pd.read_csv("./data/unique_vals.csv")
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

for (columnName, columnData) in numeric_columns.iteritems():
    print('Column Name : ', columnName)
    print('Column Contents : ', columnData.unique())
    print('Total unique value counts : ', len(columnData.unique()))

numeric_columns['GarageCars'] = numeric_columns['GarageCars'].fillna(0)

print(numeric_columns["LotFrontage"].mean())

numeric_columns['LotFrontage'] = numeric_columns['LotFrontage'].fillna(numeric_columns["LotFrontage"].mean())

print(numeric_columns['GarageYrBlt'])

print((numeric_columns["YrSold"]-numeric_columns["YearBuilt"]).median())

numeric_columns['GarageYrBlt'] = numeric_columns['GarageYrBlt'].fillna(numeric_columns['YrSold']-35)

null_numeric_counts = numeric_columns.isnull().sum()
print("Null values in each column:\n{}".format(null_numeric_counts))

fewmiss_numeric_columns = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']
numeric_columns[fewmiss_numeric_columns] = numeric_columns[fewmiss_numeric_columns].fillna(numeric_columns.mode().iloc[0])

numeric_columns.columns

"""# Column-wise Variance Plotting"""

categorical_columns['MSZoning'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['MSZoning'].value_counts()

categorical_columns['Street'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Street'].value_counts()

categorical_columns['LotShape'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['LotShape'].value_counts()

categorical_columns['LandContour'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['LandContour'].value_counts()

categorical_columns['Utilities'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Utilities'].value_counts()

categorical_columns['LotConfig'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['LotConfig'].value_counts()

categorical_columns['LandSlope'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['LandSlope'].value_counts()

categorical_columns['Neighborhood'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Neighborhood'].value_counts()

categorical_columns['Condition1'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Condition1'].value_counts()

categorical_columns['Condition2'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Condition2'].value_counts()

categorical_columns['BldgType'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['BldgType'].value_counts()

categorical_columns['HouseStyle'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['HouseStyle'].value_counts()

categorical_columns['RoofStyle'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['RoofStyle'].value_counts()

categorical_columns['RoofMatl'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['RoofMatl'].value_counts()

categorical_columns['Exterior1st'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Exterior1st'].value_counts()

categorical_columns['Exterior2nd'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Exterior2nd'].value_counts()

categorical_columns['MasVnrType'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['MasVnrType'].value_counts()

categorical_columns['ExterQual'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['ExterQual'].value_counts()

categorical_columns['ExterCond'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['ExterCond'].value_counts()

categorical_columns['Foundation'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Foundation'].value_counts()

categorical_columns['BsmtQual'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['BsmtQual'].value_counts()

categorical_columns['BsmtCond'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['BsmtCond'].value_counts()

categorical_columns['BsmtExposure'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['BsmtExposure'].value_counts()

categorical_columns['BsmtFinType1'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['BsmtFinType1'].value_counts()

categorical_columns['Heating'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Heating'].value_counts()

categorical_columns['HeatingQC'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['HeatingQC'].value_counts()

categorical_columns['CentralAir'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['CentralAir'].value_counts()

categorical_columns['Electrical'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Electrical'].value_counts()

categorical_columns['KitchenQual'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['KitchenQual'].value_counts()

categorical_columns['Functional'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['Functional'].value_counts()

categorical_columns['FireplaceQu'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['FireplaceQu'].value_counts()

categorical_columns['GarageType'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['GarageType'].value_counts()

categorical_columns['GarageFinish'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['GarageFinish'].value_counts()

categorical_columns['GarageQual'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['GarageQual'].value_counts()

categorical_columns['GarageCond'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['GarageCond'].value_counts()

categorical_columns['PavedDrive'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['PavedDrive'].value_counts()

categorical_columns['SaleType'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['SaleType'].value_counts()

categorical_columns['SaleCondition'].value_counts().plot(kind='bar',figsize=[7,7])
categorical_columns['SaleCondition'].value_counts()

numeric_columns['BsmtFullBath'].value_counts().plot(kind='bar',figsize=[7,7])
numeric_columns['BsmtFullBath'].value_counts()

numeric_columns['BsmtHalfBath'].value_counts().plot(kind='bar',figsize=[7,7])
numeric_columns['BsmtHalfBath'].value_counts()

numeric_columns['GarageCars'].value_counts().plot(kind='bar',figsize=[7,7])
numeric_columns['GarageCars'].value_counts()

"""After looking at the above variance for all the categorical columns. I have decided to remove RoofMatl,Condition2, Utilities, Street and Heating column becuase the distribution of data is very uneven in those columns."""

categorical_columns = categorical_columns.drop(['Utilities','Condition2','Heating','Utilities','RoofMatl','Street'], axis=1)

"""### Combining the numeric and categorical dataframes
In the above functions, We have handled numeric and categorical features of our dataset separately. Now, we chose to combine these two data frames for further analysis.
"""

combined_data = pd.concat([categorical_columns, numeric_columns], axis=1, sort=False)
combined_data

"""# Feature Addition

##### There are few features in our dataset which can be combined to create new features, that might be usefeul to our models in our overall prediction. Hence, we decided to observe, analyse few such features to understand their relationship with Target Variable and other features of the dataset to generate new features.
"""

import plotly.express as px
fig = px.histogram(copy_train, x='SalePrice',color='YrSold',marginal="box") 
fig.show()

import plotly.express as px
fig = px.histogram(copy_train, x='SalePrice',color='YearBuilt',marginal="box") 
fig.show()

"""#### After analysing the two features YrSold and YearBuilt, we decided to create a new feature "Age_of_house" to determine the age of the houses in our dataset. This will be useful to our analysis and prediction as the Sale Price of a house majorly depends on age and lifespan."""

numeric_columns["Age_of_house"] = numeric_columns['YrSold'] - numeric_columns['YearBuilt']

numeric_columns['Age_of_house'].describe()

import plotly.express as px
fig = px.box(numeric_columns, y="SalePrice",points="all",color="Age_of_house")   # Boxplot based on SalePrice for Age of house
fig.show()

"""Negative minimum value is identified for age of house, which is not appropriate. Hence we need to look where the negative value is found."""

find_neg = numeric_columns[numeric_columns['Age_of_house'] < 0]

find_neg

"""The remodelling date of the house is found to be 2009. So we will change the year sold date to 2009"""

numeric_columns.loc[numeric_columns['YrSold']< numeric_columns['YearBuilt'], 'YrSold'] = 2009
numeric_columns['Age_of_house'] = numeric_columns['YrSold'] - numeric_columns['YearBuilt']
numeric_columns['Age_of_house'].describe()

"""#### Now we look at few other features such as GarageArea, TotalBsmtSF, 1stFlrSF and 2ndFlrSF."""

import plotly.express as px
fig = px.histogram(copy_train, x='SalePrice',color='GarageArea',marginal="rug") 
fig.show()

import plotly.express as px
fig = px.box(copy_train, y="SalePrice",points="all",color="TotalBsmtSF", boxmode='overlay')  
fig.show()

fig = px.density_heatmap(copy_train, x="SalePrice", y="1stFlrSF", nbinsx=20, nbinsy=20, color_continuous_scale="oranges") 
fig.show()

fig = px.density_heatmap(copy_train, x="SalePrice", y="2ndFlrSF", nbinsx=20, nbinsy=20, color_continuous_scale="rainbow")
fig.show()

"""## Data Correleation

Correlation map to see how features are correlated with SalePrice
"""

def correlation_matrix():
    corr_matrix = combined_data.corr()
    plt.subplots(figsize=(10,10))
    sns.color_palette("bright")
    sns.heatmap(corr_matrix, vmax=0.9, square=True, cmap="cubehelix")
correlation_matrix()

"""### Exploratory Data Analysis

We'll use visuals to explore the data in this part. This will help us better comprehend the data and the relationships between variables, allowing us to develop a more accurate model.

Our data set contains 80 columns. It takes a long time to visualize all of the data. We'll look at variables that are strongly associated (both positively and negatively) with our goal variable, "SalePrice.
"""

correlated_matrix = copy_train.corr()
def obtain_correlated_features(correlated_data, threshold):
    feature = []
    value = []
    for i , index in enumerate(correlated_data.index):
        if abs(correlated_data[index]) > threshold:
            feature.append(index)
            value.append(correlated_data[index])
    df2 = pd.DataFrame(data = value, index=feature, columns=['corr value'] )
    return df2
corr_df = obtain_correlated_features(correlated_matrix['SalePrice'], 0.5)
corr_df

"""All the variables that are substantially linked with our target variable have been obtained. Let's look at the connected data's values now."""

correlated_data = copy_train[corr_df.index]
correlated_data.head()

"""Let's look at the heatmap of connected data to help us think visually."""

fig, ax = plt.subplots(figsize=(7, 7))
sns.heatmap(correlated_data.corr(), annot = True, annot_kws={'size': 12}, square=True, linecolor='w', linewidths=0.1)

"""The garage space, general living area, and overall quality metric are all substantially connected with our goal variable, as shown in the heatmap above.

## Bivariate Analysis

Let's start with GarageArea because it is highly associated with the goal variable.
"""

sns.boxplot(y='SalePrice', x = 'GarageArea', data=copy_train)

"""In the data, we can identify a pattern. The graph above clearly illustrates that 'SalePrice' and 'GarageArea' have a linear connection. With an increase in 'GarageArea,' the 'SalePrice' rises.

Now, because the next variable, 'OverallQual,' is highly connected with our objective variable, let's look at it.
"""

sns.regplot(x='OverallQual', y='SalePrice', data=copy_train, robust=True)

"""OverallQual and SalePrice, obviously, have a linear relationship.

Let's now examine the link between YearBuilt and SalePrice.
"""

sns.jointplot(x='SalePrice', y='YearBuilt', data=copy_train, kind='hex')

"""The univariate and bivariate plots of variables are displayed in the Joint Grid plot. The distribution of 'YearBuilt' is biased towards the year 2000 and has a lengthy tail that stretches till 1900, according to the univariate plot. In the case of newly constructed residences, the linear link between the factors is more obvious.

Next, we analyze GarageYrBlt
"""

sns.relplot(x='SalePrice', y='GarageYrBlt', data=copy_train)

"""As you can see from the plot above, GarageYrBlt is similarly substantially negatively linked with the target variable, therefore there was no discernible trend in the data.

Next, we analyse our next variable ‘YearRemodAdd’.
"""

sns.relplot(x='SalePrice', y='YearRemodAdd', data=copy_train)

"""YearRemodAdd has a linear relationship with SalePrice, as shown in the graph above.

Now we analyse the next variable ‘FullBath’.
"""

sns.jointplot(x='SalePrice', y='FullBath', data=copy_train)

"""In case of Full bath, there is no trend in data. It is highly negatively correlated with our target variable. But this helps the model to predict better.

Next, we analyze TotalBsmtSF.
"""

sns.jointplot(x='SalePrice', y='TotalBsmtSF', data=copy_train, kind='reg')

"""TotalBsmtSF has a strong linear trend and is substantially linked with our objective variable SalePrice.

Next, we analyze GrLivArea with SalePrice.
"""

g = sns.jointplot(x='SalePrice', y='GrLivArea', data=copy_train, kind='kde', color='m')
g.plot_joint(plt.scatter, color='b', s=40, linewidth=1, marker='+' )
g.ax_joint.collections[0].set_alpha(0.3)

"""Despite a few outliers, the graph above shows a strong link between 'SalePrice' and 'GrLivArea.'"""

combined_data = pd.concat([categorical_columns, numeric_columns], axis=1, sort=False)
combined_data

combined_data = combined_data.drop(['SalePrice'],axis=1)
combined_data = combined_data.set_index('Id') # Settting to id because it is required in the submission format
test_dataset = test_dataset.set_index('Id')

combined_data.shape

combined_data.iloc[:1,2]

combined_data

combined_data.isna().sum()

"""## Transforming some numerical variables that are categorical"""

#MSSubClass=The building class
combined_data['MSSubClass'] = combined_data['MSSubClass'].apply(str)

#Changing 'OverallCond' into a categorical variable
combined_data['OverallCond'] = combined_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
combined_data['YrSold'] = combined_data['YrSold'].astype(str)
combined_data['MoSold'] = combined_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# Below we process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(combined_data[c].values)) 
    combined_data[c] = lbl.transform(list(combined_data[c].values))
      
print('Shape all_data: {}'.format(combined_data.shape))

"""### Skewed Features"""

all_numeric_features = combined_data.dtypes[combined_data.dtypes != "object"].index

# Checking the skew of all numerical features
all_skewed_features = combined_data[all_numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :all_skewed_features})
skewness.head(20)

#sns.distplot(skewness, kde=False, color='red', bins=100)
skewness.plot(figsize=(10,10), kind = 'density', color='red')

"""## Box Cox Transformation of (highly) skewed features
We use the scipy function boxcox1p which computes the Box-Cox transformation of  1+x . we need to note that setting  λ=0  is equivalent to log1p used above for the target variable.

"""

skewness = skewness[abs(skewness) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    combined_data[feat] = boxcox1p(combined_data[feat], lam)

"""# Getting dummy categorical features"""

combined_data = pd.get_dummies(combined_data)
print(combined_data.shape)

# ntrain = housing_dataset.shape[0]
# ntest = test_dataset.shape[0]
# copy_train = combined_data[:ntrain]
# copy_test = combined_data[ntrain:]

# combined_data.to_csv('./data/output/output.csv') (commented out on purpose)

"""## Load the cleaned dataset"""

train_data = pd.read_csv("./data/train_data.csv")
test_data = pd.read_csv("./data/test_data.csv")

train_data

test_data

housing_dataset = pd.read_csv("./data/output/output.csv")
housing_dataset

ntrain = train_data.shape[0]
ntest = test_data.shape[0]
copy_train = housing_dataset[:ntrain]
copy_test = housing_dataset[ntrain:]

copy_train

copy_test

#Saving the variable to predict in a different variable
Current_Saleprice = train_data['SalePrice']
Current_Saleprice

"""## Modeling"""

#We start the model analysis and compare the models 
#multi- linear regression
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV,LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import KFold # for repeated K-fold cross validation
from sklearn.model_selection import cross_val_score # score evaluation
from sklearn.metrics import r2_score

#Calculate root mean squared error
#As the range of the values is going to be 0 to Current_Saleprice.max() we divide the rmse by Current_Saleprice.max() 
#so we get the rmse value on a range from 0 to 1 for better analysing purposes.
def root_mean_squared_error(y_pred):
    return np.sqrt(mean_squared_error(y_pred,Current_Saleprice))/Current_Saleprice.max()

# Return root mean square error applied cross validation (Used for training prediction)

#As the range of the values is going to be 0 to Current_Saleprice.max() we divide the cross-validation-score by Current_Saleprice.max() 
#so we get the cross-validation-score value on a range from 0 to 1 for better analysing purposes.

def cross_validation_score(model):
    return np.sqrt(-cross_val_score(model, copy_train, Current_Saleprice, scoring="neg_mean_squared_error", cv=kfolds)).mean()/Current_Saleprice.max()

#Set the number of folds
kfolds= KFold(n_splits = 10,shuffle=True)

"""### Linear regression

Linear regression is one of the most used regression algorithms so , it was a given to use this algorithm .

As there are multiple independent variables and just one target variable , we used multiple regression , where you can predict the value of the target variable using two or more independent variables.
"""

# Initialize linear regression instance
multiple_regression = linear_model.LinearRegression()
#Training the model using the training data
multiple_regression.fit(copy_train,Current_Saleprice)

#Predicting the model on the training data itself
multiple_regression_y_pred = multiple_regression.predict(copy_train)

#root mean square error of the model
multiple_regression_rmse = root_mean_squared_error(multiple_regression_y_pred)

# Return root mean square error applied cross validation (Used for training prediction)
multiple_regression_cross_val_score = cross_validation_score(multiple_regression)

multiple_regression_r2_score = r2_score(Current_Saleprice,multiple_regression_y_pred)

print("R2 score of Multiple Regression Model is : {0}".format(multiple_regression_r2_score))


print("Root mean squared error of multi linear regression model : {0}".format(multiple_regression_rmse))

print("Cross validation score of multi linear regression model : {0}".format(multiple_regression_cross_val_score))

"""### Gradient Boosting Algorithm"""

# Initialize Gradient Boosting Regression instance
gbr = GradientBoostingRegressor(n_estimators=1000)

#Training the model using the training data
gbr.fit(copy_train, Current_Saleprice)

#Predicting the model on the training data itself
gbr_y_pred = gbr.predict(copy_train)

#root mean square error of the model
gbr_rmse = root_mean_squared_error(gbr_y_pred)

# Return root mean square error applied cross validation (Used for training prediction)
gbr_cross_val_score = cross_validation_score(gbr)

gbr_r2_score = r2_score(Current_Saleprice,gbr_y_pred)

print("R2 score of Gradient Boosting Regression Model is : {0}".format(gbr_r2_score))



print("Root mean squared error of Gradient Boosting Regression model : {0}".format(gbr_rmse))

print("Cross validation score of Gradient Boosting Regression model : {0}".format(gbr_cross_val_score))

"""### XGB Regressor 

XGB is an efficient implementation of the gradient boosting algorithm.

It consists of a class of ensemble machine learning models used for regression purposes.
"""

# Initialize XGB Regressor instance
xgbr = XGBRegressor(verbosity=0,n_estimators=1000) 
#Training the model using the training data
xgbr.fit(copy_train, Current_Saleprice)

#Predicting the model on the training data itself
xgbr_y_pred = xgbr.predict(copy_train)

#root mean square error of the model
xgbr_rmse = root_mean_squared_error(xgbr_y_pred)

# Return root mean square error applied cross validation (Used for training prediction)
xgbr_cross_val_score = cross_validation_score(xgbr)

xgbr_r2_score = r2_score(Current_Saleprice,xgbr_y_pred)

print("R2 score of XGB Regression Model is : {0}".format(xgbr_r2_score))


print("Root mean squared error of XGB Regressor model : {0}".format(format(xgbr_rmse,'.5f')))

print("Cross validation score of XGB Regressor model : {0}".format(xgbr_cross_val_score))

"""### Lasso Regression"""

# Initialize lasso regression instance
# Training the model using the training data
lasso_regression = LassoCV().fit(copy_train, Current_Saleprice)

#Predicting the model on the training data itself
lasso_regression_y_pred = lasso_regression.predict(copy_train)

#root mean square error of the model
lasso_regression_rmse = root_mean_squared_error(lasso_regression_y_pred)

# Return root mean square error applied cross validation (Used for training prediction)
lasso_regression_cross_val_score = cross_validation_score(lasso_regression)

lasso_regression_r2_score  = r2_score(Current_Saleprice,lasso_regression_y_pred)

print("R2 score of Lasso Regression Model is : {0}".format(lasso_regression_r2_score))
print("Root mean squared error of Lasso regression model : {0}".format(lasso_regression_rmse))

print("Cross validation score of Lasso regression model : {0}".format(lasso_regression_cross_val_score))

"""### Lasso Regression with Hyperparameters"""

# Initialize lasso regression instance with hyperparameters
# Training the model using the training data
lasso_regression_hp = LassoCV(alphas = [1, 0.5, 0.001, 0.0005]).fit(copy_train, Current_Saleprice)

#Predicting the model on the training data itself
lasso_regression_y_pred_hp = lasso_regression_hp.predict(copy_train)

#root mean square error of the model
lasso_regression_rmse_hp = root_mean_squared_error(lasso_regression_y_pred_hp)

# Return root mean square error applied cross validation (Used for training prediction)
lasso_regression_cross_val_score_hp = cross_validation_score(lasso_regression_hp)

lasso_regression_r2_score_hp  = r2_score(Current_Saleprice,lasso_regression_y_pred_hp)

print("R2 score of Lasso Regression Model with hyperparameters is : {0}".format(lasso_regression_r2_score_hp))
print("Root mean squared error of Lasso regression model with hyperparameters is : {0}".format(lasso_regression_rmse_hp))

print("Cross validation score of Lasso regression model with hyperparameters is : {0}".format(lasso_regression_cross_val_score_hp))

"""### Ridge Regression

"""

# Initialize ridge regression instance
# Training the model using the training data
ridge_regression = RidgeCV().fit(copy_train, Current_Saleprice)

#Predicting the model on the training data itself
ridge_regression_y_pred = ridge_regression.predict(copy_train)

#root mean square error of the model
ridge_regression_rmse = root_mean_squared_error(ridge_regression_y_pred)

# Return root mean square error applied cross validation (Used for training prediction)
ridge_regression_cross_val_score = cross_validation_score(ridge_regression)

ridge_regression_r2_score  = r2_score(Current_Saleprice,ridge_regression_y_pred)

print("R2 score of Ridge Regression Model is : {0}".format(ridge_regression_r2_score))


print("Root mean squared error of Ridge regression model : {0}".format(ridge_regression_rmse))

print("Cross validation score of Ridge regression model : {0}".format(ridge_regression_cross_val_score))

"""### Ridge Regression with Hyperparameters"""

# Initialize ridge regression instance with hyperparameters
# Training the model using the training data
ridge_regression_hp = RidgeCV(alphas = [1, 0.5, 0.1, 0.001, 0.0005]).fit(copy_train, Current_Saleprice)

#Predicting the model on the training data itself
ridge_regression_y_pred_hp = ridge_regression.predict(copy_train)

#root mean square error of the model
ridge_regression_rmse_hp = root_mean_squared_error(ridge_regression_y_pred_hp)

# Return root mean square error applied cross validation (Used for training prediction)
ridge_regression_cross_val_score_hp = cross_validation_score(ridge_regression_hp)

ridge_regression_r2_score_hp  = r2_score(Current_Saleprice,ridge_regression_y_pred_hp)

print("R2 score of Ridge Regression Model with Hyperparameters is : {0}".format(ridge_regression_r2_score_hp))


print("Root mean squared error of Ridge regression model : {0}".format(ridge_regression_rmse_hp))

print("Cross validation score of Ridge regression model : {0}".format(ridge_regression_cross_val_score_hp))

"""### Elastic Net Regression"""

#Training the model using the training data
elastic_net_regression = ElasticNet().fit(copy_train,Current_Saleprice)
#Predicting the model on the training data itself
elastic_net_regression_y_pred = elastic_net_regression.predict(copy_train)


elastic_net_regression_r2_score  = r2_score(Current_Saleprice,elastic_net_regression_y_pred)

print("R2 score of Elastic Net Regression Model is : {0}".format(elastic_net_regression_r2_score))

#root mean square error of the model
elastic_net_regression_rmse = root_mean_squared_error(elastic_net_regression_y_pred)

# Return root mean square error applied cross validation (Used for training prediction)
elastic_net_regression_cross_val_score = cross_validation_score(elastic_net_regression)


print("Root mean squared error of multi linear regression model : {0}".format(elastic_net_regression_rmse))

print("Cross validation score of multi linear regression model : {0}".format(elastic_net_regression_cross_val_score))

"""### Stacking Model"""

# Initialize Stacking model instance
stack_model = StackingCVRegressor(regressors=(multiple_regression, xgbr,elastic_net_regression,
                                              gbr,lasso_regression,ridge_regression),
                                  meta_regressor=xgbr, use_features_in_secondary=True)


#Training the model using the training data
stack_model.fit(copy_train, Current_Saleprice)

#Predicting the model on the training data itself
stack_model_y_pred = stack_model.predict(copy_train)

#root mean square error of the model
stack_model_rmse = root_mean_squared_error(stack_model_y_pred)

# Return root mean square error applied cross validation (Used for training prediction)
stack_model_cross_val_score = cross_validation_score(stack_model)

stack_model_r2_score  = r2_score(Current_Saleprice,stack_model_y_pred)

print("R2 score of Stacking Model is : {0}".format(stack_model_r2_score))


print("Root mean squared error of Stacking model : {0}".format(stack_model_rmse))

print("Cross validation score of Stacking model : {0}".format(stack_model_cross_val_score))

"""## Comparisons:"""

#Function to lower the numbers after decimal point
def lower_decimal(score_list):
    for i in range(len(score_list)):
        score_list[i] = float(format(score_list[i],'.4f'))
        
    return score_list

# A list of all the available models
models = ['Linear Regression','Gradient Boosting Regression',
          'XGBR Regressor','Lasso Regression','Lasso Regression(HP)','Ridge Regression',
          'Ridge Regression (HP)',
          'Elastic Regression','Stacking Model']

#Create a list of the rmse all the available models
models_rmse = [multiple_regression_rmse,
               gbr_rmse,
               xgbr_rmse,
               lasso_regression_rmse,
               lasso_regression_rmse_hp,
               ridge_regression_rmse,
               lasso_regression_rmse_hp,
               elastic_net_regression_rmse,
               stack_model_rmse]
models_rmse = lower_decimal(models_rmse)
models_rmse

#Create a list of the cross_val_score from all the available models

models_cross_val_score =[multiple_regression_cross_val_score,
                        gbr_cross_val_score,
                        xgbr_cross_val_score,
                         lasso_regression_cross_val_score,
                         lasso_regression_cross_val_score_hp,
                         ridge_regression_cross_val_score,
                         ridge_regression_cross_val_score_hp,
                         elastic_net_regression_cross_val_score,
                         stack_model_cross_val_score
                        ]
models_cross_val_score = lower_decimal(models_cross_val_score)
models_cross_val_score

#Create a list of the r2_score from all the available models

models_r2_score = [multiple_regression_r2_score,
                   gbr_r2_score,
                   xgbr_r2_score,
                   lasso_regression_r2_score,
                   lasso_regression_r2_score_hp,
                   ridge_regression_r2_score,
                   ridge_regression_r2_score_hp,
                    elastic_net_regression_r2_score,
                   stack_model_r2_score,
                  ]
models_r2_score = lower_decimal(models_r2_score)
models_r2_score

comparison_table = pd.DataFrame({'Models':models,
                               'rmse':models_rmse,
                               'cross_val_score':models_cross_val_score,
                                'r2_score':models_r2_score})
comparison_table = comparison_table.set_index(['Models'])
comparison_table

final_y_pred = stack_model.predict(copy_test)

output_df = pd.DataFrame({'Id': copy_test.index +1 ,
                   'SalePrice': final_y_pred})
output_df

# Comparison of the models 
# Define Data

model_length = models
rmse = models_rmse
cross_val_score = models_cross_val_score
r2_score = models_r2_score

x_axis = np.arange(len(model_length))

plt.figure(figsize=(22, 7))
# Multi bar Chart

plt.bar(x_axis , rmse, width=0.2, label = 'rmse')
plt.bar(x_axis +0.20, cross_val_score, width=0.2, label = 'cross_val_score')
plt.bar(x_axis +0.20*2, r2_score, width=0.2, label = 'r2_score')

# Xticks

plt.xticks(x_axis,model_length)

# Add legend
plt.title("Performance analysis of all regression models")

plt.legend()

# Display

plt.show()

output_df.to_csv('./data/output/submission.csv', index=False)

