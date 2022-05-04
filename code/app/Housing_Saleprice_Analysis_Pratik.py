#!/usr/bin/env python
# coding: utf-8

# # Import all the necessary libraries


# # Modeling

# In[49]:


#We start the model analysis and compare the models 
#multi- linear regression
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


# ### Multiple regression
# 
# Linear regression is one of the most used regression algorithms so , it was a given to use this algorithm .
# 
# As there are multiple independent variables and just one target variable , we used multiple regression , where you can predict the value of the target variable using two or more independent variables.

# In[50]:


multiple_regression = linear_model.LinearRegression()
multiple_regression.fit(copy_train,Current_Saleprice)


# In[51]:


house_price_prediction_multiple_regression = multiple_regression.predict(copy_test)
house_price_prediction_multiple_regression


# In[52]:


output_df = pd.DataFrame({'Id': copy_test.index,
                   'SalePrice': house_price_prediction_multiple_regression})
output_df


# In[53]:



output_df.to_csv(r'C:\Users\Checkout\Desktop\submission_MultipleRegression1.csv', index=False)


# ### Logistic Regression
# 
# Logistic regression is a statistical model used for analysis of the dataset. It predicts an outcome based on one or more independent variables.

# In[54]:


logistic_regression = LogisticRegression().fit(copy_train, Current_Saleprice)
house_price_prediction_logistic_regression = logistic_regression.predict(copy_test)


# In[55]:


house_price_prediction_logistic_regression


# In[56]:


#plotting the saleprice values in a bar graph

plt.hist(house_price_prediction_logistic_regression,bins=7)


# In[57]:


output_df = pd.DataFrame({'Id': copy_test.index + 4,
                   'SalePrice': house_price_prediction_logistic_regression})
output_df


# In[58]:


output_df.to_csv(r'C:\Users\Checkout\Desktop\submission_LogisticRegression1.csv', index=False)


# ### SVM
# 
# SVM (Support Vector Machine) is used in classification as well as regression purposes.
# SVM is generally used for classification.

# In[59]:


svm_regression = svm.SVR().fit(copy_train,Current_Saleprice)
svm_house_price_prediction = svm_regression.predict(copy_test)


# In[60]:


svm_house_price_prediction


# In[61]:


plt.hist(svm_house_price_prediction,bins=7)


# In[62]:


output_df = pd.DataFrame({'Id': copy_test.index + 4,
                   'SalePrice': svm_house_price_prediction})
output_df


# In[63]:


output_df.to_csv(r'C:\Users\Checkout\Desktop\submission_SVM1.csv', index=False)


# ### KNN
# 
# KNN (K-Nearest-Neighbors) is also a classification algorithm but can be used for regression purposes as well.
# 

# In[64]:


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(copy_train, Current_Saleprice)


# In[65]:


knn_house_price_prediction = classifier.predict(copy_test)
knn_house_price_prediction


# In[66]:


#plotting the saleprice values in a bar graph

plt.hist(knn_house_price_prediction,bins=7)


# In[67]:


output_df = pd.DataFrame({'Id': copy_test.index + 4,
                   'SalePrice': knn_house_price_prediction})
output_df


# In[68]:


output_df.to_csv(r'C:\Users\Checkout\Desktop\submission_KNN1.csv', index=False)


# ### XGB Regressor 
# 
# XGB is an efficient implementation of the gradient boosting algorithm.
# 
# It consists of a class of ensemble machine learning models used for regression purposes.

# In[69]:


xgbr = xgb.XGBRegressor(verbosity=0) 


# In[70]:


xgbr.fit(copy_train, Current_Saleprice)


# In[71]:


xgb_house_price_prediction = xgbr.predict(copy_test)


# In[72]:


xgb_house_price_prediction


# In[73]:


#plotting the saleprice values in a bar graph

plt.hist(xgb_house_price_prediction,bins=7)


# In[74]:


output_df = pd.DataFrame({'Id': copy_test.index +4,
                   'SalePrice': xgb_house_price_prediction})


# In[75]:


output_df


# In[76]:


output_df.to_csv(r'C:\Users\Checkout\Desktop\Submissions_XGB1.csv', index=False)


# ## Observations:
# 
# 1. As this was an ongoing competition , the one way of knowing your accuracy was to submit the output.csv file generated using each model on the kaggle competition.
# 2. After submitting , we compared the results from each regression model and proceeded with our analysis.
# 3. On kaggle , after submitting your results , you get a score on the leaderboard, lower the score better is your model.
# 4. The multiple regression model had a score of 0.94950.
# 5. The logistic regression model had a score of 0.27592.
# 6. The SVM model had a score of 0.41647.
# 7. The KNN model had a score of 0.36350.
# 8. The XGB Regressor model had a score of 0.45174.
# 
# 
# - According to the observations , logistic regression model performed best across all the models, but that is still not  accurate enough.
# - We would not proceed with SVM because it it generally used for classification purposes and this is not a classification problem.
# - Further analysis needs to be done and different types of models need to be explored. 
# - It is also possible that we would need to club/join two-three different kinds of models to achieve maximum accuracy.

# In[ ]:




