#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


from sklearn.model_selection import cross_val_score


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the datatset

# In[9]:


df = pd.read_csv('data/houseprice_data.csv')


# # Displaying the first few rows of the dataset

# In[10]:


print(df.head())


# # Displaying the last few rows of the dataset

# In[11]:


print(df.tail())


# # Exploratory Data Analysis
# # Checking for missing values and getting a summary of the dataset

# In[12]:


print(df.info())


# There are no missing values which is good for our analysis

# In[13]:


df.describe()


# 
# # Correlation analysis

# In[14]:


correlation_matrix = df.corr()


# In[15]:


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of House Features")
plt.show()


# # Simple Linear Regression
# # Splitting the dataset into train and test sets

# In[16]:


X = df[['sqft_living']]  # Predictor
y = df['price']          # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Building a simple linear regression model

# In[17]:


simple_lr_model = LinearRegression()
simple_lr_model.fit(X_train, y_train)


# In[18]:


y_pred = simple_lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[19]:


print("MSE:", mse)
print("R-squared:", r2)


# # Predicting and evaluating the model

# In[20]:


y_pred = simple_lr_model.predict(X_test)


# # Multiple Linear Regression
# # Selecting more features for the multiple linear regression model

# In[21]:


features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'condition', 
            'grade', 'sqft_above', 'sqft_basement']
X_multi = df[features]


# # Splitting the dataset into train and test sets for multiple regression

# In[22]:


X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y, test_size=0.2, random_state=42)


# # Building the multiple linear regression model

# In[23]:


multi_lr_model = LinearRegression()
multi_lr_model.fit(X_train_multi, y_train_multi)


#  # Predicting and evaluating the multiple regression model

# In[24]:


y_pred_multi = multi_lr_model.predict(X_test_multi)
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)


# In[25]:


print("MSE_MULTI:", mse_multi)
print("R-squared_MULTI:", r2_multi)


# # Extended Linear Regression
# # Adding more features including 'zipcode', 'yr_built', and 'yr_renovated' Splitting the dataset for the extended model

# In[26]:


extended_features = features + ['zipcode', 'yr_built', 'yr_renovated']
X_extended = df[extended_features]
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_extended, y, test_size=0.2, random_state=42)


# # Building the extended linear regression model

# In[27]:


extended_lr_model = LinearRegression()
extended_lr_model.fit(X_train_ext, y_train_ext)


# In[28]:


y_pred_ext = extended_lr_model.predict(X_test_ext)


# # Cross-Validation

# In[29]:


cv_scores = cross_val_score(extended_lr_model, X_extended, y, cv=5)


# # Predicting and evaluating the extended model

# In[30]:


y_pred_ext = extended_lr_model.predict(X_test_ext)
mse_ext = mean_squared_error(y_test_ext, y_pred_ext)
r2_ext = r2_score(y_test_ext, y_pred_ext)


# In[31]:


print("CV_Scores.mean:", cv_scores.mean())
print("MSE_ext:", mse_ext)
print("R-squared_ext:", r2_ext)


# In[32]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test_ext, y_pred_ext, color='blue', alpha=0.5)
plt.plot(y_test_ext, y_test_ext, color='red')  # Ideal line where predicted = actual
plt.title('Actual vs. Predicted Prices: Extended Model')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


# In[ ]:




