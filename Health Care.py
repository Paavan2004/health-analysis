#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[57]:


data = pd.read_csv('C:/Users/india/framingham.csv')
data.head()


# In[23]:


data.shape


# In[24]:


data.keys()


# In[25]:


data.info()


# In[26]:


data.describe()


# In[27]:


data.isna().sum()


# In[28]:


data.dropna(axis = 0, inplace = True)
print(data.shape)


# In[29]:


data['TenYearCHD'].value_counts()


# In[30]:


plt.figure(figsize=(8, 6))
sns.countplot(x='education', data=data)
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.title('Distribution of Education Levels')
plt.xticks(rotation=45)
plt.show()


# In[31]:


chd_no = data[data['TenYearCHD'] == 0]
chd_yes = data[data['TenYearCHD'] == 1]

plt.hist(chd_no['cigsPerDay'], bins=15, alpha=0.5, label='No CHD', edgecolor='black')
plt.hist(chd_yes['cigsPerDay'], bins=15, alpha=0.5, label='CHD', edgecolor='black')

plt.xlabel('Cigarettes Per Day')
plt.ylabel('Frequency')
plt.title('Histogram: Cigarettes Per Day vs TenYearCHD')
plt.legend()
plt.show()


# In[32]:


chd_no = data[data['TenYearCHD'] == 0]
chd_yes = data[data['TenYearCHD'] == 1]

plt.scatter(chd_no['diabetes'], chd_no['totChol'], label='No CHD', color='blue', alpha=0.7)
plt.scatter(chd_yes['diabetes'], chd_yes['totChol'], label='CHD', color='red', alpha=0.7)

plt.xlabel('Diabetes')
plt.ylabel('Total Cholesterol')
plt.title('Scatter Plot: Diabetes vs Total Cholesterol with TenYearCHD')
plt.legend()
plt.show()


# In[33]:


plt.figure(figsize = (14, 10))
sns.heatmap(data.corr(), cmap='YlGnBu',annot=True, linecolor='Green', linewidths=1.0)
plt.show()


# In[34]:


sns.catplot(data=data, kind='count', x='male',hue='currentSmoker')
plt.show()


# In[ ]:


sns.catplot(data=data, kind='count', x='TenYearCHD', col='male',row='currentSmoker', palette='Blues')
plt.show()


# Machine Learning Part

# In[36]:


X = data.iloc[:,0:15]
y = data.iloc[:,15:16]


# In[37]:


X.head()


# In[38]:


y.head()


# Importing the model and assigning the data for training and test set

# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21)


# In[40]:


from sklearn.preprocessing import StandardScaler
cs=StandardScaler()
X_train=cs.fit_transform(X_train)
X_test=cs.transform(X_test)


# Logistic Regression(ML Model)

# In[41]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=21)


# Training the data

# In[42]:


logreg.fit(X_train, y_train)


# Testing the data

# In[43]:


y_pred = logreg.predict(X_test)
print(y_pred)


# 
# Confusion Matrix
# 

# In[44]:


from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix is:\n",cm)


# In[45]:


score = logreg.score(X_test, y_test)
print("Prediction score is:",score)


# In[46]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=21)


# In[47]:


rf_model.fit(X_train, y_train)


# In[48]:


rf_predictions = rf_model.predict(X_test)
print(rf_predictions)


# In[49]:


from sklearn.metrics import confusion_matrix

# Assuming rf_predictions contains the predictions

# Generating the confusion matrix
conf_matrix = confusion_matrix(y_test, rf_predictions)
print("Confusion Matrix:")
print(conf_matrix)


# In[50]:


from sklearn.metrics import accuracy_score

# Assuming y_test contains the actual labels and rf_predictions contains the predictions

# Calculating accuracy
accuracy = accuracy_score(y_test, rf_predictions)
print(f"Accuracy: {accuracy}")


# In[51]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[52]:


svm_model = SVC(kernel='linear')  # You can change the kernel type as needed


# In[53]:


svm_model.fit(X_train, y_train)


# In[54]:


predictions = svm_model.predict(X_test)
print(predictions)


# In[55]:


from sklearn.metrics import confusion_matrix
# Generating the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
# Displaying the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)


# In[56]:


from sklearn.metrics import accuracy_score
# Assuming y_test contains the actual labels and rf_predictions contains the predictions
# Calculating accuracy
accuracy = accuracy_score(y_test, rf_predictions)
print(f"Accuracy: {accuracy}")


# In[ ]:




