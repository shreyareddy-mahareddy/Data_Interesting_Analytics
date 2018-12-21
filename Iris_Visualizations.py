
# coding: utf-8

# In[2]:


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


# Import the dataset using pandas as a mxn matrix
# Note - The working directory has been set as the location where the csv is located

pddata = pd.read_csv("C:\\Users\\shrey\\Documents\\UIC\\UIC- Semester 3\\Machine Learning with Python\\Homework\\HW1\\Iris.csv")
pddata.head(5)


# In[12]:


# Check the shape (m x n) of the imported dataset
# m - number of training examples
# n - number of features + 1 

print(pddata.shape)


# In[4]:


# Now the number of features = Total number of columns - 1
# Because the column 'Class' is the dependent variable and not the feature

# Hence number of features = 5-1 = 4


# In[18]:


# Check the number of labels in the dataset

uniqueclasses = pddata['Species'].unique()
uniqueclasses.size


# In[19]:


# Check the number of each label in the Class variable

pddata.groupby('Species').size()


# In[20]:


# Convert pandas dataframe into Numpy Array

npdata = np.array(pddata)
npdata


# In[22]:


# Create a class called colors such that we get difference colors for each label

colors = np.where(pddata["Species"]=='Iris-setosa','b','-')
colors[pddata["Species"]=='Iris-versicolor'] = 'g'
colors[pddata["Species"]=='Iris-virginica'] = 'r'


# In[9]:


SLSW = pddata.plot(x="SepalLengthCm", y="SepalWidthCm", kind="scatter", c=colors, title ='Sepal Length vs Sepal Width')


# In[24]:


SLPL = pddata.plot(x="SepalLengthCm", y="PetalLengthCm", kind="scatter", c=colors, title ='Sepal Length vs Petal Length')


# In[25]:


SLPW = pddata.plot(x="SepalLengthCm", y="PetalWidthCm", kind="scatter", c=colors, title ='Sepal Length vs Petal Width')


# In[26]:


SWPL = pddata.plot(x="SepalWidthCm", y="PetalLengthCm", kind="scatter", c=colors, title ='Sepal Width vs Petal Length')


# In[27]:


SWPW = pddata.plot(x="SepalWidthCm", y="PetalWidthCm", kind="scatter", c=colors, title ='Sepal Width vs Petal Width')


# In[28]:


PLPW = pddata.plot(x="PetalLengthCm", y="PetalWidthCm", kind="scatter", c=colors, title ='Petal Length vs Petal Width')

