# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Fitting Simple Linear Regression to the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)
#Dumping the model using Pickle
pickle.dump(regressor , open('model.pkl' , 'wb'))

model = pickle.load(open('model.pkl' , 'rb'))
print (model.predict([[12]]))