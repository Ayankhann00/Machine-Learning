import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor

dataset=pd.read_csv('Position_Salaries.csv')

X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

rf=RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(X,y)

rf.predict([[6]])
