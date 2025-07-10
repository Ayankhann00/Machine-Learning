import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
Lr=LinearRegression()

Lr.fit(X_train,y_train)


Lr.predict(X_test)

plt.scatter(X_train,y_train,color='teal', marker='x', alpha=0.7)
plt.plot(X_train,Lr.predict(X_train),color='blue')
plt.title('Training set of years of experience vs salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,y_test,color='teal', marker='x', alpha=0.7)
plt.plot(X_test,Lr.predict(X_test),color='blue')
plt.title('Testing set of years of experience vs salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
