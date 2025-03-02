import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv(r"C:\Users\Gunji Madhav\OneDrive\Desktop\Salary_Data.csv")
x = dataset.iloc[:, :-1].values #to get frist three columns independent variables
y = dataset.iloc[:, -1].values # to get last column dependent variable
#train and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#Regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
#predicting the xtest
#y_pred=regressor.predict(x_test)
#ploting the graph to see error
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#predict the future
m_slope=regressor.coef_
m_slope
c_intercept=regressor.intercept_
c_intercept
#experence is 15years
emp15=m_slope*15+c_intercept
emp15