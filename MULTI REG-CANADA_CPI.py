import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 

dataset = pd.read_csv('D:\course docs\ML\datasets\canada_cpi.csv') 

dataset.head()  

dataset.describe()

X = dataset[['Food','Shelter','Goods','Services']]
y = dataset['Year']  

dataset.plot(x='Services', y='Year', style='o')  
plt.title('Services vs Year ')  
plt.xlabel('Items utilisation')  
plt.ylabel('Year')  
plt.show()  

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)  

print(regressor.intercept_)  

print(regressor.coef_)  

#Prints R Square value
regressor.score(X_train, y_train)
 
regressor.score(X_test,y_test)

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
coeff_df  

y_pred = regressor.predict(X_test) 

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#y=a1x1+a2x2+a3x3+a4x4+b
