#import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#load dataset
#replace 'advertising.csv'with your dataset file name
df=pd.read_csv("advertising.csv")
#display first 5 row
print(df.head())
#feature and target
#example dataset columns: 'TV','Radio','Newspaper','Sales'
x=df[['TV','Radio','Newspaper']]
y=df['Sales']
#split dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#create model
model=LinearRegression()
#train model
model.fit(x_train,y_train)
#predict
y_pred=model.predict(x_test)
#evaluation
print("/n Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
#predict new data
new_data=[[230.1,37.8,68.2]]
prediction=model.predict(new_data)
print("/n Predicted Sales:", prediction[0])
