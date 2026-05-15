#import Libraries
import pandas as pd
import numpy as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
mean_saqure_error,r2_score
#load dataset
df=pd.read_csv("movie.csv")
#display first 5 row
print("First 5 row of dataset:")
print(df.head())
#check missing values
print("/n Missing values:")
print(df.isnull().sum())
#remove missing values
df=df.dropna()
#encode categorical columns
label_encoder=LadelEncoder()
df['Genre']=label_encoder.fit_transform(df['Genre'])
df['Director']=label_encoder.fit_transform(df['Director'])
df['Actor 1']=label_encoder.fit_transform(df['Actor 1'])
df['Actor 2']=label_encoder.fit_transform(df['Actor 2'])
df['Actor 3']=label_encoder.fit_transform(df['Actor 3'])
#select features and target 
x=df[['Genre','Director','Actor 1','Actor 2','Actor 3','duration','votes']]
y=df['Rating']
#split dataset into training and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#create model
model=RandomForestRegressor(n_estimators=100,random_state=42)
#train model
model.fit(x_train,y_train)
#predict ratings
y_pred=model.predict(x_test)
#evaluate model
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
#print result
print("\n Model Evaluation:")
print("Mean Absolute Error:",mae)
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("R2 Score:",r2)
#predict new movie rating
new_movie=pd.dataframe({
    'Genre':[1],
    'Director':[5],
    'Actor 1':[10],
    'Actor 2':[15],
    'Actor 3':[20],
    'duration':[120],
    'votes':[5000]
})
predicted_rating=model.predict(new_movie)
print("\n Predicted Rating for new movie:",predicted_rating[0])

