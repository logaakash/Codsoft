import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LableEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matlotlib.pyplot as plt
import seaborn as sns
#load dataset
df=pd.read_csv("Titanic-dataset.csv")
#show first 5 rows
print(df.head())
#check missing values
print(df.isnullc().sum())
#Fill missing age values
df=df['Age'].fillna(df['Age'].median())
#Fill missing embarked values
df['Embarked']=df['Embarked'].fillna(df['embarked'].mode()[0])
#drop cabin column
df.drop(columns=['cabin','Name','Ticket','PassengerID'],inplace=True)
#convert categorical columns
encoder=LabelEncoder()
df['sex']=encoder.fit_transform(df['Sex'])
df['Embarked']=encoder.fit_transform(df['Embarked'])
#Feature and target
x=df.drop('survived',axis=1)
y=df['survived']
#split dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size 
                                                =0.2,random_state=42)
# create model
model=RandomForestClassifier()
#train model
model.fit(x_train,y_train)
#predictions
y_pred=model.predict(x_test)
#accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Model Accuracy:",accuracy)
#survival count graph
sns.countplot(x='survived',data=df)
plt.title("Titanic Survival Count")
plt.show()