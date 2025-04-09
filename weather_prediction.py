

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('seattle-weather.csv')

data.head()

data.drop(['date'],axis=1)

data['weather'].unique()

data['weather'].value_counts()

data.isnull().sum()

data['weather']=data['weather'].replace(["rain","sun","fog","drizzle","snow"],[1,2,3,4,5])

data['weather']=data['weather'].replace(0,1)

data.tail()

data=data.drop(['date'],axis=1)

X=data.iloc[:,:-1]
Y=data.iloc[:,-1]
print(X)

print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,Y_train)

rfc.predict(X_test)

from sklearn.metrics import accuracy_score
y_pred=rfc.predict(X_test)
accurecy=accuracy_score(Y_test,y_pred)
print(accurecy)

prec=float(input("enter the precipitation"))
max=float(input("enter the max temp"))
min=float(input("enter the min temp"))
wind=float(input("enter the wind"))
weatherpred=[prec,max,min,wind]
final_pred=rfc.predict([weatherpred])
print(final_pred)
if 1 in final_pred:
  print("Rain predicted Today !")
elif 2 in final_pred:
  print("Sun predicted Today !")
elif 3 in final_pred:
  print("Fog predicted Today !")
elif 4 in final_pred:
  print("Dizzle predicted Today !")
else:
  print("Sorry couldn't predict correctly")

