

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('tsla_raw_data.csv')

data.head()

data.drop('date',axis=1)

data.head()

data=data.drop(['date','adjusted_close','change_percent','avg_vol_20d'],axis=1)

data.head()

plt.plot(data.close)

data.dropna()
#data

data.isna().sum()

data=data.dropna()

ma100=data.close.rolling(100).mean()
#ma100 #for first 100 values it will have nan because it should have previous 100 values

plt.figure(figsize=(12,6))
plt.plot(data.close)
plt.plot(ma100,'r')

ma200=data.close.rolling(200).mean()
#ma200

plt.figure(figsize=(12,6))
plt.plot(data.close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')

#data.shape

#splitting data
data_traning=pd.DataFrame(data['close'][0:int(len(data)*0.70)])  #70 % of data from 0 to 70% of index
data_testing=pd.DataFrame(data['close'][int(len(data)*0.70):int(len(data))]) #from 70 % to end of the data
print(data_traning.shape)
print(data_testing.shape)

data_traning.head()

from sklearn.preprocessing import MinMaxScaler
scaller=MinMaxScaler(feature_range=(0,1))

data_traning_array=scaller.fit_transform(data_traning)
#data_traning_array

#data_traning_array.shape

x_train=[]
y_train=[]
for i in range(100,data_traning_array.shape[0]):
  x_train.append(data_traning_array[i-100:i])   # first 0 to 100 will be input and i will be the output and next 100 will be the input and next i will be the output
  y_train.append(data_traning_array[i,0]) # [i,0] because it should be 2 dimenctional

x_train,y_train=np.array(x_train),np.array(y_train)
#x_train.shape

# creating a ml or dl model
from keras.layers import Dropout,Dense,LSTM
from keras.models import Sequential

model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,
               input_shape=(x_train.shape[1],1))) # [1] is 100 from above (2330,100,1 ) and as we are working only with close column we use 1 if we work with open,close,volume then it would be 3
model.add(Dropout(0.2))


model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))


model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))  # units 1 because we need only one output

model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)

model.save('keras_model.h5')

data_testing.head()

data_traning.tail(100)

past_100_days=data_traning.tail(100)

#final_df=past_100_days.append(data_testing,ignore_index=True) it is showing an error

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

final_df.head()

input_data=scaller.fit_transform(final_df)
#input_data

#input_data.shape

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)

y_predict=model.predict(x_test)

#y_predict.shape

#y_test

#y_predict

#scaller.scale_ # it will give us the scalling factor through which we have scalled the values

scale_factor=1/0.00046934
y_predict=y_predict*scale_factor
y_test=y_test*scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predict,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

