import streamlit as st
from datetime import date
import yfinance as yf

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import math
import pandas_datareader as data
from keras.models import load_model


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Trend Prediction")
stocks = ("AAPL","GOOG","MSFT","GME","ADA","BTC")
selected_stocks = st.selectbox("Select dataset for predictions", stocks)

n_years = st.slider("Years of predictions:", 1, 4)
period = n_years * 365
df = data.DataReader(selected_stocks, 'yahoo', START, TODAY)
st.subheader('Data from 2015 - 2021')
st.write(df.describe())
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

df = pd.read_csv("MSFT.csv", index_col='Date', parse_dates=["Date"])
df1=df['Close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
#splitting dataset
training_size=int(len(df1)*0.75)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
training_size,test_size

def create_dataset(dataset, time_step=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-time_step-1):
    a = dataset[i:(i+time_step), 0]
    dataX.append(a)
    dataY.append(dataset[i + time_step, 0])
  return numpy.array(dataX), numpy.array(dataY)

import numpy
time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

# The GRU architecture
model = Sequential()
# First GRU layer with Dropout regularisation
model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
# Second GRU layer
model.add(GRU(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Third GRU layer
model.add(GRU(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Fourth GRU layer
model.add(GRU(units=50))
model.add(Dropout(0.2))
# The output layer
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer='adam',loss='mean_squared_error')
# Fitting to the training set
model.fit(x_train,y_train,epochs=10,batch_size=32)

train_predict=model.predict(x_train)
test_predict=model.predict(x_test)
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

x_input = test_data[2115:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

#predictions for next 10 days
from numpy import array
lst_output=[]
n_steps=100
i=0
while(i<10):
  if(len(temp_input)>100):
    x_input=np.array(temp_input[1:])
    print("{} day input {}".format(i,x_input))
    x_input=x_input.reshape(1,-1)
    x_input=x_input.reshape((1,n_steps,1))
    yhat = model.predict(x_input,verbose=0)
    print("{} day output {}".format(i,yhat))
    temp_input.extend(yhat[0].tolist())
    temp_input=temp_input[1:]
    lst_output.extend(yhat.tolist())
    i=i+1
  else:
    x_input = x_input.reshape((1,n_steps,1))
    yhat = model.predict(x_input,verbose=0)
    print(yhat[0])
    temp_input.extend(yhat[0].tolist())
    print(len(temp_input))
    lst_output.extend(yhat.tolist())
    i=i+1

day_new=np.arange(1,101)
day_pred=np.arange(101,111)


st.subheader('Next 10 Days Predictions for MSFT')

fig2 = plt.figure(figsize=(16,10))
plt.plot(day_new,scaler.inverse_transform(df1[8757:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.title('Next 10 Days Prediction of MSFT')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price',fontsize=18)
plt.legend(['Actual ','10 Days Predictions'], loc='upper left')
st.pyplot(fig2)




#============================================


df4 = pd.read_csv("DataFrame.csv", index_col='Date', parse_dates=["Date"])
df4.drop('Unnamed: 7' , axis = 1 , inplace = True)
df5=df4['close']
scaler=MinMaxScaler(feature_range=(0,1))
df5=scaler.fit_transform(np.array(df5).reshape(-1,1))
#splitting dataset
training_size=int(len(df5)*0.75)
test_size=len(df5)-training_size
train_data,test_data=df5[0:training_size,:],df5[training_size:len(df5),:1]
training_size,test_size

def create_dataset(dataset, time_step=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-time_step-1):
    a = dataset[i:(i+time_step), 0]
    dataX.append(a)
    dataY.append(dataset[i + time_step, 0])
  return numpy.array(dataX), numpy.array(dataY)


time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

model1=Sequential()
model1.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model1.add(LSTM(50,return_sequences=True))                               # MODEL PREPARED FOR LSTM
model1.add(LSTM(50))
model1.add(Dense(1))
model1.compile(loss='mean_squared_error',optimizer='adam')
model1.summary()

model1.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64,verbose=1)    # fitting Model

train_predict=model1.predict(x_train)
test_predict=model1.predict(x_test)
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

x_input = test_data[5602:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

#predictions for next 10 minutes
from numpy import array
min_output=[]
n_steps=100
i=0
while(i<10):
  if(len(temp_input)>100):
    x_input=np.array(temp_input[1:])
    print("{} day input {}".format(i,x_input))
    x_input=x_input.reshape(1,-1)
    x_input=x_input.reshape((1,n_steps,1))
    yhat = model1.predict(x_input,verbose=0)
    print("{} day output {}".format(i,yhat))
    temp_input.extend(yhat[0].tolist())
    temp_input=temp_input[1:]
    min_output.extend(yhat.tolist())
    i=i+1
  else:
    x_input = x_input.reshape((1,n_steps,1))
    yhat = model1.predict(x_input,verbose=0)
    print(yhat[0])
    temp_input.extend(yhat[0].tolist())
    print(len(temp_input))
    min_output.extend(yhat.tolist())
    i=i+1

min_new=np.arange(1,101)
min_pred=np.arange(101,111)

st.subheader('Next 10 Minutes Predictions for NIFTY')
fig3 = plt.figure(figsize=(16,10))
plt.plot(min_new,scaler.inverse_transform(df5[22705:]))
plt.plot(min_pred,scaler.inverse_transform(min_output))
plt.title('1 Minute Prediction')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price',fontsize=18)
plt.legend(['Actual Price','10 Minutes Predictions'], loc='upper left')
st.pyplot(fig3)
