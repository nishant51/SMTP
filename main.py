# prompt: merge all the above 
from fastapi import FastAPI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

app = FastAPI()
# Define the stock symbol and the date range
symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'

# Download historical data
data = yf.download(symbol, start=start_date, end=end_date)

# Display the data
print(data)

data.head()
data.tail()
data = data.reset_index()
data.head()
data = data.drop(['Date','Adj Close'], axis = 1)
data.head()
plt.plot(data.Close)
ma100 = data.Close.rolling(100).mean()
ma100
ma100[100]
ma100[98]
plt.figure(figsize = (12,6))
plt.plot(data.Close)
plt.plot(ma100,'g')
ma200 = data.Close.rolling(200).mean()
ma200
plt.figure(figsize = (12,6))
plt.plot(data.Close)
plt.plot(ma100,'g')
plt.plot(ma200,'r')
data.shape
#Splitting Data into Training and Testing

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.7)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.7):int(len(data))])


data_training.head()
data_testing.head()
scaler = MinMaxScaler(feature_range = (0,1))
data_training_array = scaler.fit_transform(data_training)
data_training_array
data_training_array.shape
data_training_array.shape[0]
x_train = []
y_train = []

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape
#ML Model
# Define the model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50)  # Increase the number of epochs for better convergence
model.save('my_model.h5')

model.save('my_model.h5')
data_testing.head()
past_100_days = data_training.tail(100)
print(type(past_100_days))
final_data = pd.concat([past_100_days, data_testing], ignore_index=True)

final_data.head()
input_data = scaler.fit_transform(final_data)
input_data
input_data.shape
x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)
#Making Prediction

y_predicted = model.predict(x_test)
y_predicted.shape
y_predicted
scaler.scale_
scale_factor = 1/0.00682769
y_predicted = y_predicted * scale_factor
y_test = y_test *scale_factor
plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted, 'r', label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
