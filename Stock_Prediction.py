import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pandas_datareader as web 
import datetime as dt 
 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
 
#Load Data
company = 'FB'

start = dt.datetime(2012,1,1)
end = dt.datetime(2021,1,1)

data = web.DataReader(company, 'yahoo', start, end)

# Prepare Data
scaler = MinMaxScalaer(feature_range=(0,1))
scaler_data = scaler.fit_transform(data['claose'].values.reshape(-1,1))

predictions_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
     x_train.append(scaled_data[x-prediction_days:x, 0])
     y_train.append(scaled_data[x, 0])

x_train,  y_train = np.array(x_train), np.array(y_array)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#  The Model/Algoithm
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2)) 
model.add(LSTM(units=50))
model.add(Dropout(0.2)) 
model.add(Dense(units=1)) #Prediction of the next closing

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#''Test the model Accuracy on the Existing Data''

#Load Test data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_input = total_dataset[len(total_dataset) - len(test_data) - predictions_days:].value
model_input = model_input.reshape(-1, 1)
model_input = scaler.transform(model_inputs)

#Make Predictions on the test data

x_test= []

for x in range(prediction_days,len(model_inputs)):
     x_test.append(model_inputs[x-prediction_days:x,0])

x_test = np_array(x_test)
x_test = np.reshape(x_test, x_test.shapep[0], x_test.shape[1],1)

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


#plot the tests predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} Prices")
plt.plot(predicted_prices, color= 'green', label= f"Predicted {company} Price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

