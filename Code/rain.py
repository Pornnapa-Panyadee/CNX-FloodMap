from pandas import read_csv
from datetime import datetime
import numpy as np
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

dataset = pd.read_csv('00_rain.csv')
# dataset=dataset.drop(columns=['Value_20','Value_73'])
dataset=dataset.drop(columns=['date'])
values = dataset.values

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0.01, 1))
scaled = scaler.fit_transform(values)
# scaled=values

reframed4h = series_to_supervised(scaled, 3, 3)
# reframed4h

values = reframed4h.values

n_train_hours = int(len(reframed4h)*0.60)
n_valid_hours = int(len(reframed4h)*0.20)+n_train_hours

train = values[:n_train_hours,:]
train_X, train_y = train[:, :-14], train[:, -14]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
train_y = train[:, 42:56]

valid = values[n_train_hours:n_valid_hours, :]
valid_X, valid_y = valid[:, :-14], valid[:, -14]
valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
valid_y = valid[:, 42:56]

test = values[n_valid_hours:, :]
test_X, test_y = test[:, :-14], test[:, -14]
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
test_y = test[:, 42:56]



in_dim = (train_X.shape[1], train_X.shape[2])
out_dim = 14

#n_train_hours = int(len(reframed4h)*0.80)
#train = values[:n_train_hours,:]

# frame as supervised learning
# reframed4h = series_to_supervised(scaled, 16, 1)

#train_X, train_y = train[:, :-14], train[:, -14]
#train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#train_y = train[:, 42:56]
#test = values[n_train_hours:, :]
#test_X, test_y = test[:, :-14], test[:, -14]
#test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#test_y = test[:, 42:56]

#in_dim = (train_X.shape[1], train_X.shape[2])
#out_dim = 14


# n_train_hours = int(len(reframed4h)*0.80)
# train = values[:n_train_hours,:]

# train_X, train_y = train[:, :-16], train[:, -16]
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# train_y = train[:, 48:64]
# test = values[n_train_hours:, :]
# test_X, test_y = test[:, :-16], test[:, -16]
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# test_y = test[:, 48:64]

# in_dim = (train_X.shape[1], train_X.shape[2])
# out_dim = 16

data = {
    'ex':[],
    'i': [],
    'rmse_train': [],
    'mape_train': [],
    'mae_train' : [],
    'R_train' : [],
    'rmse_test': [],
    'mape_test' : [],
    'mae_test' : [],
    'R_test' : [],
}

error = pd.DataFrame(data)

for ex in range(100,1010,100):
  for i in range(10):
    model = Sequential()
    model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(out_dim, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # fit network
    history = model.fit(train_X, train_y, epochs=ex, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # predict
    yhat_train= model.predict(train_X)
    rmse_train1 = sqrt(mean_squared_error(train_y, yhat_train))
    mape_train1 = mean_absolute_percentage_error(train_y, yhat_train)
    mae_train1 = mean_absolute_error(train_y, yhat_train)
    r_train1 = r2_score(train_y, yhat_train)

    yhat_test = model.predict(test_X)
    rmse_test1 = sqrt(mean_squared_error(test_y, yhat_test))
    mape_test1 = mean_absolute_percentage_error(test_y, yhat_test)
    mae_test1 = mean_absolute_error(test_y, yhat_test)
    r_test1= r2_score(test_y, yhat_test)
    
    # rmse_train.append(rmse_train1)
    # mape_train.append(mape_train1)
    # rmse_test.append(rmse_test1)
    # mape_test.append(mape_test1)

    # Sava data
    datasetT = pd.read_csv('water.csv')
    time_trian=datasetT[0:len(train_y)]
    time_trian=time_trian.reset_index()
    Training = pd.DataFrame()
    Training["DateTime"] = time_trian['date']
    Training['S15'] = train_y[:, 0]
    Training['S16'] = train_y[:,1]
    Training['S17'] = train_y[:,2]
    Training['S19'] = train_y[:,3]
    Training['S3'] = train_y[:,4]
    Training['S5'] = train_y[:,5]
    Training['S1'] = train_y[:,6]
    Training['S8'] = train_y[:,7]
    Training['S7'] = train_y[:,8]
    Training['S9'] = train_y[:,9]
    Training['S11'] = train_y[:,10]
    Training['S6'] = train_y[:,11]
    Training['S10'] = train_y[:,12]
    Training['S12'] = train_y[:,13]
    Training['S2'] = train_y[:,14]
    Training['S13'] = train_y[:,15]

    Training['S15_pre'] = yhat_train[:, 0]
    Training['S16_pre'] = yhat_train[:,1]
    Training['S17_pre'] = yhat_train[:,2]
    Training['S19_pre'] = yhat_train[:,3]
    Training['S3_pre'] = yhat_train[:,4]
    Training['S5_pre'] = yhat_train[:,5]
    Training['S1_pre'] = yhat_train[:,6]
    Training['S8_pre'] = yhat_train[:,7]
    Training['S7_pre'] = yhat_train[:,8]
    Training['S9_pre'] = yhat_train[:,9]
    Training['S11_pre'] = yhat_train[:,10]
    Training['S6_pre'] = yhat_train[:,11]
    Training['S10_pre'] = yhat_train[:,12]
    Training['S12_pre'] = yhat_train[:,13]
    Training['S2_pre'] = yhat_train[:,14]
    Training['S13_pre'] = yhat_train[:,15]

    name="output_train_day_river_"+str(ex)+"_"+str(i)+".csv"
    Training.to_csv(name)

    # test 
    le = len(test_y)+len(train_y)
    time_test=datasetT[len(datasetT):le]
    time_test=time_test.reset_index()
    Testing = pd.DataFrame()
    Testing["DateTime"] = time_test['date']
    Testing['S15'] = test_y[:, 0]
    Testing['S16'] = test_y[:,1]
    Testing['S17'] = test_y[:,2]
    Testing['S19'] = test_y[:,3]
    Testing['S3'] = test_y[:,4]
    Testing['S5'] = test_y[:,5]
    Testing['S1'] = test_y[:,6]
    Testing['S8'] = test_y[:,7]
    Testing['S7'] = test_y[:,8]
    Testing['S9'] = test_y[:,9]
    Testing['S11'] = test_y[:,10]
    Testing['S6'] = test_y[:,11]
    Testing['S10'] = test_y[:,12]
    Testing['S12'] = test_y[:,13]
    Testing['S2'] = test_y[:,14]
    Testing['S13'] = test_y[:,15]


    Testing['S15_pre'] = yhat_test[:, 0]
    Testing['S16_pre'] = yhat_test[:,1]
    Testing['S17_pre'] = yhat_test[:,2]
    Testing['S19_pre'] = yhat_test[:,3]
    Testing['S3_pre'] = yhat_test[:,4]
    Testing['S5_pre'] = yhat_test[:,5]
    Testing['S1_pre'] = yhat_test[:,6]
    Testing['S8_pre'] = yhat_test[:,7]
    Testing['S7_pre'] = yhat_test[:,8]
    Testing['S9_pre'] = yhat_test[:,9]
    Testing['S11_pre'] = yhat_test[:,10]
    Testing['S6_pre'] = yhat_test[:,11]
    Testing['S10_pre'] = yhat_test[:,12]
    Testing['S12_pre'] = yhat_test[:,13]
    Testing['S2_pre'] = yhat_test[:,14]
    Testing['S13_pre'] = yhat_test[:,15]

    name="output_test_day_river_"+str(ex)+"_"+str(i)+".csv"
    Testing.to_csv(name)

    new_row = {
                 'ex':ex,
                'i': i,
                'rmse_train': rmse_train1,
                'mape_train': mape_train1,
                'mae_train' : mae_train1,
                'R_train' : r_train1,
                'rmse_test': rmse_test1 ,
                'mape_test' : mape_test1,
                'mae_test' : mae_test1,
                'R_test' : r_test1,
              }
              
    error.loc[len(error)] = new_row
	
error.to_csv("error_rain_100S10R.csv")
