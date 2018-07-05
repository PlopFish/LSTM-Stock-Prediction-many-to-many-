import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import logging
import math
import os
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Flatten
from keras import optimizers
from sklearn.metrics import mean_squared_error
from keras.models import load_model


np.random.seed(7)



## 데이터 불러오기 / Data Preparation
stock_code = '000660_indi03.csv' # SK하이닉스 / SK hynix (In korea)
encoding = 'euc-kr'
names01 = ['Date', 'Open', 'High', 'Low', 'Close', 'Close5ma', 'Close10ma', 'Close20ma', 
         'Close60ma', 'BBU', 'BBD', 'Volume', 'Volume5ma', 'Volume10ma', 'Volume20ma', 'Volume60ma',
         'RT', 'INST', 'FRGN', 'PGRM', 'SSR', 'ERD', 'IRK', 'OBV', 'OBVSignal', 'MACDOscillator', 'MACD', 'MACDSignal',
         'CCI', 'CCISignal', 'ADX', 'DIP', 'DIM', 'ADXR', 'Disparity5', 'Disparity10', 'Disparity20', 'Disparity60',
         'RSI', 'RSISignal', 'SlowK', 'SlowD', 'WilliamsR', 'WilliamsD']

df = pd.read_csv(stock_code, names=names01, index_col='Date', engine='python', thousands=',', encoding=encoding)

## names01 데이터 목록 해설 / names01's data list
# BBU = '볼린저 밴드 상한선 / Bollinger Band Up', BBD = '볼린저 밴드 하한선 Bollinger Band Down', RT = '개인 순매수 / Retail Investor's straight purchase'
# INST = '기관순매수 / Institution Investor's straight purchase', FRGN = '외국인 순매수 / Foreigner Investor's straight purchase'
# PGRM = '프로그램 순매수 / Program Investor's straight purchase', SSR = '공매도 비율 / short selling ratio', ERD = '원-달러 환율 / Exchange Rate KRW-USD'
# IRK = '한국 콜금리 / Korea Interest Rate', DIP = 'DMI 지표중 PDI / DMI's PDI', DIM = 'DMI 지표중 MDI / DMI's MDI', Disparity = '이격도'
# SlowK = 'Stochastic Slow %K', SlowD = 'Stochastic Slow %D'



## 데이터 전처리 / Data Preprocessing
price_indicator = df.loc[:,'Open':'BBD'].values[1:].astype(np.float) # 가격 관련 지표 / Price Indicator
volume_indicator = df.loc[:,'Volume':'Volume60ma'].values[1:].astype(np.float) # 거래량 관련 지표 / Volume Indicator
etc_indicator = df.loc[:,'RT':'WilliamsD'].apply(lambda x: x.str.replace('%','')).values[1:].astype(np.float) # 추세 또는 거래량 활용 지표 / Trend and Volume Indicator

scaler = MinMaxScaler(feature_range=(0, 1)) # 0~1 값으로 스케일링
scaler_etc = MinMaxScaler(feature_range=(-1, 1)) # 0~1 값으로 스케일링

scaled_price_indicator = scaler.fit_transform(price_indicator) # 가격 관련 지표에 스케일링
scaled_volume_indicator = scaler.fit_transform(volume_indicator) # 거래량 관련 지표에 스케일링
scaled_etc_indicator = scaler_etc.fit_transform(etc_indicator) # 추세 또는 거래량 활용 지표에 스케일링



## 데이터셋 생성하기 / Creating Dataset

# 행은 그대로 두고 열을 우측에 붙여 합친다 / leave the row, and attach columns to right
x = np.concatenate((scaled_price_indicator, scaled_volume_indicator, scaled_etc_indicator), axis=1) # axis=1
y = x[:, [3]] # 타켓은 주식 종가이다 / target is 'Close' price

# dataX와 dataY 생성 / Creating dataset 'dataX' and 'dataY'
seq_length = 30
predict_day = 5

dataX = [] # 입력으로 사용될 Sequence Data / Input - Sequence Data
dataY = [] # 출력(타켓)으로 사용 / Output(target)
for i in range(0, int(len(y) - seq_length - predict_day)):
    _x = x[i : i + seq_length]
    _y = y[i + predict_day : i + seq_length + predict_day] # 다음 나타날 주가(정답) / after 5 days later (in _x), 'Close' price is the correct answer.
    dataX.append(_x) # dataX 리스트에 추가 / add to dataX's list
    dataY.append(_y) # dataY 리스트에 추가 / add to dataY's list


# 학습용 데이터 생성 / Creating Train Set
train_size = int(len(dataY))

# 데이터를 잘라 학습용 데이터 생성 / Slicing Train Set
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

print("="*50)
print(_x.shape)
print(_y.shape)
print(trainX.shape)
print(trainY.shape)
print("="*50)



## LSTM 모델 / LSTM Model

input_columns = 43 # 데이터 셋의 '열' 개수 (dataX) / dataset's columns (dataX)

model = Sequential()
model.add(LSTM(256, batch_input_shape=(30, 30, input_columns), return_sequences=True, stateful=True, dropout=0.7))
model.add(LSTM(256, return_sequences=True, stateful=True, dropout=0.7))
model.add(LSTM(256, return_sequences=True, stateful=True, dropout=0.7))
model.add(Dense(1))
model.add(Activation('softsign'))
model.summary()

# 모델 학습 설정 및 진행 / Model Training options and Progress
keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam')
hist = model.fit(trainX, trainY, epochs=10, batch_size=30, verbose=1)


# 학습 과정 살펴보기 / watching train loss
print(hist.history['loss'])
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()



# SK하이닉스 전체 데이터를 학습하여, 이외의 종목(삼성전자, LG전자, 현대차 등)을 예측합니다. 그렇기 때문에 여기 코드에서는 학습만 진행하여 모델을 저장합니다.
# 예측은 03_01.py 등의 파일에서 모델을 불러와 예측을 실시합니다.
# In this code, we are going to train SK hynix's data, and predict with other companys (ex: samsung, Hyundai car, LG).
# You have to run 03_01 and other file to run stock prediction.


# 모델 저장 / Model Save
model.save('lstm_stock_prediction_indi03_01.h5')
