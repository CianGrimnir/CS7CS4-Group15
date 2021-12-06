from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy
from keras.optimizers import Adam
import keras
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from matplotlib import pyplot

pov_16 = pd.read_csv("ChildPov_16_2006-2016.csv")
target = pd.read_csv("Crime_2000-2017.csv")
pov_18 = pd.read_csv("ChildPov_18_2006-2016.csv")
earning = pd.read_csv("Earnings_2000-2017.csv")
female65 = pd.read_csv("Female65LifeExpect_2000-2017.csv")
female = pd.read_csv("FemaleLifeExpect_2000-2017.csv")
male65 = pd.read_csv("Male65LifeExpect_2000-2017.csv")
male = pd.read_csv("MaleLifeExpect_2000-2017.csv")
neet = pd.read_csv("NEETs_2009-2015.csv")
rent = pd.read_csv("Rents_2011-2017.csv")
traffic = pd.read_csv("Traffic_2010-2017.csv")
workless = pd.read_csv("Workless_2004-2017.csv")

earning = earning.iloc[:, 6:-1]
female = female.iloc[:-1, 6:-1]
female65 = female65.iloc[:, 5:-1]
male = male.iloc[:-1, 6:-1]
male65 = male65.iloc[:-1, 5:-1]
traffic = traffic.iloc[:, 6:-1]
workless = workless.iloc[:, 2:-1]
target = target.iloc[:, 6:-1]

earning_matrix = earninig.iloc[:, :].to_numpy()
earning_matrix = earning.iloc[:, :].to_numpy()
female_matrix = female.iloc[:, :].to_numpy()
female65_matrix = female65.iloc[:, :].to_numpy()
male_matrix = male.iloc[:, :].to_numpy()
male65_matrix = male65.iloc[:, :].to_numpy()
traffic_matrix = traffic.iloc[:, :].to_numpy()
workless_matrix = workless.iloc[:, :].to_numpy()
target_matrix = target.iloc[:, :].to_numpy()

x1 = earning_matrix.reshape(-1, 1)
x2 = female65_matrix.reshape(-1, 1)
x3 = female_matrix.reshape(-1, 1)
x4 = male_matrix.reshape(-1, 1)
x5 = male65_matrix.reshape(-1, 1)
x6 = traffic_matrix.reshape(-1, 1)
x7 = workless_matrix.reshape(-1, 1)
y = target_matrix.reshape(-1, 1)

XStack = np.column_stack((x1, x2, x3, x4, x5, x6, x7))

X_train, X_test, y_train, y_test = train_test_split(XStack, y, random_state=1)

model = Sequential()
model.add(Dense(128, activation="relu", input_dim=7))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000000, batch_size=100, verbose=2,
                    callbacks=[es])
model.predict(X_test)
ypred = model.predict(X_test)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
plt.title('Training Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.plot(y_test,ypred,'ro')
scores = r2_score(y_test,ypred)
print(scores)
plt.show()

# Score - 0.871233676865

# model 2 

model = Sequential()
model.add(Dense(20, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000000, batch_size=100, verbose=2,
                    callbacks=[es])
model.predict(X_test)
ypred = model.predict(X_test)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
plt.title('Training Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.plot(y_test,ypred,'ro')
scores = r2_score(y_test,ypred)
print(scores)
plt.show()

plt.plot(y_test, color = 'red', label = 'Real data')
plt.plot(ypred, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

# Score - 0.8816908050848434



model = Sequential()
model.add(Dense(32, activation = 'relu', input_dim = 7))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, batch_size = 10, epochs = 100)
y_pred = model.predict(X_test)


# Score - 0.8385995798362442