import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv('train.csv')
data = df.drop(columns=['id','MedHouseVal']).to_numpy()
label = df['MedHouseVal'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.1, random_state = 43)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# linear_model = LinearRegression()
# linear_model.fit(X_train, y_train)
# Y_predict = linear_model.predict(X_test)
# rmse = root_mean_squared_error(y_test, Y_predict)
# print(rmse)
# svr_model = SVR()
# svr_model.fit(X_train, y_train)
# Y_pred = svr_model.predict(X_test)
# rmse = root_mean_squared_error(y_test, Y_pred)
# print(rmse)
R_model = GradientBoostingRegressor()
R_model.fit(X_train, y_train)
Y_pred = R_model.predict(X_test)
rmse = root_mean_squared_error(y_test, Y_pred)
# print(rmse)
df = pd.read_csv('test.csv')
id_column = df['id']
df_new = df.drop(columns=["id"]).to_numpy()
X_trans = scaler.transform(df_new)

Y_pred = R_model.predict(X_trans)
predict_column = pd.Series(Y_pred)
pd.concat([id_column, predict_column], axis = 1).to_csv('Result.csv', header=['id','MedHouseVal'], index = False)




