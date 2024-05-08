
#导入所需的库
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import keras.layers as layers  
import keras.backend as K 
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from numpy import loadtxt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score  # 模型评估指标
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization  # 全连接层 展平层 卷积层 最大池化层
warnings.filterwarnings("ignore")





# Load the data
X_new = pd.read_csv('C:/Users/RH/ACE_regression/develop_this/data/bitter_data/bitter_320.csv', delimiter=',',header = None,index_col= 0, skiprows=1)
X = X_new.values
print(X.shape)
y = pd.read_csv('C:/Users/RH/ACE_regression/develop_this/data/bitter_data/y_bitter_data_log.csv',header=None).values
print(y.shape)
X_out = X_out = pd.read_csv('C:/Users/RH/A_regression/ESM_embedding_320/data/X_test_n_320.csv',delimiter=',',header = None,index_col= 0, skiprows=1)
X_out_Sequence = pd.read_csv('C:/Users/RH/ACE_regression/ESM/data/test_data.csv',delimiter=',',header = None,index_col= 0, skiprows=1)

# ## LSTM model development


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)



#RNN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.optimizers import adam_v2


# dataset processing
X_train = X_train_scaler.reshape(X_train_scaler.shape[0], 1, X_train_scaler.shape[1])
X_test = X_test_scaler.reshape(X_test_scaler.shape[0], 1, X_test_scaler.shape[1])

y_test = y_test.reshape(y_test.shape[0], 1, X_test.shape[1])
input_shape = (1, X_train.shape[2])


model = Sequential()
model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32,return_sequences = True))
model.add(Dense(50, activation='tanh'))
model.add(Dropout(0.15))
model.add(Dense(25, activation='tanh'))
model.add(Dropout(0.15))
model.add(Dense(1, activation='linear'))

early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min', baseline=None, restore_best_weights=True)
model.compile(optimizer=adam_v2.Adam(learning_rate = 0.001), loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=96, batch_size=32, verbose=1,validation_split=0.2)#,callbacks=[early_stopping])





# Make predictions on the test set
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)

y_pred_train = model.predict(X_train)

y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
y_pred_train = y_pred_train.reshape(y_pred_train.shape[0], y_pred_train.shape[1])





import numpy as np
list_of_arrays = y_pred
list_of_arrays_test = y_test
pred_array = np.concatenate(list_of_arrays, axis=0)
test_array = np.concatenate(list_of_arrays_test,axis = 0)
print(pred_array)
print(test_array)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
 

if __name__ == '__main__':
    x = test_array
    y = pred_array
    fig, ax = plt.subplots(figsize=(5, 5), dpi=600)
    ax.plot((0, 1), (0, 1), linewidth=1, transform=ax.transAxes, ls='--', c='k', label="1:1 line", alpha=0.5)
    ax.plot(x, y, 'o', c='#e25508', markersize=5)
    ax.set_xlabel('Measured values($mmol\cdot L^{-1}$)', fontsize=7)
    ax.set_ylabel("Predicted values($mmol\cdot L^{-1}$)", fontsize=7)
    # 设置图片title
    ax.tick_params(labelsize=7)

 
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(1)
    ax.yaxis.set_major_locator(y_major_locator)

    ax.set(xlim=(-3, 3), ylim=(-3, 3))
 
    plt.savefig("Figure 6-2.jpeg", bbox_inches='tight')




# Calculate the MSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Test RMSE: {rmse}')

r2 = r2_score(y_test, y_pred)
print(f'Test R^2: {r2}')
r2_train = r2_score(y_train, y_pred_train)
print(f' R^2_train: {r2_train}')
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
print(f' MSE_train: {rmse_train}')


def show_history(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)
  plt.figure(figsize=(12, 4),dpi = 600)
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, color = "#2e7ebb", label='Training loss')
  plt.plot(epochs, val_loss, color = "#d92523", label='Test loss')
  plt.title('Training and Test loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig("Figure WOA.jpeg", bbox_inches='tight')



print('EVS:', round(explained_variance_score(y_test, y_pred), 4))
print('AE:', round(mean_absolute_error(y_test, y_pred), 4))




