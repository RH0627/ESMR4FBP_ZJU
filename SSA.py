
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from numpy import loadtxt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import keras.layers as layers  
import keras.backend as K 
from keras.optimizers import adam_v2
import pandas as pd
warnings.filterwarnings("ignore")



def fitness_spaction(parameter, X_train, X_test, y_train, y_test):
    if int(abs(parameter[0])) > 0: 
        units = int(abs(parameter[0]) / 100) + 10  
    else:
        units = int(abs(parameter[0]) + 16)

    if int(abs(parameter[1])) > 0:
        epochs = int(abs(parameter[1]) / 100) + 10  
    else:
        epochs = int(abs(parameter[1]) / 100) + 10 


    lstm = Sequential()
    lstm.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    lstm.add(LSTM(units=64, return_sequences=True) 
    lstm.add(LSTM(units=32, return_sequences=True)         
    lstm.add(Dense(units, activation='tanh')) 
    lstm.add(Dense(units/2,activation='tanh'))
    lstm.add(Dense(1)) 
    lstm.compile(loss='mean_squared_error',
                 optimizer=adam_v2.Adam(learning_rate=0.001),
                 metrics=['mse'])
    lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=16) 
    y_pred = lstm.predict(X_test, batch_size=10) 
    score = round(r2_score(y_test, y_pred), 4) 

    fitness_value = 1 - score

    return fitness_value


def Bounds(s, Lb, Ub):
    te_r = s
    for i in range(len(s)):
        if te_r[i] < Lb[0, i]:  
            te_r[i] = Lb[0, i]  
        elif te_r[i] > Ub[0, i]: 
            te_r[i] = Ub[0, i]

    return te_r


# define SSA
def SSA(pop, M, c, d, dim, fun, X_train, X_test, y_train, y_test):
   
    P_percent = 0.2
    pNum = round(pop * P_percent)
    lb = c * np.ones((1, dim))
    ub = d * np.ones((1, dim))
    X = np.zeros((pop, dim)) 
    fit = np.zeros((pop, 1))
    # 种群循环
    for i in range(pop):
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim) 
        fit[i, 0] = fun(X[i, :], X_train, X_test, y_train, y_test)
    pFit = fit 
    pX = X 
    fMin = np.min(fit[:, 0])
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :]  # 最优位置
    Convergence_curve = np.zeros((1, M))
    # processing
    for t in range(M):
        print('*************************Now', t + 1, '**********************************')
        sortIndex = np.argsort(pFit.T)
        fmax = np.max(pFit[:, 0])
        B = np.argmax(pFit[:, 0]) 
        worse = X[B, :] 
        r2 = np.random.rand(1) 
        if r2 < 0.8: 
    
            for i in range(pNum):
                r1 = np.random.rand(1)  
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] * np.exp(-(i) / (r1 * M)) 
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub) 
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :], X_train, X_test, y_train, y_test)
        elif r2 >= 0.8: 
            
            for i in range(pNum):
                X[sortIndex[0, i], :] = pX[sortIndex[0, i], :] + np.random.rand(1) * np.ones((1, dim))  
                X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)  # 位置边界处理
                fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :], X_train, X_test, y_train, y_test)  
        bestII = np.argmin(fit[:, 0]) 
        bestXX = X[bestII, :] 
        for ii in range(pop - pNum):
            i = ii + pNum
            A = np.floor(np.random.rand(1, dim) * 2) * 2 - 1 
            if i > pop / 2:
                X[sortIndex[0, i], :] = np.random.rand(1) * np.exp(worse - pX[sortIndex[0, i], :] / np.square(i))
            else:
                X[sortIndex[0, i], :] = bestXX + np.dot(np.abs(pX[sortIndex[0, i], :] - bestXX),
                                                        1 / (A.T * np.dot(A, A.T))) * np.ones((1, dim))
            X[sortIndex[0, i], :] = Bounds(X[sortIndex[0, i], :], lb, ub)
            fit[sortIndex[0, i], 0] = fun(X[sortIndex[0, i], :], X_train, X_test, y_train, y_test)
        arrc = np.arange(len(sortIndex[0, :]))

        c = np.random.permutation(arrc) 
        b = sortIndex[0, c[0:20]]
        for j in range(len(b)):
            if pFit[sortIndex[0, b[j]], 0] > fMin:
                X[sortIndex[0, b[j]], :] = bestX + np.random.rand(1, dim) * np.abs(
                    pX[sortIndex[0, b[j]], :] - bestX) 
            else: 
                X[sortIndex[0, b[j]], :] = pX[sortIndex[0, b[j]], :] + (2 * np.random.rand(1) - 1) * np.abs(
                    pX[sortIndex[0, b[j]], :] - worse) / (pFit[sortIndex[0, b[j]]] - fmax + 10 ** (-50))  
            X[sortIndex[0, b[j]], :] = Bounds(X[sortIndex[0, b[j]], :], lb, ub)  # 位置边界处理
            fit[sortIndex[0, b[j]], 0] = fun(X[sortIndex[0, b[j]]], X_train, X_test, y_train, y_test) 
        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :] 
            if pFit[i, 0] < fMin:
                fMin = pFit[i, 0]
                bestX = pX[i, :]  
        Convergence_curve[0, t] = fMin  
    return fMin, bestX, Convergence_curve 



if __name__ == "__main__":
    data = df = pd.read_excel('data_ABTS_320.xlsx')



X = data.drop(columns=['y'])
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train) 







SearchAgents_no = 10  
Max_iteration = 1  
dim = 2  # 优化参数的个数
lb = [10 ** (-1), 2 ** (-5)]  
ub = [10 ** 1, 2 ** 4]  

fMin, bestX, SSA_curve = SSA(SearchAgents_no, Max_iteration, lb, ub, dim, fitness_spaction, X_train, X_test,
                             y_train, y_test)

if int(abs(bestX[0])) > 0: 
    best_units = int(abs(bestX[0]) / 100) + 50  
else:
    best_units = int(abs(bestX[0]) + 50)  

if int(abs(bestX[1])) > 0:  
    best_epochs = int(abs(bestX[1])) + 80  
else:
    best_epochs = int(abs(bestX[1])) + 100  

print('----------------Results-----------------')
print("The best units is " + str(abs(best_units)))
print("The best epochs is " + str(abs(best_epochs)))



lstm = Sequential()  
lstm.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm.add(LSTM(units=64, return_sequences=True)
lstm.add(LSTM(units=64, return_sequences=True)
         
lstm.add(LSTM(units=32, return_sequences=True)
lstm.add(Dense(best_units, activation='tanh'))
lstm.add(Dense(best_units/2, activation='tanh'))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error',
                 optimizer=adam_v2.Adam(learning_rate=0.001),
                 metrics=['mse'])
history = lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_epochs, batch_size=16)
print('********************************************************')
print(lstm.summary())
plot_model(lstm, to_file='model.png', show_shapes=True)





def show_history(history):
    loss = history.history['loss'] 
    val_loss = history.history['val_loss'] 
    plt.figure(figsize=(12, 4))  
    plt.subplot(1, 2, 1) 
    plt.plot(epochs, loss, 'r', label='Training loss') 
    plt.plot(epochs, val_loss, 'b', label='Test loss') 
    plt.title('Training and Test loss') 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.legend() 
    plt.show() 

show_history(history)


