
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


# 鲸鱼优化算法类的定义
class WOA:
    # 类初始化方法
    def __init__(self, trainingPara):

        # 初始化参数
        self.agent = trainingPara["agent"]  # 鲸鱼种群规模大小
        self.ndim = trainingPara["ndim"]  # 维度数量
        self._iter_max = trainingPara["iter_max"]  # 最大迭代次数
        self.pos = trainingPara["pos"]  # 鲸鱼位置
        self.scores = np.zeros([self.agent, 1])  # 初始分值
        self.scores_gBest = float('Inf')  # 全局最好分值
        self.isTrained = False

        # 设置边界
        self.boundarySetting()  # 调用边界设置方法
        X_x = np.random.uniform(low=self.X_min, high=self.X_max, size=[self.agent, self.ndim])  # 生成(0,1)中的随机数
        X_Pra = np.random.uniform(low=self.Pra_min, high=self.Pra_max, size=[self.agent, self.pos])  # 生成(0,1)中的随机数
        self.X = np.concatenate([X_Pra, X_x], axis=1)  # 数组连接

    # 定义优化方法
    def Optimize(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

        self.initialCheckBest()  # 初始化 特征编码  适应度  获取最优值

        # 迭代模型
        self._iter = 0
        while (self._iter < self._iter_max):
            self.whaleOptizeAlgorithm()  # 调用鲸鱼优化算法
            self.boundaryHandle()  # 调用边界处理方法
            self.featureEncoding()  # 调用特征编码方法
            self.fitness()  # 调用适应度方法
            self.checkBest()  # 调用获取最优值方法
            self._iter = self._iter + 1  # 迭代自增加1

        return self.X_gBest

    # 定义鲸鱼优化算法
    def whaleOptizeAlgorithm(self):
        A = 2 * (self._iter_max - self._iter / self._iter_max)  # 定义A系数

        for i in range(self.agent):
            p = np.random.uniform(0.4, 1)  # 生成0.4~1间的随机数 概率
            r = np.random.uniform()  # 生成0~1间的随机数
            C = 2 * r  # C系数
            l = np.random.uniform(-1, 1)  # 生成-1~1间的随机数
            b = 0.1  # 常数 用来定义螺线的形状

            if p >= 0.7:
                # 环绕的猎物
                if np.abs(A) < 1:  # 算法设定当 A < 1  时，鲸鱼向猎物发起攻击。
                    self.X_gBest = self.X[i, :]
                    D = np.abs(r * self.X_gBest - self.X[i, :])  # 鲸鱼和猎物之间的距离
                    self.X[i, :] = self.X_gBest - A * D  # 目前为止最好的位置向量

                # 算法设定当 A ≥ 1  时，随机选择一个搜索代理，根据随机选择的鲸鱼位置来更新其他鲸鱼的位置，迫使鲸鱼偏离猎物，借此找到一个更合适的猎物
                else:
                    rand = np.random.randint(0, i + 1)  # 生成随机整数
                    Xrand = self.X[rand, :]  # 随机选择的鲸鱼位置向量
                    D = np.abs(C * Xrand - self.X[i, :])  # 鲸鱼和猎物之间的距离
                    self.X[i, :] = Xrand - A * D  # 目前为止最好的位置向量

            # 寻找猎物
            else:
                self.X_gBest = self.X[i, :]
                D = np.abs(C * self.X_gBest - self.X[i, :])  # 鲸鱼和猎物之间的距离
                self.X[i, :] = self.X_gBest - A * D  # 目前为止最好的位置向量

    # 定义边界处理方法
    def boundaryHandle(self):
        # 如超出边界 给出一个在边界内的随机值
        mask = (self.X[:, :self.pos] > self.Pra_max) | (self.X[:, :self.pos] < self.Pra_min)  # 大于最大值 或者小于最小值
        self.X[:, 0:self.pos][mask] = np.random.uniform(self.Pra_min, self.Pra_max,
                                                        mask[mask == True].shape[0])  # 边界内随机生成数据赋值

        mask = (self.X[:, self.pos:] > self.X_max) | (self.X[:, self.pos:] < self.X_min)  # 大于最大值 或者小于最小值
        self.X[:, self.pos:][mask] = np.random.uniform(self.X_min, self.X_max,
                                                       mask[mask == True].shape[0])  # 边界内随机生成数据赋值

    # 特征编码
    def featureEncoding(self):
        # 通过sigmoid函数转换为二进制类型
        X_temp = self.X[:, self.pos:].copy()
        rand = np.random.uniform(0, 1)  # 生成0~1之间的随机数
        encoding = 1 / (1 + np.exp(-X_temp))  # 应用sigmoid函数
        X_temp = 1 * (encoding <= rand)

        for i in range(X_temp.shape[0]):
            while (np.sum(X_temp[i, :]) == 0):
                X_temp[i, :] = np.random.randint(2, size=[1, self.ndim])  # 生成随机整数  并赋值

        self.X_feature = (X_temp.copy() == 1)  # 使X_feature的值变为True False

    # 定义适用度方法
    def fitness(self):
        # 计算所有位置适应度
        for i in range(self.X_feature.shape[0]):
            self.scores[i] = 0

            if int(abs(self.X[i][0])) > 0:  # 判断取值
                units = int(abs(self.X[i][0]) / 100) + 10  # 赋值
            else:
                units = int(abs(self.X[i][0]) + 16)  # 赋值
            if int(abs(self.X[i][1])) > 0:  # 判断取值
                epochs = int(abs(self.X[i][1]) / 100) + 10  # 赋值
            else:
                epochs = int(abs(self.X[i][1]) / 100) + 10  # 赋值

            # 建立卷积神经网络回归模型并训练，可调节
            rnn_model = Sequential()
            rnn_model = Sequential()
            rnn_model.add(LSTM(units = 128, input_shape=input_shape, return_sequences=True))
            rnn_model.add(LSTM(units = units, return_sequences=True))
            rnn_model.add(LSTM(units = 32, return_sequences = True))
            rnn_model.add(Dense(units, activation='relu'))
            rnn_model.add(Dropout(0.15))
            rnn_model.add(Dense(units/2, activation='relu'))
            rnn_model.add(Dropout(0.15))
            rnn_model.add(Dense(1, activation='linear'))
            rnn_model.compile(optimizer='adam', loss='mean_squared_error')
            
            rnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=16)  # 拟合
            
            
            y_pred = rnn_model.predict(X_test, batch_size=10)  # 预测
            
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
            score = round(r2_score(y_test, y_pred), 4)

            # 使错误率降到最低
            fitness_value = 1 - score  # 错误率 赋值 适应度函数值

            self.scores[i] = self.scores[i] + fitness_value  # 最终适用度

    # 定义最优值获取方法
    def checkBest(self):
        if np.nanmin(self.scores) < self.scores_gBest:
            indexBest = self.scores.argmin()  # 获取最小值赋值
            self.scores_gBest = self.scores[indexBest].copy()  # 最小损失赋值
            self.X_gBest = self.X.copy()[indexBest, :]  # 获取系数赋值
            self.X_feature_best = self.X_feature[indexBest, :]  # 获取特征赋值

    def initialCheckBest(self):
        self.featureEncoding()  # 特征编码
        self.fitness()  #  调用适用度方法
        self.checkBest()  # 最优值获取

    # 定义边界设置方法
    def boundarySetting(self):
        # X边界设置
        self.X_min = -5  # 边界最小值
        self.X_max = 5  # 边界最大值
        bound_X = np.ones([2, self.ndim])  # 数值全部为1的数组
        bound_X[0, :] = bound_X[0, :] * self.X_min  # 生成最小边界数组值
        bound_X[1, :] = bound_X[1, :] * self.X_max  # 生成最大边界数组值

        # 边界设置
        self.Pra_min = 1  # 边界最小值
        self.Pra_max = 10  # 边界最大值
        bound_Pra = np.ones([2, self.pos])  # 数值全部为1的数组
        bound_Pra[0, :] = bound_Pra[0, :] * self.Pra_min  # 生成最小边界数组值
        bound_Pra[1, :] = bound_Pra[1, :] * self.Pra_max  # 生成最大边界数组值

        self.bounary_X = np.concatenate([bound_Pra, bound_X], axis=1)  # 边界数组连接



df = pd.read_excel('data_ABTS_320.xlsx')
print(df.head())


print(df.info())


print(df.describe())


# In[4]:


y = pd.read_csv('C:/Users/RH/ACE_regression/develop_this/data/y_data_ABTS.csv',header=None).values
X = pd.read_csv('C:/Users/RH/ACE_regression/develop_this/data/data_ABTS_320.csv', delimiter=',',header = None,index_col= 0, skiprows=1).values

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


input_shape = (1, X_train.shape[2])





trainingPara = {
    "agent": 40,  
    "ndim": 2,  
    "pos": len(X_train),  
    "iter_max": 5 
    }

    
opt = WOA(trainingPara)
   
X_gBest=opt.Optimize(X_train, y_train)

if int(abs(X_gBest[0])) > 0: 
        best_units = int(abs(X_gBest[0]) / 100) + 20  
else:
        best_units = int(abs(X_gBest[0]) + 20)  

if int(abs(X_gBest[1])) > 0: 
        best_epochs = int(abs(X_gBest[1])) + 60  
else:
        best_epochs = int(abs(X_gBest[1])) + 100 
print('----------------Results-----------------')
print("The best units is " + str(abs(best_units)))
print("The best epochs is " + str(abs(best_epochs)))





rnn_model = Sequential()
rnn_model = Sequential()
rnn_model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
rnn_model.add(LSTM(best_units, return_sequences=True))
rnn_model.add(LSTM(32,return_sequences = True))
rnn_model.add(Dense(best_units, activation='relu'))
rnn_model.add(Dropout(0.15))
rnn_model.add(Dense(best_units/2, activation='relu'))
rnn_model.add(Dropout(0.15))
rnn_model.add(Dense(1, activation='linear'))
rnn_model.compile(optimizer='adam', loss='mean_squared_error')            

history = rnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_epochs, batch_size=16)

print('********************************************************')
print(rnn_model.summary())  

plot_model(rnn_model, to_file='model.png', show_shapes=True)

 




def show_history(history):
    loss = history.history['loss'] 
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)  
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




