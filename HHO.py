
from numpy.random import rand
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import keras.layers as layers  
import keras.backend as K 
from keras.optimizers import adam_v2
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout
from numpy import loadtxt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score 
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization
import math
warnings.filterwarnings("ignore")

# 定义初始化位置函数
def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')  # 位置初始化为0
    for i in range(N):  # 循环
        for d in range(dim):  # 循环
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()  # 位置随机初始化

    return X  # 返回位置数据


# 定义转换函数
def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')  # 位置初始化为0
    for i in range(N):  # 循环
        for d in range(dim):  # 循环
            if X[i, d] > thres:  # 判断
                Xbin[i, d] = 1  # 赋值
            else:
                Xbin[i, d] = 0  # 赋值

    return Xbin  # 返回数据


# 定义边界处理函数
def boundary(x, lb, ub):
    if x < lb:  # 小于最小值
        x = lb  # 赋值最小值
    if x > ub:  # 大于最大值
        x = ub  # 赋值最大值

    return x  # 返回位置数据


# 定义莱维飞行函数
def levy_distribution(beta, dim):
    # Sigma计算赋值
    nume = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)  # 计算
    deno = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)  # 计算
    sigma = (nume / deno) ** (1 / beta)  # Sigma赋值
    # Parameter u & v
    u = np.random.randn(dim) * sigma  # u参数随机赋值
    v = np.random.randn(dim)  # v参数随机赋值
    # 计算步骤
    step = u / abs(v) ** (1 / beta)  # 计算
    LF = 0.01 * step  # LF赋值

    return LF  # 返回数据


# 定义错误率计算函数
def error_rate(X_train, y_train, X_test, y_test, x, opts):
    if abs(x[0]) > 0:  # 判断取值
        units = int(abs(x[0])) * 10  # 赋值
    else:
        units = int(abs(x[0])) + 16  # 赋值

    if abs(x[1]) > 0:  # 判断取值
        epochs = int(abs(x[1])) * 10  # 赋值
    else:
        epochs = int(abs(x[1])) + 10  # 赋值

    # 建支持LSTM模型并训练
    lstm = Sequential()  # 序贯模型
    lstm.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # LSTM层
    lstm.add(LSTM(units=64, return_sequences=True))
    lstm.add(LSTM(units=32))
    lstm.add(Dense(units, activation='tanh'))  # 全连接层
    lstm.add(Dense(units/2, activation='tanh'))  # 全连接层
    lstm.add(Dense(1))  # 输出层
    lstm.compile(loss='mean_squared_error',
                 optimizer=adam_v2.Adam(learning_rate=0.001),
                 metrics=['mse'])  # 编译
    lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=16)  # 拟合
    y_pred = lstm.predict(X_test, batch_size=10)  # 预测
    score = round(r2_score(y_test, y_pred), 4)  # 计算R2

    # 使错误率降到最低
    fitness_value = 1 - score  # 错误率 赋值 适应度函数值

    return fitness_value  # 返回适应度


# 定义目标函数
def Fun(X_train, y_train, X_test, y_test, x, opts):
    # 参数
    alpha = 0.99  # 赋值
    beta = 1 - alpha  # 赋值
    # 原始特征数
    max_feat = len(x)
    # 选择特征数
    num_feat = np.sum(x == 1)
    # 无特征选择判断
    if num_feat == 0:  # 判断
        cost = 1  # 赋值
    else:
        # 调用错误率计算函数
        error = error_rate(X_train, y_train, X_test, y_test, x, opts)
        # 目标函数计算
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost  # 返回数据


# 定义哈里斯鹰优化算法主函数
def jfs(X_train, y_train, X_test, y_test, opts):
    # 参数
    ub = 1  # 上限
    lb = 0  # 下限
    thres = 0.5  # 阀值
    beta = 1.5  # levy 参数

    N = opts['N']  # 种群数量
    max_iter = opts['T']  # 最大迭代次数
    if 'beta' in opts:  # 判断
        beta = opts['beta']  # 赋值

    # 维度
    dim = np.size(X_train, 1)  # 获取维度
    if np.size(lb) == 1:  # 判断
        ub = ub * np.ones([1, dim], dtype='float')  # 初始化上限为1
        lb = lb * np.ones([1, dim], dtype='float')  # 初始化下限为1

    # 调用位置初始化函数
    X = init_position(lb, ub, N, dim)

    fit = np.zeros([N, 1], dtype='float')  # 适应度初始化为0
    Xrb = np.zeros([1, dim], dtype='float')  # 猎物位置初始化为0
    fitR = float('inf')  # 初始化为无穷

    curve = np.zeros([1, max_iter], dtype='float')  # 适应度初始化为0
    t = 0  # 赋值

    while t < max_iter:  # 循环
        # 调用转换函数
        Xbin = binary_conversion(X, thres, N, dim)

        # 计算适应度
        for i in range(N):  # 循环
            fit[i, 0] = Fun(X_train, y_train, X_test, y_test, Xbin[i, :], opts)  # 调用目标函数
            if fit[i, 0] < fitR:  # 判断
                Xrb[0, :] = X[i, :]  # 猎物位置赋值
                fitR = fit[i, 0]  # 适应度赋值

        # 存储结果
        curve[0, t] = fitR.copy()  # 复制
        print("*********************************", "当前迭代次数: ", t + 1, "***************************************")
        print("最好的适应度数值: ", curve[0, t])
        t += 1

        # 平均位置
        X_mu = np.zeros([1, dim], dtype='float')  # 初始化为0
        X_mu[0, :] = np.mean(X, axis=0)  # 计算平均位置

        for i in range(N):  # 循环
            E0 = -1 + 2 * rand()  # 猎物的初始能量  [-1,1] 之间的随机数
            E = 2 * E0 * (1 - (t / max_iter))  # 逃逸能量
            # 当|E|≥1 时进入搜索阶段
            if abs(E) >= 1:
                q = rand()  # 生成随机数 [0,1]
                if q >= 0.5:  # 判断
                    k = np.random.randint(low=0, high=N)  # 生成随机整数  个体
                    r1 = rand()  # [0,1]之间的随机数
                    r2 = rand()  # [0,1]之间的随机数
                    for d in range(dim):  # 循环
                        X[i, d] = X[k, d] - r1 * abs(X[k, d] - 2 * r2 * X[i, d])  # 更新位置
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  # 边界处理

                elif q < 0.5:  # 判断
                    r3 = rand()  # [0,1]之间的随机数
                    r4 = rand()  # [0,1]之间的随机数
                    for d in range(dim):  # 循环
                        X[i, d] = (Xrb[0, d] - X_mu[0, d]) - r3 * (lb[0, d] + r4 * (ub[0, d] - lb[0, d]))  # 更新位置
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  # 边界处理


            elif abs(E) < 1:  # 开发阶段
                J = 2 * (1 - rand())  # 生成随机数
                r = rand()  # 生成随机数
                # 软围攻策略进行位置更新
                if r >= 0.5 and abs(E) >= 0.5:
                    for d in range(dim):  # 循环
                        DX = Xrb[0, d] - X[i, d]  # 猎物位置与个体当前位置的差值
                        X[i, d] = DX - E * abs(J * Xrb[0, d] - X[i, d])  # 更新位置
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  # 边界处理

                # 硬围攻策略进行位置更新
                elif r >= 0.5 and abs(E) < 0.5:
                    for d in range(dim):  # 循环
                        DX = Xrb[0, d] - X[i, d]  # 猎物位置与个体当前位置的差值
                        X[i, d] = Xrb[0, d] - E * abs(DX)  # 更新位置
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])  # 边界处理

                # 渐近式快速俯冲的软包围策略进行位置更新
                elif r < 0.5 and abs(E) >= 0.5:
                    LF = levy_distribution(beta, dim)  # 莱维飞行
                    Y = np.zeros([1, dim], dtype='float')  # 初始化为1
                    Z = np.zeros([1, dim], dtype='float')  # 初始化为1

                    for d in range(dim):  # 循环

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X[i, d])  # 更新位置

                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])  # 边界处理

                    for d in range(dim):  # 循环

                        Z[0, d] = Y[0, d] + rand() * LF[d]  # 更新位置

                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])  # 边界处理

                        # 调用转换函数
                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    # 适应度计算
                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)
                    # 根据适应度进行判断
                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY  # 赋值
                        X[i, :] = Y[0, :]  # 赋值
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ  # 赋值
                        X[i, :] = Z[0, :]  # 赋值

                # 带有莱维飞行的硬围攻策略进行位置更新
                elif r < 0.5 and abs(E) < 0.5:
                    # Levy distribution (9)
                    LF = levy_distribution(beta, dim)  # 莱维飞行
                    Y = np.zeros([1, dim], dtype='float')  # 初始化为0
                    Z = np.zeros([1, dim], dtype='float')  # 初始化为0

                    for d in range(dim):  # 循环

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X_mu[0, d])  # 更新位置

                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])  # 边界处理

                    for d in range(dim):  # 循环

                        Z[0, d] = Y[0, d] + rand() * LF[d]  # 更新位置

                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])  # 边界处理

                        # 调用转换函数
                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    # 适应度计算
                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)
                    # 根据适应度进行判断
                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY  # 赋值
                        X[i, :] = Y[0, :]  # 赋值
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ  # 赋值
                        X[i, :] = Z[0, :]  # 赋值

    return X  # 返回数据


if __name__ == '__main__':
    # 读取数据
    df = pd.read_excel('data_ABTS_320.xlsx')

    # 用Pandas工具查看数据
    print(df.head())

    # 查看数据集摘要
    print(df.info())

    # 数据描述性统计分析
    print(df.describe())

    # y变量分布直方图
    fig = plt.figure(figsize=(8, 5))  # 设置画布大小
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_tmp = df['y']  # 过滤出y变量的样本
    # 绘制直方图  bins：控制直方图中的区间个数 auto为自动填充个数  color：指定柱子的填充色
    plt.hist(data_tmp, bins='auto', color='g')
    plt.xlabel('y')
    plt.ylabel('数量')
    plt.title('y变量分布直方图')
    plt.show()

    # 数据的相关性分析

    #sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
    #plt.title('相关性分析热力图')
    #plt.show()

    # 提取特征变量和标签变量
    y = df['y']
    X = df.drop('y', axis=1)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)  # 增加维度

    print('***********************查看训练集的形状**************************')
    print(X_train.shape)  # 查看训练集的形状

    X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)  # 增加维度
    print('***********************查看测试集的形状**************************')
    print(X_test.shape)  # 查看测试集的形状

    # 参数初始化
    N = 10  # 种群数量
    T = 2  # 最大迭代次数

    opts = {'N': N, 'T': T}

    # 调用哈里斯鹰优化算法主函数
    fmdl = jfs(X_train, y_train, X_test, y_test, opts)

    if abs(fmdl[0][0]) > 0:  # 判断
        best_units = int(abs(fmdl[0][0])) * 10 + 48  # 赋值
    else:
        best_units = int(abs(fmdl[0][0])) + 48  # 赋值

    if abs(fmdl[0][1]) > 0:  # 判断
        best_epochs = int(abs(fmdl[0][1])) * 10 + 60  # 赋值
    else:
        best_epochs = (int(abs(fmdl[0][1])) + 100)  # 赋值

    print('----------------HHO哈里斯鹰优化算法优化LSTM模型-最优结果展示-----------------')
    print("The best units is " + str(abs(best_units)))
    print("The best epochs is " + str(abs(best_epochs)))




# 应用优化后的最优参数值构建LSTM回归模型
lstm = Sequential()  # 序贯模型
lstm.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # LSTM层
lstm.add(LSTM(units=128, return_sequences=True)
lstm.add(LSTM(units=32))  # LSTM层
lstm.add(Dense(best_units, activation='tanh'))  # 全连接层
lstm.add(Dense(1))  # 输出层
lstm.compile(loss='mean_squared_error',
             optimizer='adam',
             metrics=['mse'])  # 编译
history = lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_epochs, batch_size=16)  # 拟合
print('*************************输出模型摘要信息*******************************')
print(lstm.summary())  # 输出模型摘要信息

plot_model(lstm, to_file='model.png', show_shapes=True)  # 保存模型结构信息


# 定义绘图函数：损失曲线图和准确率曲线图
def show_history(history):
    loss = history.history['loss']  # 获取损失
    val_loss = history.history['val_loss']  # 测试集损失
    epochs = range(1, len(loss) + 1)  # 迭代次数
    plt.figure(figsize=(12, 4))  # 设置图片大小
    plt.subplot(1, 2, 1)  # 增加子图
    plt.plot(epochs, loss, 'r', label='Training loss')  # 绘制曲线图
    plt.plot(epochs, val_loss, 'b', label='Test loss')  # 绘制曲线图
    plt.title('Training and Test loss')  # 设置标题名称
    plt.xlabel('Epochs')  # 设置x轴名称
    plt.ylabel('Loss')  # 设置y轴名称
    plt.legend()  # 添加图例
    plt.show()  # 显示图片


show_history(history)  # 调用绘图函数



