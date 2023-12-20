# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
import pandas as pd
import glob

# データの読み込み
directory_path = "./traf_matrix/"
file_list = os.listdir(directory_path)
filtered_file_list = [file for file in file_list if file.startswith('matrix_16_') and file.endswith('.txt')]


##


#file_pattern = "traf_matrix/matrix_16.*.txt"
#file_paths = glob.glob(file_pattern)

network = TwoLayerNet(input_size=256, hidden_size=50, output_size=4)

iters_num = 100  # 繰り返しの回数を適宜設定する
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

for file_path in filtered_file_list:
    file_path2 = directory_path + file_path #file_path2=./traf_matrix/matrix_16_*.txt
    with open(file_path2,'r') as file:
        lines = file.readlines()
        matrix = []
        for line in lines:
            row = [float(value) for value in line.split()]
            matrix.append(row)
        npmatrix = np.array(matrix)
        x_train = npmatrix.flatten().reshape(1,-1) 
        #ここまででトラフィック行列の取り込み完了

    #base_directory = "traf_matrix/"
    directory_path2 = "./logfiles/"
    file_list2 = os.listdir(directory_path2)
    #print(file_list2)
    matrix_files = [file2 for file2 in file_list2 if file_path in file2]
    #file_pattern_t = os.path.relpath(file_path, base_directory)
    #file_list = os.listdir(directory_path)
    #matrix_files = [file_name for file_name in file_list if file_pattern_t in file_name ]
    first_lines = []
    for matrix_file in matrix_files:
        file_path_t = os.path.join(directory_path2, matrix_file)
        with open(file_path_t, 'r') as file_t:
            first_line = file_t.readline()
            first_lines.append(first_line)
    min_index = first_lines.index(min(first_lines))
    t_train = [1 if i == min_index else 0 for i in range(len(first_lines))]
    t_train = np.array(t_train)
    t_train = t_train.reshape((1, 4))
    #t_train.flatten().reshape(1,-1)
    
    #ここまでで正解ラベルが完了
    
    matrix_64x64 = [[0 for _ in range(64)] for _ in range(64)]
    # 勾配の計算
    #grad = network.numerical_gradient(x_batch, t_batch)
    
    grad = network.gradient(x_train, t_train)
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    loss = network.loss(x_train, t_train)
    print(network.predict(x_train))
    train_loss_list.append(loss)
    train_acc = network.accuracy(x_train, t_train)
    train_acc_list.append(train_acc)
    print("train acc | " + str(train_acc))

# グラフの描画
#markers = {'train': 'o', 'test': 's'}
#x = np.arange(len(train_acc_list))
#plt.plot(x, train_acc_list, label='train acc')
#plt.plot(x, test_acc_list, label='test acc', linestyle='--')
#plt.xlabel("epochs")
#plt.ylabel("accuracy")
#plt.ylim(0, 1.0)
#plt.legend(loc='lower right')
#plt.show()