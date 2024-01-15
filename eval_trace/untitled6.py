# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:39:25 2023

@author: do927
"""

import torch
import torch.nn.functional as f
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
#from two_layer_net import TwoLayerNet
import pandas as pd
import glob
from typing_extensions import Self

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = torch.nn.Linear(16*16, 8)
        self.fc2 = torch.nn.Linear(8, 16)
 
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        #x = torch.relu(x)
        x = self.fc2(x)
        #x = torch.sigmoid(x)
 
        return x #f.log_softmax(x, dim=1)
    
def load_matrix():
    tensor_list_x = []
    tensor_list_t = []
    for file_path in filtered_file_list:
        file_path2 = directory_path + file_path #file_path2=./traf_matrix/matrix_16_*.txt
        with open(file_path2,'r') as file:
            lines = file.readlines()
            matrix = []
            for line in lines:
                row = [float(value) for value in line.split()]
                matrix.append(row)
            npmatrix = np.array(matrix)
            tensor_data_x = torch.tensor(npmatrix).unsqueeze(0).float()
            tensor_list_x.append(tensor_data_x)
        directory_path2 = "./logfiles/"
        file_list2 = os.listdir(directory_path2)
        matrix_files = [file2 for file2 in file_list2 if file_path in file2]
        first_lines = []
        for matrix_file in matrix_files:
            file_path_t = os.path.join(directory_path2, matrix_file)
            with open(file_path_t, 'r') as file_t:
                first_line = file_t.readline()
                first_lines.append(float(first_line))
        #min_index = first_lines.index(min(first_lines))
        #t_train = [1 if i == min_index else 0 for i in range(len(first_lines))]
        t_train = np.array(first_lines)
        #t_train = t_train.reshape((1, 18))
        tensor_data_t = torch.tensor(t_train).unsqueeze(0).float()
        tensor_list_t.append(tensor_data_t)
    
    half_index = len(tensor_list_x) //2
    #t_x_train = tensor_list_x[:half_index]
    t_x_train = torch.cat(tensor_list_x[:half_index])
    t_t_train = torch.cat(tensor_list_t[:half_index])
    t_x_valid = torch.cat(tensor_list_x[half_index:])
    t_t_valid = torch.cat(tensor_list_t[half_index:])
    
    #t_t_train = tensor_list_t[:half_index]
    #t_x_valid = tensor_list_x[half_index:]
    #t_t_valid = tensor_list_t[half_index:]
    #print(torch.is_tensor(t_x_train))
    dataset_train = torch.utils.data.TensorDataset(t_x_train,t_t_train)
    dataset_valid = torch.utils.data.TensorDataset(t_x_valid,t_t_valid)
    train_loader = DataLoader(dataset_train,batch_size=BATCH_SIZE,shuffle=False)
    test_loader = DataLoader(dataset_valid,batch_size=BATCH_SIZE,shuffle=False)

    return {'train': train_loader, 'test': test_loader}

directory_path = "./traf_matrix/"
file_list = os.listdir(directory_path)
filtered_file_list = [file for file in file_list if file.startswith('matrix_16_') and file.endswith('.txt')]
BATCH_SIZE=20
#BATCH_SIZE=5

original_stdout = sys.stdout
file_name = 'output.txt'

result_vertical = pd.DataFrame()

if __name__ == '__main__':
    # 学習回数
    epoch = 100
    #epoch = 200

    # 学習結果の保存用
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
        'top_4_acc': [],
    }

    # ネットワークを構築
    net: torch.nn.Module = MyNet()

    # MNISTのデータローダーを取得
    loaders = load_matrix()
    #print(len(loaders['train'].dataset))

    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    
    original_stdout = sys.stdout
    file_name = 'output.txt'
    
    #with open(file_name, 'w') as f:
        #sys.stdout = f

    for e in range(epoch):

        """ Training Part"""
        loss = None
        # 学習開始 (再開)
        net.train(True)  # 引数は省略可能
        
        for i, (t_x_train, t_t_train) in enumerate(loaders['train']):
            # 全結合のみのネットワークでは入力を1次元に
            # print(data.shape)  # torch.Size([128, 1, 28, 28])
            #print(t_x_train.shape)
            data_batch_1 = t_x_train

            t_x_train = t_x_train.view(-1, 16*16)
            # print(data.shape)  # torch.Size([128, 784])
            
            optimizer.zero_grad()
            output = net(t_x_train)
            criterion = torch.nn.MSELoss()
            #criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(output, t_t_train)
            #loss = f.nll_loss(output, t_t_train)
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Training log: {} epoch ({} / 1000 train. data). Loss: {}'.format(e+1,
                                                                                         (i+1)*20,
                                                                                         loss.item())
                      )
            
            data_tensor_1 = output
            data_tensor_2 = t_t_train
        selected_matrix_1 = data_batch_1[19, :, :]
        selected_tensor_1 = data_tensor_1[19, :]
        selected_tensor_2 = data_tensor_2[19, :]
        df_matrix_1 = pd.DataFrame(selected_matrix_1.detach().numpy())
        df_tensor_1 = pd.DataFrame(selected_tensor_1.detach().numpy())
        df_tensor_2 = pd.DataFrame(selected_tensor_2.detach().numpy())
        empty_row = pd.DataFrame([""] * len(df_matrix_1))
        empty_row.columns = [None] * len(empty_row.columns)
        empty_row_2 = pd.DataFrame([""] * df_matrix_1.shape[1]).T
        empty_row_2.columns = df_matrix_1.columns
        result_df_2 = pd.concat([df_matrix_1, empty_row, df_tensor_1, empty_row, df_tensor_2], axis = 1, ignore_index=True)
        result_vertical = pd.concat([result_vertical ,result_df_2, empty_row_2], axis=0, ignore_index=True)

        history['train_loss'].append(loss)

        """ Test Part """
        # 学習のストップ
        net.eval()  # または net.train(False) でも良い
        test_loss = 0
        correct1 = 0
        correct2 = 0

        with torch.no_grad():
            for data, target in loaders['test']:
                data = data.view(-1, 16 * 16)
                output = net(data)
                test_loss += criterion(output, target).item()
                pred1 = output.argmin(dim=1, keepdim=True)
                pred3,pred4 = torch.topk(output, 4, largest=False)
                #print(output)
                #print(pred1)
                pred2 = target.argmin(dim=1, keepdim=True)
                #print(pred2)
                #print(torch.sum(pred1 == pred2).item())
                correct1 += torch.sum(pred4 == pred2).item()
                correct2 += torch.sum(pred1 == pred2).item()
                #print(output)
                #print(target)
                #correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= 1000
        print('Test loss (avg): {}, Accuracy: {}, Top 4 Accuracy: {}'.format(test_loss,
                                                         correct2 / 1000, correct1/1000))

        history['test_loss'].append(test_loss)
        history['test_acc'].append(correct2 / 1000)
        history['top_4_acc'].append(correct1 / 1000)
        
        train_loss_values = [loss_item.detach().numpy() for loss_item in history['train_loss']]
        #test_loss_values = [loss_item.detach().numpy() for loss_item in history['test_loss']]
        #test_acc_values = [loss_item.detach().numpy() for loss_item in history['test_acc']]
        

    # 結果の出力と描画
    #print(history['train_loss'])
    #print(train_loss_values)
    plt.figure()
    #plt.plot(range(1, epoch+1), history['train_loss'], label='train_loss')
    plt.plot(range(1, epoch+1), train_loss_values , label='train_loss')
    plt.plot(range(1, epoch+1), history['test_loss'] , label='test_loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(range(1, epoch+1), history['test_acc'])
    plt.title('test accuracy')
    plt.xlabel('epoch')
    plt.savefig('test_acc.png')
    
    plt.figure()
    plt.plot(range(1, epoch+1), history['top_4_acc'])
    plt.title('top 4 accuracy')
    plt.xlabel('epoch')
    plt.savefig('top_4_acc.png')
    result_vertical.to_excel('output_data.xlsx', index=False, header=False)
    #sys.stdout = original_stdout