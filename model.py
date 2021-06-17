# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:35:48 2021

@author: 24233
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations
from mne.decoding import CSP
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Neural Networks model : Bidirection GRU
"""


class BiGRU(nn.Module):

    def __init__(self):
        super(BiGRU, self).__init__()
        # V = args.embed_num
        self.Cls = 3      #args.class_num
        self.BN0 = nn.BatchNorm1d(num_features = 3)
        self.conv1 = nn.Conv2d(1, 8, (3, 1))    #kernel_size need to be adjust
        #self.maxp1 = nn.MaxPool2d(kernel_size = (4,1))
        #self.ReLU1 = nn.LeakyReLU(inplace=True)
        # gru
        #self.BN1 = nn.BatchNorm1d(num_features = 16)               #middle of the dimension
        #self.lstm=
        self.conv2 = nn.Conv1d(8, 40, 10,stride=10)
        # self.maxp2 = nn.MaxPool2d(kernel_size = (4,1))                 #Pooling in the last dim
        # self.ReLU2 = nn.LeakyReLU(inplace=True)
        #self.bigru = nn.GRU(input_size = 32, hidden_size = 32, num_layers=2, bidirectional=True, bias=False)
        # linear
        #self.ReLU2 = nn.LeakyReLU(inplace=True)
        self.maxp3 = nn.MaxPool2d(kernel_size = (8,1)) 
        self.hidden1= nn.Linear(80, self.Cls)

    def forward(self, input1):

        # conv1
        # print("input:",  input.shape)
        input1 = self.BN0(input1)

        process1=input1.unsqueeze(0)
        process1 = self.conv1(process1)
        #process1=process1.permute(0,2,1,3)
        #process1=self.maxp1(self.ReLU1(process1))
        
        process1=process1.reshape(1,-1,164)
        
        process1 = self.conv2(process1)
        process1 = self.maxp3(process1)
        #process1 = self.BN1(process1)
        
        #process1=process1.unsqueeze(0)
        #process1=self.maxp1(process1)
        #process1=self.ReLU1(process1)
        # process2 = self.conv2(process1)
        # process2 = process2.squeeze().unsqueeze(0).unsqueeze(0)
        # process2 = self.maxp2(process2)
        # process2=process2.squeeze().unsqueeze(0)
        # process2 = self.ReLU2(process2)


        # print("process2:", process2.shape) #16*16*126
       
        # print(lstm_out.shape)
        # print(lstm_out.shape)             #16*64*56
        # linear
        # print(lstm_out.shape)
        y = self.hidden1(process1.reshape(-1,80))
        # print(y.shape)
        #y = F.softmax(y, dim = 1)
        
        # print(y.shape)
        logit = y

        return logit

class classifier():
    def __init__(self):
        self.dataset_path=load_path('./原始数据-3-3-640')
        self.dataset=load_data(self.dataset_path)
        self.train_data,self.test_data=split_train(self.dataset,0.2)
        self.train_loss=[]
        self.loss=nn.CrossEntropyLoss()
        self.test_accuracy=[]
        self.net=BiGRU().cuda()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.005, weight_decay=1e-7)
        #torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 480, gamma=0.9, last_epoch=-1)
    
    def train(self,iterations):
        self.net.train()
        acc_max = 0
        length1=len(self.train_data)
        length2=len(self.test_data)
        for i in range(iterations):
            train_loss=[]
            for j in range(length1):
                data,label=self.train_data[j]
                data=torch.FloatTensor(data).to(device)
                data.requires_grad=True
                label=torch.Tensor([label]).long().to(device)
                res=self.net.forward(data)
                self.optimizer.zero_grad()
                loss=self.loss(res, label)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.cpu().detach().numpy())
            av_loss=np.array(train_loss).mean()
            self.train_loss.append(av_loss)
            print('loss:{0}'.format(av_loss))
            res_accuracy=np.zeros(2)
            for k in range(length2):
                data,label=self.test_data[k]
                data=torch.FloatTensor(data).to(device)
                label=torch.Tensor([label]).long().to(device)
                res=self.net.forward(data)
                if torch.argmax(res)==label:
                    res_accuracy[0]+=1
                else:
                    res_accuracy[1]+=1
            accuracy=res_accuracy[0]/res_accuracy.sum()
            if (accuracy >= acc_max):
                torch.save(self.net,  './model/EEG_Best_Net_2_164.pth')
            self.test_accuracy.append(accuracy)
            print('accuracy:{0}%'.format(accuracy*100))
            if i%10==0:
                plt.plot(self.train_loss)
                plt.show()
                plt.plot(self.test_accuracy)
                plt.show()
                
                
        



def load_path(fname):
    names=os.listdir(fname)
    fpath=[]
    for i in names:
        fpath.append(fname+'/'+i)
    return fpath

def load_data(path_list):
    g=[]
    for i in path_list:
        g.append(np.load(i,allow_pickle=True))
    return np.array(g)

def split_train(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data[train_indices],data[test_indices]

if __name__ == '__main__':
    p=classifier()
    p.train(1000)