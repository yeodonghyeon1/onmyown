import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.image as imgs


import torch.nn as nn # 신경망들이 포함됨
import torch.optim as optim # 최적화 알고리즘들이 포함됨
import torch.nn.init as init # 텐서에 초기값을 줌

import torchvision.datasets as datasets # 이미지 데이터셋 집합체
import torchvision.transforms as transforms # 이미지 변환 툴

from torch.utils.data import DataLoader # 학습 및 배치로 모델에 넣어주기 위한 툴


class CNN(nn.Module):
    def __init__(self):
    	# super함수는 CNN class의 부모 class인 nn.Module을 초기화
        super(CNN, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)          
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,100),                                              
            nn.ReLU(),
            nn.Linear(100,10)                                                   
        )       
        
    def forward(self,x):
    	# self.layer에 정의한 연산 수행
        out = self.layer(x)
        # view 함수를 이용해 텐서의 형태를 [100,나머지]로 변환
        out = out.view(batch_size,-1)
        # self.fc_layer 정의한 연산 수행    
        out = self.fc_layer(out)
        return out
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


list_root = []
dic = {}
list_root_label = []
def file_recursively(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if((os.path.join(root,file).find(".jpg")) != -1):
                list_root.append(os.path.join(root ,file))


file_recursively('E:/동현_개인폴더/공부/Pistachio_Image_Dataset/Pistachio_Image_Dataset')

for i in list_root:
    dic[i] = 1 if (i.find("kirmizi") != -1) else 0
    list_root_label.append(dic[i])    

img_end = []

for i in list_root:
    img = imgs.imread(i, cv2.IMREAD_GRAYSCALE)
    min_value = np.min(img)
    max_value = np.max(img)
    output = (img - min_value) / (max_value - min_value)
    img_end.append(output)


x_train, x_valid, y_train, y_valid = train_test_split(img_end, list_root_label, test_size= 0.2, random_state= 47 ,shuffle=True)
learning_rate = 0.001
num_epoch = 10


model = CNN().to(device)

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


loss_arr =[]
j = 0

for i in range(num_epoch):
    for x in x_train:
        x = x
        y= y_train
        
        optimizer.zero_grad()
        
        output = model.forward(x)
        
        loss = loss_func(output,y)
        loss.backward()
        optimizer.step()
        
        if j % 1000 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())
        j += 1
        
    
            