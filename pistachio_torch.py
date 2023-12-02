import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import matplotlib.image as imgs
from sklearn.model_selection import train_test_split




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

batch_size = 64
train_loader = torch.utils.data.DataLoader(x_train, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(x_valid, batch_size=8, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break