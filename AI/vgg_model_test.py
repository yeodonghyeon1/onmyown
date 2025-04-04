import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import matplotlib.image as imgs
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary as summary
from torchvision import transforms
from PIL import Image
import sys
from vgg_model_copy import NeuralNetwork
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import platform


device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
model = torch.load('./model/vgg_best.pth')
model = NeuralNetwork().to(device)

model.eval().to(device)
img =  Image.open('./siirt.jpg').convert("L")
img = img.resize((64, 64))


img = np.array(img)
img = torch.from_numpy(img).float().to(device)
img  = torch.unsqueeze(img, 0)
img  = torch.unsqueeze(img, 0)

output = model.forward(img)
_, predicted = torch.max(output.data, 1)
print(predicted)