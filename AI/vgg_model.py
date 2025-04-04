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


class CustomDataset(Dataset): 
    def __init__(self, x_data, y_data):
        self.x_train = np.array(x_data)
    
        # self.x_train = np.transpose(self.x_train, (2, 0 , 1))
        self.y_train = np.array(y_data)
        self.x_train = torch.from_numpy(self.x_train).float()
        self.x_train = torch.unsqueeze(self.x_train, 1)
        print(self.x_train.shape)
        self.transform = ToTensor()
        self.target_transform = ToTensor()
    def __len__(self):
            return len(self.x_train)
    def __getitem__(self, idx):
            if self.transform:
                x = self.x_train
            if self.target_transform:
                y = torch.from_numpy(self.y_train[idx]).float()
            return x, y


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        output_channel = 512
        input_channel = 1
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
        int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=self.output_channel[0], kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(in_channels=self.output_channel[0], out_channels=self.output_channel[1], kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(in_channels=self.output_channel[1], out_channels=self.output_channel[2],kernel_size=3, stride=1, padding=1), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(in_channels=self.output_channel[2], out_channels=self.output_channel[2], kernel_size=3, stride=1, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(in_channels=self.output_channel[2], out_channels=self.output_channel[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(in_channels=self.output_channel[3], out_channels=self.output_channel[3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(in_channels=self.output_channel[3], out_channels=self.output_channel[3], kernel_size=3, stride=1, padding=1), nn.ReLU(True))  # 512x1x24
        
        self.fc_layer = nn.Sequential(
        	# [100,64*3*3] -> [100,100]
            nn.Linear(32768,100),                                              
            nn.ReLU(),
            nn.Linear(100,10),  
            nn.ReLU(),
            nn.Linear(10,2)                                                   
        )     
    def forward(self, x):
        out = self.ConvNet(x)
        out = torch.flatten(out, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        out = self.fc_layer(out)
        return out
    
    
def file_recursively(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if((os.path.join(root,file).find(".jpg")) != -1):
                list_root.append(os.path.join(root ,file))
             
                
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model.forward(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
                 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")   

list_root = []
dic = {}
list_root_label = []

file_recursively('E:/동현_개인폴더/공부/Pistachio_Image_Dataset/Pistachio_Image_Dataset')


for i in list_root:
    dic[i] = 1 if (i.find("kirmizi") != -1) else 0
    list_root_label.append(dic[i])    
    
img_end = []


for i in list_root:
    img = imgs.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_LINEAR)
    min_value = np.min(img)
    max_value = np.max(img)
    output = (img - min_value) / (max_value - min_value)
    img_end.append(output)
   

x_train, x_valid, y_train, y_valid = train_test_split(img_end, list_root_label, test_size= 0.2, random_state= 47 ,shuffle=True)
        
        
batch_size = 16
            
dataset_train = CustomDataset(x_train, y_train)
dataset_valid = CustomDataset(x_valid, y_valid)


dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=0, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, num_workers=0, shuffle=True)
model = NeuralNetwork().to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

print(dataloader_train)
summary(model, (1,64,64))


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_func = nn.CrossEntropyLoss()

# dataloader_train = dataloader_train.squeeze(dim=0)


nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch, (x, y) in enumerate(dataloader_train):
        x_train = x.to(device)
        y = y.type(torch.LongTensor)
        y_train = y.to(device)
        # H(x) 계산
        optimizer.zero_grad()

        pred = model.forward(x_train)
        
        loss = loss_func(pred, y_train)
        
        # 역전파
        print(batch)
        loss.backward()
        optimizer.step()
        correct,test_loss = 0,0
        if batch % 2 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(x_train):>5d}]")
 
    print("에포크 한번")
    
PATH = './vgg_net.pth'
torch.save(model.state_dict(), PATH)
    
model.eval()