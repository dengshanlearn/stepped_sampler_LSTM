import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from torch.utils.data.sampler import Sampler, SequentialSampler, ScanningBatchSampler
#from myUCFdataset import VideoDataset


class UCF101_Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, info_list, data_path, folders, transform=None):
        "Initialization"
        self.annot_marks = pd.read_csv(info_list,delimiter=' ', header=None)
        self.data_path = data_path
        #self.labels = labels
        self.folders = folders
        self.transform = transform
        #self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        y = []
        #for item in os.listdir(str(self.folders)):
        for item in selected_folder:
            #i = 1
            #image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            #image = Image.open(os.path.join(path, selected_folder, item))
            #i += 1
            image = Image.open(os.path.join(path, selected_folder, item))
            #if use_transform is not None:
                #image = use_transform(image)

            X.append(image)
            
        loc1 = selected_folder.find('v_')
        loc2 = selected_folder.find('_g')
        label = selected_folder[(loc1 + 2): loc2]
            #label = self.annot_marks.iloc[i, 1]
        y.append(label)
        #X = torch.stack(X, dim=0)
        #y = torch.stack(y, dim=0)
        return X, y

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X, y = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        #y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y



if __name__=='__main__':
    #usage
    root_list='/home/zhonsheng/LSTM_video/video-classification/UCF101-HHTSENG/jpegs_256/'
    annot_list='/home/zhonsheng/LSTM_video/video-classification/UCF101-HHTSENG/class_index.txt'
    image_list = os.listdir(root_list)
    
    #myUCF101=UCF101_Dataset(annot_list,image_list,transform=transforms.Compose([ClipSubstractMean(),Rescale(),RandomCrop(),ToTensor()]))
    myUCF101=UCF101_Dataset(annot_list,root_list,image_list,transform=None)

    dataloader=DataLoader(myUCF101,batch_size=8,shuffle=False,num_workers=12)
    #with open('./result.txt', 'a') as f:
    #file = open('./result.txt', 'a')
    #sys.stdout = file
    f = open('./result.txt', 'a')
    for i, image, label in enumerate(dataloader):
        print(i, image, label)
        #print(i,item.iloc[i,1],item.iloc[i,0])
        #f.write(i,item.iloc[i,1],item.iloc[i,0])
        #f.write(print())
    #file.close()
    f.close()
    
