import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
#from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle



action_name_path = r'./UCF101actions.pkl'
data_path = r"/home/zhonsheng/LSTM_video/video-classification/UCF101-HHTSENG/jpegs_256/" 

with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)

# convert labels -> category
#le = LabelEncoder()
#le.fit(action_names)

# show how many classes there are
#list(le.classes_)

# convert category -> 1-hot
#action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
#enc.fit(action_category)
enc.fit(action_names)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

actions = []
fnames = os.listdir(data_path)

all_names = []
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f[(loc1 + 2): loc2])

    all_names.append(f)


def labels2cat(label_encoder, list):
    return le.transform(list)

# list all data files
all_X_list = all_names                  # all video file names
all_y_list = labels2cat(le, actions)    # all video labels

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

