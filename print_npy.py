import os
import numpy as np

path_A=r'./E:\paper\3\LSTM_video\video-classification\CRNN\CRNN_epoch_training_losses.npy' 
path_D=r'./E:\paper\3\LSTM_video\video-classification\CRNN\CRNN_epoch_test_score.npy'

#data = np.load(datapath).reshape([-1, 2]) # (39, 2)
data_A = np.load(path_A) 
data_D = np.load(path_D)
np.savetxt('CRNN_epoch_training_losses.txt',data_A)
np.savetxt('CRNN_epoch_test_score.txt',data_D)
print ('OK') 
