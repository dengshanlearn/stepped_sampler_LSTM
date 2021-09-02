import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

A = np.load(r'./E:\paper\3\LSTM_video\video-classification\CRNN\CRNN_epoch_training_losses.npy')
D = np.load(r'./E:\paper\3\LSTM_video\video-classification\CRNN\CRNN_epoch_test_score.npy')
# plot
fig = plt.figure(figsize=(5, 5))
#plt.subplot(121)
plt.plot(np.arange(1, 150 + 1), A[:, -1], linewidth=2.0)  # train loss (on epoch end)
#plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
#plt.title("C")
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.ylabel('loss/accuracy')
#plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
#plt.subplot(122)
#plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
plt.plot(np.arange(1, 150 + 1), D, linewidth=2.0)         #  test accuracy (on epoch end)
plt.title("model loss & test accuracy")
plt.xlabel('epochs')
plt.ylabel('loss/accuracy')
plt.legend(['train loss', 'test accuracy'], loc="upper left")

y_major_locator=MultipleLocator(0.5)
ax=plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
# plt.close(fig)
title = r"fig_UCF101_CRNN.png"
plt.savefig(title, dpi=600)
plt.show()
