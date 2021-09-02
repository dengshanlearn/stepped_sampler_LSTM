import os
import numpy as np
import pandas as pd

import pickle
#fr=open('./UCF101_videos_prediction.pkl','rb')
df = pd.read_pickle("./UCF101_videos_prediction.pkl")
#inf = pickle.load(fr)
#doc = open(inf, 'a')
#print(inf, file=doc)
df.to_csv('UCF101_videos_prediction.txt', sep='\t',index=False, header=None)


