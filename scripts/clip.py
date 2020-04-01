import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import random

file = open('data/CongreG8/CongreG8.pkl', 'rb')
data = pkl.load(file)
file.close()

train, val, y1, t2 = train_test_split(data, data, test_size = 0.15)
pass
# val = random.sample(range(1,380), int(380*0.15))
# train 

train_data = open("data/CongreG8/train_data.pkl","wb")
pkl.dump(train, train_data)
train_data.close()

val_data = open("data/CongreG8/val_data.pkl","wb")
pkl.dump(val, val_data)
val_data.close()
pass