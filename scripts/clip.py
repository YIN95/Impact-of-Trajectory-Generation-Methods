import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import random

file = open('data/CongreG8/CongreG8.pkl', 'rb')
data = pkl.load(file)
file.close()

x = []
y = []
for data_sample in data:
    x.append(data_sample[:, :, :, :, 1:])
    y.append(data_sample[:, [0, 2], :, 5, 0])


train_x, val_x, train_y, val_y = train_test_split(x, y, test_size = 0.15)
pass
# val = random.sample(range(1,380), int(380*0.15))
# train 
train_label = open("data/CongreG8/train_label.pkl","wb")
pkl.dump(train_y, train_label)
train_label.close()

val_label = open("data/CongreG8/val_label.pkl","wb")
pkl.dump(val_y, val_label)
val_label.close()

train_data = open("data/CongreG8/train_data.pkl","wb")
pkl.dump(train_x, train_data)
train_data.close()

val_data = open("data/CongreG8/val_data.pkl","wb")
pkl.dump(val_x, val_data)
val_data.close()
pass