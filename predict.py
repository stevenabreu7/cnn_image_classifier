import torch 
import argparse
import numpy as np
from torch.autograd.variable import Variable

from hw2 import all_cnn
from hw2 import preprocessing

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('filename', help='Path to the model')
args = parser.parse_args()

# load the model
net = torch.load(args.filename)
net = net.cpu()

# load data
load = lambda x: np.load(x, encoding='bytes')
train_data = load('dataset/train_feats.npy')
test_data = load('dataset/test_feats.npy')

# preprocessing
train_data, test_data = preprocessing.cifar_10_preprocess(train_data, test_data)
test_data = torch.Tensor(test_data)

# check if gpu is available
gpu = torch.cuda.is_available()
net = net.cuda() if gpu else net
test_data = test_data.cuda() if gpu else test_data

# get predictions
predictions = []
for i in range(len(test_data)):
    if i % 100 == 0:
        print('\r', i, sep='', end='')
    output = net(test_data[i].unsqueeze(0))
    pred = torch.max(output, 1)[1]
    predictions.append(pred)

# save predictions
output_file='predictions.txt'
with open(output_file, 'w') as f:
    for y in predictions:
        f.write("{}\n".format(y))