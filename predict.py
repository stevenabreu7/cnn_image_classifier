import torch 
import argparse
import numpy as np
from torch.autograd.variable import Variable

from hw2 import all_cnn
from hw2 import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument(
    'filename',
    help='Path to the model')
args = parser.parse_args()

net = torch.load(args.filename)
net = net.cpu()

load = lambda x: np.load(x, encoding='bytes')
train_data = load('dataset/train_feats.npy')
test_data = load('dataset/test_feats.npy')
# preprocessing
train_data, test_data = preprocessing.cifar_10_preprocess(train_data, test_data)

gpu = False # torch.cuda.is_available()
net = net.cuda() if gpu else net

train_data = torch.Tensor(train_data)
train_data = train_data.cuda() if gpu else train_data
train_data = Variable(train_data)

predictions = []

for i in range(len(train_data)):
    output = net(train_data[i].unsqueeze(0))
    pred = torch.max(output, 1)[1]
    predictions.append(pred)

# output = net(train_data)
# predictions = torch.max(output, 1)[1]

output_file='predictions.txt'
with open(output_file, 'w') as f:
    for y in predictions:
        f.write("{}\n".format(y))