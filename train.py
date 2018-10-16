import torch 
import argparse
import numpy as np 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.autograd.variable import Variable

from utils import preprocessing
from utils import all_cnn

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        # raw data
        self.data = data
        self.labels = labels 
        # convert to tensors
        self.data = torch.Tensor(self.data).float()
        self.labels = torch.Tensor(self.labels)
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'name',
        help='Name of the model.')
    parser.add_argument(
        '--preprocessing', 
        '-p',
        default=False, 
        action='store_true', 
        help='Turn on and off preprocessing.')
    parser.add_argument(
        '--scheduler', 
        '-s',
        default=False, 
        action='store_true', 
        help='Turn on and off scheduler.')
    parser.add_argument(
        '--batchsize', 
        '-b',
        type=int,
        default=16,
        help='Batch Size.')
    parser.add_argument(
        '--epochs', 
        '-e',
        type=int,
        default=100,
        help='Epochs')
    parser.add_argument(
        '--lr',
        '-l',
        type=float,
        default=0.01,
        help='Learning rate')
    parser.add_argument(
        '--wdecay',
        '-w',
        type=float,
        default=0.001,
        help='Weight decay')
    parser.add_argument(
        '--init',
        '-i',
        type=int,
        default=0,
        help='0: no initialization. 1: xavier. 2: kaimin')
    parser.add_argument(
        '--verbose',
        '-v',
        default=False,
        action='store_true',
        help='Increase verbosity - output stats for every batch.')

    return parser.parse_args()

class ModelParams:
    def __init__(self, args):
        self.name = args.name 
        self.preprocessing = args.preprocessing
        self.scheduler = args.scheduler
        self.batch_size = args.batchsize
        self.epochs = args.epochs
        self.learning_rate = args.lr
        self.weight_decay = args.wdecay
        self.init_fn = args.init
        self.verbose = args.verbose 

# parse arguments
args = parse_arguments()

# loading the data
load = lambda x: np.load(x, encoding='bytes')
train_data = load('dataset/train_feats.npy')
train_labels = load('dataset/train_labels.npy')
test_data = load('dataset/test_feats.npy')

# preprocessing
if args.preprocessing:
    print('Preprocessing data')
    train_data, test_data = preprocessing.cifar_10_preprocess(train_data, test_data)
else:
    print('No preprocessing')
    train_data = train_data.reshape((-1, 3, 32, 32))
    test_data = test_data.reshape((-1, 3, 32, 32))

# data loader
train_dataset = CustomDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

# network
net = all_cnn.all_cnn_module()

# weight initialization
for layer in net:
    if isinstance(layer, nn.Conv2d):
        if args.init == 1:
            nn.init.xavier_normal_(layer.weight)
        elif args.init == 2:
            nn.init.kaiming_normal_(layer.weight)
        layer.bias.data.fill_(0)

# logging
print('Training', args.name)
print('Using learning rate', args.lr)
print('Using weight decay', args.wdecay)
print('Using batch size', args.batchsize)
print('Using total # of epochs', args.epochs)
init_fn_names = {0: 'no function', 1: 'xavier', 2: 'kaiming'}
print('Using', init_fn_names[args.init], 'for weight initialization')

# optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
# criterion
criterion = nn.CrossEntropyLoss()

# move to GPU if possible
gpu = torch.cuda.is_available()
net = net.cuda() if gpu else net

# training 
print('Batch/Epoch\t\tAccuracy\tLoss')
for epoch in range(args.epochs):
    
    train_n = 0
    train_loss = 0.
    train_correct = 0

    net.train()

    for batch_i, (batch_data, batch_labels) in enumerate(train_loader):

        optimizer.zero_grad()

        batch_data = Variable(batch_data)
        batch_labels = Variable(batch_labels.type(torch.LongTensor))

        if gpu:
            batch_data = batch_data.cuda()
            batch_labels = batch_labels.cuda()
        
        # forward pass
        batch_output = net(batch_data)
        batch_loss = criterion(batch_output, batch_labels)
        # backward pass
        batch_loss.backward()
        optimizer.step()

        # make prediction
        batch_prediction = torch.max(batch_output, 1)[1]
        count = (batch_prediction == batch_labels).sum()

        # update metrics
        train_correct += count
        train_n += batch_data.data.shape[0]
        train_loss += batch_loss.data.item()

        if batch_i % 10 == 9 or batch_i == 0:
            print('\rBatch {:04}/{:04}\t\t{:6.4f}\t\t{:6.4f}'.format(
                batch_i+1,
                len(train_loader),
                train_correct.cpu().item() / train_n,
                train_loss / (batch_i + 1)
            ), end='')
    
    # compute epoch metrics
    train_loss = train_loss / len(train_loader)
    train_acc = train_correct.cpu().item() / train_n

    print('\rEpoch {:04}/{:04}\t\t{:6.4f}\t\t{:6.4f}'.format(
        epoch + 1,
        args.epochs,
        train_acc, 
        train_loss
    ))

    # save model at each epoch
    torch.save(net, 'models/{}_{}'.format(
        args.name, 
        epoch+1
    ))