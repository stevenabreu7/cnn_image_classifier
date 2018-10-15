import torch 
import numpy as np 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.autograd.variable import Variable

from hw2 import preprocessing
from hw2 import all_cnn

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

def load_data():
    load = lambda x: np.load(x, encoding='bytes')
    train_data = load('dataset/train_feats.npy')
    train_labels = load('dataset/train_labels.npy')
    test_data = load('dataset/test_feats.npy')    
    # preprocessing
    print('Preprocessing data')
    train_data, test_data = preprocessing.cifar_10_preprocess(train_data, test_data)
    # training dataset
    train_dataset = CustomDataset(train_data, train_labels)
    return train_dataset, test_data

class Trainer:
    def __init__(self, train_loader, name, net, optimizer, criterion, scheduler):
        print('Loading Trainer class for {}. '.format(name))
        # save the loaders
        self.update_data(train_loader)
        # update training model
        self.update_model(name, net, optimizer, criterion, scheduler)
        # check GPU availability
        self.gpu = torch.cuda.is_available()
        print('Using GPU' if self.gpu else 'Not using GPU')
    
    def save_model(self):
        torch.save(self.net.state_dict(), 'models/{}'.format(self.name))
    
    def update_data(self, train_loader):
        self.train_loader = train_loader
    
    def update_model(self, name, net, optimizer, criterion, scheduler):
        self.net = net
        self.name = name 
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
    
    def train(self, epochs):
        print('Start training {}.'.format(self.name))

        # move network to GPU if possible
        self.net = self.net.cuda() if self.gpu else self.net 

        for epoch in range(epochs):

            if self.scheduler:
                self.scheduler.step()

            train_num = 0
            train_loss = 0
            train_correct = 0

            for batch_i, (batch_data, batch_labels) in enumerate(self.train_loader):
                
                # reset optimizer gradients to zero
                self.optimizer.zero_grad()

                # initialize the data as variables
                batch_data = Variable(batch_data)
                batch_labels = Variable(batch_labels.type(torch.LongTensor))

                # move data to GPU if possible
                batch_data = batch_data.cuda() if self.gpu else batch_data
                batch_labels = batch_labels.cuda() if self.gpu else batch_labels

                # forward pass of the data
                batch_output = self.net(batch_data)

                # evaluate the prediction and correctness
                batch_prediction = batch_output.data.max(1, keepdim = True)[1]
                batch_prediction = batch_prediction.eq(batch_labels.data.view_as(batch_prediction))
                train_correct += batch_prediction.sum()
                train_num += batch_data.data.shape[0]

                # compute the losss
                batch_loss = self.criterion(batch_output, batch_labels)
                
                # backward pass and optimizer step
                batch_loss.backward()
                self.optimizer.step()

                # sum up this batch's loss
                train_loss += batch_loss.data.item()

                # print training progress
                if batch_i % 10 == 0:
                    print('\rEpoch {:3} Progress {:7.2%} Accuracy {:7.2%}'.format(
                        epoch + 1, 
                        batch_i * self.train_loader.batch_size / len(self.train_loader.dataset),
                        train_correct.cpu().item() / ((batch_i + 1) * self.train_loader.batch_size)
                    ), end='')

            # compute epoch loss and accuracy
            train_loss = train_loss / train_num
            train_accuracy = train_correct.cpu().item() / train_num

            # print summary for this epoch
            print('\rEpoch {:3} finished.\t\t\t\nTraining Accuracy: {:5.2%}\nTraining Loss: {:10.7f}'.format(
                epoch + 1, 
                train_accuracy, 
                train_loss
            ))

            torch.save(self.net, 'models/{}_{}'.format(self.name, epoch))

        # move network back to CPU if needed
        self.net = self.net.cpu() if self.gpu else self.net 

def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0,1)

def init_xavier(m):
    if type(m) == nn.Conv2d:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)

def write_results(predictions, output_file='predictions.txt'):
    """
    Write predictions to file.
    File should be:
        named 'predictions.txt'
        in the root of the tar file
    :param predictions: iterable of integers
    :param output_file:  path to output file.
    :return: None
    """
    with open(output_file, 'w') as f:
        for y in predictions:
            f.write("{}\n".format(y))

def main():
    # data parameters
    batch_size = 32

    # datasets and loaders
    print('Loading datasets')
    train_dataset, test_data = load_data()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))

    # model
    net = all_cnn.all_cnn_module()

    # apply initialization
    net.apply(init_xavier)

    # training parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=1e-3)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 250, 300], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30], gamma=0.1)
    criterion = nn.modules.loss.CrossEntropyLoss()

    # initialize the trainer
    trainer = Trainer(train_loader, 'first_try', net, optimizer, criterion, scheduler)

    # run the training
    epochs = 350
    trainer.train(epochs)

if __name__ == '__main__':
    main()
