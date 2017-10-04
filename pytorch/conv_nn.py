'''

basic example of a convolutional neural network.  We will use this to do
calssification on the CIFAR-10 dataset:

        http://www.cs.utoronto.ca/~kriz/cifar-10-python.tar.gz

'''

import torch
from torch.autograd import Variable
import pickle
import numpy as np


def get_training_data():
    with open("/home/jared/cifar-10-batches-py/data_batch_1", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def main():

    # import images/labels
    dict = get_training_data()

    y = dict[b'labels']
    X = dict[b'data']

    X = torch.from_numpy(X).view([10000, 32, 32, 3]).type(torch.FloatTensor).permute(0,3,1,2)

    # batch this!
    X = X[0:1000]

    X = Variable(X)

    # TODO: use a sane architecture
    # define model
    model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3),
            torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
    )

    A = Variable(torch.randn(100, 3, 32, 32))

    model(X)


    # train


    # cross validate


    # test, but only after training is complete...






if __name__ == "__main__":
    main()
