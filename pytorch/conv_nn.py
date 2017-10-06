'''

basic example of a convolutional neural network.  We will use this to do
calssification on the CIFAR-10 dataset:

        http://www.cs.utoronto.ca/~kriz/cifar-10-python.tar.gz

'''

import torch
from torch import nn
from torch.autograd import Variable
import pickle
import numpy as np
import matplotlib.pyplot as plt


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        # define types of layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(50)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(576, 50)
        self.fc2 = nn.Linear(50, 10)

        self.log_softmax = nn.LogSoftmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.batch_norm2(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.batch_norm3(x)

        # reshape so we can use a fully connected layer
        x = x.view(-1, 576)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch_norm4(x)

        x = self.fc2(x)
        x = self.log_softmax(x)

        return x


# a generator to return batch beginning and ending points
def batch_iter(batch_size, set_size):
    set_size = set_size
    batch_size = batch_size
    number_of_batches = int(np.round(set_size/batch_size))

    start_of_batch = 0

    end_of_batch = 0

    while end_of_batch < set_size:
        end_of_batch = min(start_of_batch + batch_size, set_size)

        yield start_of_batch, end_of_batch

        start_of_batch = start_of_batch + batch_size


def load_data():
    # Training data

    # import images/labels
    with open("/home/jared/cifar-10-batches-py/data_batch_1", 'rb') as batch_1:
        train_dict = pickle.load(batch_1, encoding='bytes')

    train_x = train_dict[b'data']
    train_y = train_dict[b'labels']

    with open("/home/jared/cifar-10-batches-py/data_batch_2", 'rb') as batch_2:
        train_dict = pickle.load(batch_2, encoding='bytes')

    train_x = np.concatenate((train_x, train_dict[b'data']), axis=0)
    train_y += train_dict[b'labels']

    with open("/home/jared/cifar-10-batches-py/data_batch_3", 'rb') as batch_3:
        train_dict = pickle.load(batch_3, encoding='bytes')

    train_x = np.concatenate((train_x, train_dict[b'data']), axis=0)
    train_y += train_dict[b'labels']

    with open("/home/jared/cifar-10-batches-py/data_batch_4", 'rb') as batch_4:
        train_dict = pickle.load(batch_4, encoding='bytes')

    train_x = np.concatenate((train_x, train_dict[b'data']), axis=0)
    train_y += train_dict[b'labels']

    train_x = torch.from_numpy(train_x).view([40000, 32, 32, 3]).type(torch.FloatTensor).permute(0,3,1,2)
    train_y = torch.Tensor(train_y).type(torch.LongTensor)

    train_x = Variable(train_x)
    train_y = Variable(train_y, requires_grad=False)


    # Cross vaildation data
    with open("/home/jared/cifar-10-batches-py/data_batch_5", 'rb') as fo:
        cross_dict = pickle.load(fo, encoding='bytes')

    cross_x = cross_dict[b'data']
    cross_y = cross_dict[b'labels']

    cross_x = torch.from_numpy(cross_x).view([10000, 32, 32, 3]).type(torch.FloatTensor).permute(0,3,1,2)
    cross_y = torch.Tensor(cross_y).type(torch.LongTensor)

    cross_x = Variable(cross_x)
    cross_y = Variable(cross_y, requires_grad=False)

    return train_x, train_y, cross_x, cross_y


def main():
    # setup model
    epochs        = 100
    learning_rate = 1e-3
    weight_decay  = 1e-1
    batch_size    = 300
    model         = BasicCNN()
    loss_fn       = torch.nn.CrossEntropyLoss()
    optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    plot          = False

    # get data
    train_x, train_y, cross_x, cross_y = load_data()

    # Initialize plot stuff
    if plot:
        epoch_list          = []
        loss_list           = []
        train_accuracy_list = []
        cross_accuracy_list = []

        loss_plot = plt.figure()
        accuracy_plot = plt.figure()

        # axis
        loss_ax = loss_plot.gca()
        accuracy_ax = accuracy_plot.gca()

        loss_plot.show()
        accuracy_plot.show()

    # train
    for epoch in range(epochs):
        total_loss = 0
        train_total = 0
        train_total_right = 0

        batches = batch_iter(batch_size, train_x.data.shape[0])
        for start_of_batch, end_of_batch in batches:
            x_batch, y_batch = train_x[start_of_batch:end_of_batch], train_y[start_of_batch:end_of_batch]
            y_pred = model(x_batch)

            # calculate training accuracy
            _, y_pred_max = torch.max(y_pred, 1)

            for guess, actual in zip(y_pred_max, y_batch):
                guess = guess.data[0]
                actual = actual.data[0]

                train_total += 1
                if guess == actual:
                    train_total_right += 1

            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test on cross validation set
        cross_total = 0
        cross_total_right = 0
        batches = batch_iter(batch_size, cross_x.data.shape[0])
        for start_of_batch, end_of_batch in batches:

            x_batch, y_batch = cross_x[start_of_batch:end_of_batch], cross_y[start_of_batch:end_of_batch]

            y_pred = model(x_batch)

            # calculate training accuracy
            _, y_pred_max = torch.max(y_pred, 1)

            for guess, actual in zip(y_pred_max, y_batch):
                guess = guess.data[0]
                actual = actual.data[0]

                cross_total += 1
                if guess == actual:
                    cross_total_right += 1


        # print helpful info at the end of the epoch.  train accuracy belongs to
        # the previous iteration.  just didn't want to have to calculate it
        # twice
        train_accuracy = train_total_right/train_total
        print("train accuracy:             %.2f%%" % (train_accuracy * 100))

        cross_accuracy = cross_total_right/cross_total
        print("cross validation accuracy:  %.2f%%" % (cross_accuracy * 100))

        print("\nepoch:                      %d" % (epoch + 1))
        print("loss:                       %f" % total_loss)

        # plot loss graph and accuracy graphs
        if plot:
            epoch_list.append(epoch)
            loss_list.append(total_loss)
            train_accuracy_list.append(train_accuracy)
            cross_accuracy_list.append(cross_accuracy)

            loss_ax.plot(epoch_list, loss_list, 'r')
            loss_plot.canvas.draw()

            # TODO: figure out how to add legends
            accuracy_ax.plot(epoch_list, train_accuracy_list, 'g', epoch_list, cross_accuracy_list, 'b')
            accuracy_plot.canvas.draw()


    # test, but only after training is complete...
    # TODO: create a function for this

if __name__ == "__main__":
    main()
