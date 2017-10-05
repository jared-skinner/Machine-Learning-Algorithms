'''

basic example of a convolutional neural network.  We will use this to do
calssification on the CIFAR-10 dataset:

        http://www.cs.utoronto.ca/~kriz/cifar-10-python.tar.gz

'''

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import pickle
import numpy as np



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # define types of layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=1)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(1152, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # TODO: use a sane architecture
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)

        # reshape so we can use a fully connected layer
        x = x.view(-1, 1152)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = F.log_softmax(x)

        return x


def get_training_data():
    with open("/home/jared/cifar-10-batches-py/data_batch_1", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def main():

    # import images/labels
    dict = get_training_data()

    epochs = 200
    learning_rate = 1e-3

    batch_size = 500


    y = dict[b'labels']
    x = dict[b'data']

    number_of_batches = int(np.round(x.data.shape[0]/batch_size))

    y = torch.Tensor(y).type(torch.LongTensor)
    x = torch.from_numpy(x).view([10000, 32, 32, 3]).type(torch.FloatTensor).permute(0,3,1,2)

    # permute changes the size to 10000 X 3 X 32 X 32

    x = Variable(x)
    y = Variable(y, requires_grad=False)


    model = Net()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    for epoch in range(epochs):
        start_of_batch = 0
        total_loss = 0
        for batch in range(number_of_batches):
            end_of_batch = min(start_of_batch + batch_size, x.data.shape[0])
            x_batch, y_batch = x[start_of_batch:end_of_batch], y[start_of_batch:end_of_batch]
            start_of_batch = start_of_batch + batch_size

            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)

            total_loss += loss.data[0]

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


        print("\epoch:           %d" % epoch)
        print("\nloss:           %f" % total_loss)

        total = 0
        total_right = 0
        start_of_batch = 0
        for batch in range(number_of_batches):
            end_of_batch = min(start_of_batch + batch_size, x.data.shape[0])
            x_batch, y_batch = x[start_of_batch:end_of_batch], y[start_of_batch:end_of_batch]
            start_of_batch = start_of_batch + batch_size

            _, y_pred = torch.max(model(x_batch), 1)

            for guess, actual in zip(y_pred, y_batch):
                guess = guess.data[0]
                actual = actual.data[0]

                total += 1
                if guess == actual:
                    total_right += 1

        accuracy = total_right/total

        print("train accuracy: %f" % accuracy)

        # test, but only after training is complete...

    # cross validate

if __name__ == "__main__":
    main()
