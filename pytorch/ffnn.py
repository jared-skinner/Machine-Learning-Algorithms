'''
basic foward feed neural network using the nn module from pytorch
'''
import torch
from torch.autograd import Variable
import pickle
import numpy as np

dtype = torch.LongTensor

images = []
y_vals = []

# load the pickle
with open("../mnist/mnist_train.pickle", 'rb') as mnist_pickle:
    pickled_data = pickle.load(mnist_pickle)
    y_vals, images = pickled_data[0], pickled_data[1]

x = Variable(torch.from_numpy(images))
y = Variable(torch.from_numpy(y_vals).type(torch.LongTensor), requires_grad=False)

#weight_decay = 0

batch_size    = 100
epochs        = 10
learning_rate = 1e-4

number_of_batches = int(np.round(x.data.shape[0]/batch_size))

model = torch.nn.Sequential(
    torch.nn.Linear(x.data.shape[1], 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, y.data.shape[1]),
    torch.nn.Softmax()
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    start_of_batch = 0

    for batch in range(number_of_batches):
        end_of_batch = min(start_of_batch + batch_size, x.data.shape[0])
        x_batch, y_batch = x[start_of_batch:end_of_batch,:], y[start_of_batch:end_of_batch,:]
        start_of_batch = start_of_batch + batch_size

        y_pred = model(x_batch)

        _, y_batch = torch.max(y_batch, 1)

        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print("epoch: %d, loss: %f" % (epoch + 1, loss.data[0]))


