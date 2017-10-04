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

# load the previously generated pickle.  I didn't want to bother with a lot of
# loading logic in this file, since the purpose is to simply defined an
# architecture with pytorch and see it in action.
with open("../mnist/mnist_train.pickle", 'rb') as mnist_pickle:
    pickled_data = pickle.load(mnist_pickle)
    y_vals, images = pickled_data[0], pickled_data[1]

x = Variable(torch.from_numpy(images))
y = Variable(torch.from_numpy(y_vals).type(torch.LongTensor), requires_grad=False)

# TODO: How do we do weight decay?
#weight_decay = 0

batch_size    = 100
epochs        = 100
learning_rate = 1e-3

number_of_batches = int(np.round(x.data.shape[0]/batch_size))

# TODO: How do we specify an initializer?  What is used by default?
# TODO: try to understand what the last layer should be (maybe sigmoid since
# these are either 0 or 1, followed by softmax)
model = torch.nn.Sequential(
    torch.nn.Linear(x.data.shape[1], 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, y.data.shape[1]),
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):

    # TODO: Add some smarts to this
    # progressively reduce the learning rate to account for smaller changes
    if epoch % 10 == 0:
        learning_rate *= 1e-1

    # TODO: Functionalize batching processing
    start_of_batch = 0

    for batch in range(number_of_batches):
        end_of_batch = min(start_of_batch + batch_size, x.data.shape[0])
        x_batch, y_batch = x[start_of_batch:end_of_batch,:], y[start_of_batch:end_of_batch,:]
        start_of_batch = start_of_batch + batch_size

        y_batch_pred = model(x_batch)

        _, y_batch = torch.max(y_batch, 1)

        loss = loss_fn(y_batch_pred, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print("epoch: %d, loss: %f" % (epoch + 1, loss.data[0]))


# TODO: get test accuracy as well
# TODO: plot the two side by side each epoch
# Print accuracies
y_pred = model(x)

_, y_pred = torch.max(y_pred, 1)
_, y = torch.max(y, 1)

total = 0
total_right = 0
for guess, actual in zip(y_pred, y):

    guess = guess.data[0]
    actual = actual.data[0]

    total += 1
    if guess == actual:
        total_right += 1

accuracy = total_right/total

print("Training Accuracy: %f" % accuracy)

