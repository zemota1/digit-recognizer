import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from cnn_net import CNNModel


def get_data(train_path, test_path):

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    x_train = train_data.values[:, 1:] / 255
    y_train = train_data.values[:, 0]
    x_test = test_data.values / 255

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)

    return x_train, y_train, x_test


def train_step(train_set, test_set, count):
    model.train()
    for i, (images, labels) in enumerate(train_set):

        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(train.float())

        loss = error(outputs, labels)
        loss.backward()

        optimizer.step()
        count += 1
        if count % 50 == 0:

            accuracy = test_step(test_set)
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.item(), accuracy))
    return count


def test_step(test_data):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():

        for images, labels in test_data:
            test = Variable(images.view(100, 1, 28, 28))

            # Forward propagation
            outputs = model(test.float())

            predicted = torch.max(outputs.data, 1)[1]

            total += len(labels)

            correct += (predicted == labels).sum()

    return 100 * correct / float(total)


if __name__ == '__main__':

    features, targets, to_predict = get_data('../data/train.csv', '../data/test.csv')

    xx_train, xx_test, yy_train, yy_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42
    )

    train = torch.utils.data.TensorDataset(xx_train, yy_train)
    test = torch.utils.data.TensorDataset(xx_test, yy_test)

    batch_size = 100
    n_iters = 3000
    num_epochs = n_iters/(len(train)/batch_size)
    num_epochs = int(num_epochs)

    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    model = CNNModel()

    error = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

    count = 0
    for epoch in range(num_epochs):
        count = train_step(train_loader, test_loader, count)
    torch.save(model.state_dict(), "model/mnist_cnn")
