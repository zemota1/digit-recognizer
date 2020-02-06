import torch
import pandas as pd
from matplotlib import pyplot as plt


def get_data(train_path, test_path):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    x_train = train.values[:, 1:] / 255
    y_train = train.values[:, 0]
    x_test = test.values[:, 1:] / 255
    y_test = test.values[:, 0]

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    xx_train, yy_train, xx_test, yy_test = get_data('../data/train.csv', '../data/test.csv')
    
