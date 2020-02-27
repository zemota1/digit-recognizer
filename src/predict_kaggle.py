import pandas as pd

import torch
from torch.autograd import Variable

from src.cnn_net import CNNModel

if __name__ == '__main__':

    model = CNNModel()
    model.load_state_dict(torch.load("model/mnist_cnn"))
    model.eval()

    test_data = pd.read_csv("../data/test.csv")
    x_test = test_data.values/255
    x_test = torch.from_numpy(x_test)

    test = Variable(x_test.view(28000, 1, 28, 28))
    outputs = model(test.float())

    predicted = torch.max(outputs.data, 1)[1].numpy()
    print(predicted)

    submission = pd.read_csv("../data/sample_submission.csv")
    submission['Label'] = predicted

    submission.to_csv("../data/submission.csv", index=False)