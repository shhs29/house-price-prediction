import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda
from torch import nn, rand, float64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

house_data = pd.read_csv("train.csv").drop("Id", axis=1)
test_data = pd.read_csv("test.csv").drop("Id", axis=1)
cols_with_missing = [col for col in house_data.columns if house_data[col].isnull().any()]
house_data = house_data.drop(cols_with_missing, axis=1)

string_data = house_data.select_dtypes(include=['object'])
le = preprocessing.LabelEncoder()
for column in string_data:
    house_data[column] = le.fit_transform(house_data[column])

cols_with_missing = [col for col in house_data.columns if house_data[col].isna().any()]
house_data = house_data.drop(cols_with_missing, axis=1)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self.flatten = nn.Flatten()
        self.net = nn.Sequential(nn.Linear(60, 30), nn.Sigmoid(), nn.Linear(30, 1))

    def forward(self, x):
        # x = self.flatten(x)
        return self.net(x)


eta = 0.5
epochs = 15

model = Network()

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), eta)

optimizer.zero_grad()


def train_data(training_dataset, salePrice, model, loss_function, optimizer):
    loss = 0
    for x, y in zip(training_dataset, salePrice):
        x = x
        y = y
        output = model(x)
        loss += loss_function(output, torch.Tensor([y]))
        print(output)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss_epoch:{loss/len(salePrice)}")



def test_data(testing_set, model):
    size = len(testing_set)
    correct = 0
    for X, y in testing_set:
        X = X
        y = y
        output = model(X)
        if output.argmax() == y.argmax():
            correct += 1
    print(f'Accuracy: {correct}/{size}')


train_values = torch.from_numpy(house_data.loc[:, house_data.columns != 'SalePrice'].values.astype(np.float32))
train_values = train_values.float()

for epoch in range(epochs):
    print(f"Epoch {epoch}")
    train_data(train_values, house_data['SalePrice'], model, loss_function, optimizer)
    # test_data(house_data.loc[house_data['LotShape'] == 3], model)
