# Linear Regression model

import sys
sys.path.append('../EEG-ECoG') # adding path for packages
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.dataloader import dataloader
from sklearn.decomposition import PCA
from utils import data_preprocessing as dp
import random

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)
    elif isinstance(m, SaverModule):
        nn.init.normal_(m.module.weight)

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLayerPerceptron, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

class MultiLayerPerceptronDict(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[]):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = []
        self.num_layers = len(hidden_sizes) + 1
        if len(hidden_sizes) > 0:
            self.layers.append(nn.Linear(input_size, output_size))
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            elif i == len(hidden_sizes):
                self.layers.append(nn.Linear(hidden_sizes[i], output_size))
            else:
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

    def forward(self, x):
        score = self.layers[0](x)
        for i in range(1,self.num_layers):
            score = self.layers[i](score)
        return score

class MultiLayerPerceptron2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MultiLayerPerceptron2, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

class SaverModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.saved_output = None

    def forward(self, x):
        output = self.module(x)
        self.saved_output = output
        return output


# # downsample eeg to ecog (gaussian kernel)
# if eeg_data.shape[1] != ecog_data.shape[1]:
#     # gaussian normalization
#     if eeg_data.shape[1] > ecog_data.shape[1]:
#         eeg_data = dp.downsample_data(eeg_data, ecog_data.shape[1])
#     elif eeg_data.shape[1] < ecog_data.shape[1]:
#         ecog_data = dp.downsample_data(ecog_data, eeg_data.shape[1])

# eeg_data = eeg_data.T     # (samples, channel)
# ecog_data = ecog_data.T   # (samples, channel)

# print("eeg_data.shape: ", eeg_data.shape)
# print("ecog_data.shape: ", ecog_data.shape)

# # Split data into training, validation and test
# random_section = random.randint(0,9)
# X_train = np.vstack((ecog_data[:ecog_data.shape[0]*random_section//10,:], ecog_data[ecog_data.shape[0]*(random_section+1)//10:,:]))
# X_test = ecog_data[ecog_data.shape[0]*random_section//10:ecog_data.shape[0]*(random_section+1)//10,:]
# y_train = np.vstack((eeg_data[:eeg_data.shape[0]*random_section//10,:], eeg_data[eeg_data.shape[0]*(random_section+1)//10:,:]))
# y_test = eeg_data[eeg_data.shape[0]*random_section//10:eeg_data.shape[0]*(random_section+1)//10,:]
# # print("X_train shape: ",X_train.shape)  # (287310, 129)
# # print("X_test shape: ", X_test.shape)   # (31924, 129)
# # print("y_train shape: ",y_train.shape)  # (287310, 19)
# # print("y_test shape: ", y_test.shape)   # (31924, 19)

# train
def train(dataloader, model, loss_fn, optimizer, device):
    epoch_loss = 0.0
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * pred.size(0)

        if batch % 400 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # if isinstance(model[0], SaverModule):
            #     print("first layer", model[0].saved_output)

    return epoch_loss

def test(X_tensor, y_tensor, model, loss_fn):
    # Make predictions on validation set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        test_loss = loss_fn(y_pred, y_tensor).item()
        accuracies = torch.empty(y_tensor.shape[1])
        # print("y_tensor.shape: ", y_tensor.shape)
        for i in range(y_tensor.shape[1]):
            corc = torch.corrcoef(torch.stack((y_pred[:,i], y_tensor[:,i])))
            accuracies[i] = (corc[0,1]+1)/2
        # print(accuracies)
        avg_corcs = torch.mean(accuracies)
        print(f"Accuracy: {(100*avg_corcs):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, accuracies
