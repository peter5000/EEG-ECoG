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

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Import data
eeg_path = '../Datasets/20110607S2_EEGandECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG05_anesthesia.mat'
ecog_path = '../Datasets/20110607S2_EEGandECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG05_anesthesia.mat'
_, eeg_data = dp.loadMatFile(eeg_path)
_, ecog_data = dp.loadMatFile(ecog_path)
# print(eeg_data.shape)   (19, 323262)
# print(ecog_data.shape)  (129, 319234)

# downsample eeg to ecog
eeg_data = dp.downsample_data(eeg_data, ecog_data.shape[1])

eeg_data = eeg_data.T     # (samples, channel)
ecog_data = ecog_data.T   # (samples, channel)

# Split data into training, validation and test
random_section = random.randint(0,9)
X_train = np.vstack((ecog_data[:ecog_data.shape[0]*random_section//10,:], ecog_data[ecog_data.shape[0]*(random_section+1)//10:,:]))
X_test = ecog_data[ecog_data.shape[0]*random_section//10:ecog_data.shape[0]*(random_section+1)//10,:]
y_train = np.vstack((eeg_data[:eeg_data.shape[0]*random_section//10,:], eeg_data[eeg_data.shape[0]*(random_section+1)//10:,:]))
y_test = eeg_data[eeg_data.shape[0]*random_section//10:eeg_data.shape[0]*(random_section+1)//10,:]
# print("X_train shape: ",X_train.shape)  # (287310, 129)
# print("X_test shape: ", X_test.shape)   # (31924, 129)
# print("y_train shape: ",y_train.shape)  # (287310, 19)
# print("y_test shape: ", y_test.shape)   # (31924, 19)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# train
def train(dataloader, model, loss_fn, optimizer):
    epoch_loss = 0.0
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item() * pred.size(0)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return epoch_loss

def test(X_tensor, y_tensor, model, loss_fn):
    # Make predictions on validation set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        test_loss = loss_fn(y_pred, y_tensor).item()
        corcs = []
        print("y_tensor.shape: ", y_tensor.shape)
        for i in range(y_tensor.shape[1]):
            corc = torch.corrcoef(torch.stack((y_pred[:,i], y_tensor[:,i])))
            corcs.append(corc)
        avg_corcs = np.average(corcs)
        print(f"Test Error: \n Accuracy: {(100*avg_corcs):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss

# 4-fold cross-validation
k = 4
loss_dict = {}
for i in range(k):
    X_train_new = np.vstack((X_train[:i*X_train.shape[0]//k,:],X_train[(i+1)*X_train.shape[0]//k:,:]))
    X_val = X_train[i*X_train.shape[0]//k:(i+1)*X_train.shape[0]//k,:]
    # print("X_train_new shape: ",X_train_new.shape) # (215482, 129)
    # print("X_val shape: ",X_val.shape)             # (71827, 129)
    y_train_new = np.vstack((y_train[:i*y_train.shape[0]//k,:],y_train[(i+1)*y_train.shape[0]//k:,:]))
    y_val = y_train[i*y_train.shape[0]//k:(i+1)*y_train.shape[0]//k,:]
    # print("y_train_new shape: ",y_train_new.shape) # (215482, 19)
    # print("y_val shape: ",y_val.shape)             # (71827, 19)

    # possible hyperparameters
    low_bound = 0.5
    high_bound = 45
    sampling_rate = 1000

    print("filtering...")
    # bandwidth_butterworth_filter
    X_train_filtered = dp.butter_bandpass_filter(X_train_new.T, low_bound, high_bound, sampling_rate)
    X_val_filtered = dp.butter_bandpass_filter(X_val.T, low_bound, high_bound, sampling_rate)

    print("whitening...")
    # PCA whitening
    X_train_w = dp.whitening(X_train_filtered.T)
    X_val_w = dp.whitening(X_val_filtered.T)

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X_train_w.T, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    print("X_tensor shape", X_tensor.shape)
    print("y_tensor shape", y_tensor.shape)

    # Instantiate the model
    input_size = X_tensor.shape[1]   # 129
    output_size = y_tensor.shape[1]  # 19
    model = LinearRegressionModel(input_size, output_size).to(device)

    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.99))

    # Create DataLoader for batch processing
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Keep track of loss and error
    loss_values = []

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = train(dataloader, model, loss_fn, optimizer)

        # Calculate average loss for the epoch
        epoch_loss /= len(dataloader.dataset)

        # Append the loss values to the lists
        loss_values.append(epoch_loss)

        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    loss_dict[i] = loss_values

    # # Plot loss function and MSE
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, num_epochs + 1), loss_values, label='Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')

    # plt.tight_layout()
    # plt.show()

    # Evaluate on validation set

    X_val_tensor = torch.tensor(X_val.T, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    test(X_val_tensor, y_val_tensor, model, loss_fn)

    # # Calculate evaluation metrics
    # mse = mean_squared_error(y_val_tensor.numpy(), y_pred.numpy())
    # rmse = mean_squared_error(y_val_tensor.numpy(), y_pred.numpy(), squared=False)
    # mae = mean_absolute_error(y_val_tensor.numpy(), y_pred.numpy())
    # r2 = r2_score(y_val_tensor.numpy(), y_pred.numpy())

    # print(f'Mean Squared Error (MSE): {mse:.4f}')
    # print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    # print(f'Mean Absolute Error (MAE): {mae:.4f}')
    # print(f'R-squared (R2): {r2:.4f}')