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

# Import data
eeg_path = '../Datasets/20110607S2_EEGandECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG05_anesthesia.mat'
ecog_path = '../Datasets/20110607S2_EEGandECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG05_anesthesia.mat'
# eeg_path = '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/EEG_low-anesthetic.mat'
# ecog_path = '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/ECoG_low-anesthetic.mat'
_, eeg_data = dp.loadMatFile(eeg_path)
_, ecog_data = dp.loadMatFile(ecog_path)
# print(eeg_data.shape)  # (19, 323262)
# print(ecog_data.shape) # (129, 319234)

# Synthetic data
def generateSineWave(start_time, end_time, sample_rate, frequency, amplitude=1, offset=0):
    time = np.arange(start_time, end_time, 1/sample_rate)
    return amplitude * np.sin(2 * np.pi * frequency * time + offset)
st = 0     # start time
et = 60    # end time
sr = 1000  # sampling rate
num_ch_ecog = 129
synth_ecog = generateSineWave(st, et, sr, 10) + generateSineWave(st, et, sr, 5) + generateSineWave(st, et, sr, 3)
for i in range(num_ch_ecog-1):
    synth_ecog = np.vstack((synth_ecog, generateSineWave(st, et, sr, random.randint(1,44), random.randint(1,3)) + generateSineWave(st, et, sr, random.randint(1,44), random.randint(1,3)) + generateSineWave(st, et, sr, random.randint(1,44), random.randint(1,3))))

num_ch_eeg = 19
synth_eeg = generateSineWave(st, et, sr, 6) + generateSineWave(st, et, sr, 2)
for i in range(num_ch_eeg-1):
    synth_eeg = np.vstack((synth_eeg, generateSineWave(st, et, sr, random.randint(1,44), random.randint(1,3)) + generateSineWave(st, et, sr, random.randint(1,44), random.randint(1,3))))

print(synth_ecog.shape)
print(synth_eeg.shape)

# plt.figure(figsize=[50,20])
# for ch in range(10):
#     plt.subplot(10 + 1, 1, ch+1)
#     plt.plot(synth_ecog.T[:,ch])
#     plt.xlabel('samples')
#     plt.ylabel('potential(uV)')
#     plt.title(ch+1)

# plt.tight_layout()
# plt.show()

# plt.figure(figsize=[50,20])
# for ch in range(10):
#     plt.subplot(10 + 1, 1, ch+1)
#     plt.plot(synth_eeg.T[:,ch])
#     plt.xlabel('samples')
#     plt.ylabel('potential(uV)')
#     plt.title(ch+1)

# plt.tight_layout()
# plt.show()

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * pred.size(0)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            if isinstance(model[0], SaverModule):
                print("first layer", model[0].saved_output)

    return epoch_loss

def test(X_tensor, y_tensor, model, loss_fn):
    # Make predictions on validation set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        test_loss = loss_fn(y_pred, y_tensor).item()
        accuracies = []
        # print("y_tensor.shape: ", y_tensor.shape)
        for i in range(y_tensor.shape[1]):
            corc = torch.corrcoef(torch.stack((y_pred[:,i], y_tensor[:,i])))
            accuracies.append((corc[0,1]+1)/2)
        avg_corcs = np.average(accuracies)
        print(accuracies)
        print(f"Test Error: \n Accuracy: {(100*avg_corcs):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, accuracies

# -----------------------------------------------------------------------------
# Synthetic data test
# X = torch.tensor(synth_ecog, dtype=torch.float32).T
# y = torch.tensor(synth_eeg, dtype=torch.float32).T
# print(y.shape)
# print(X.shape)
# dataset = TensorDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# model = LinearRegressionModel(X.shape[1], y.shape[1]).to(device)
# loss_fn = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.99))

# loss_values = []
# num_epochs = 10
# for epoch in range(num_epochs):
#     epoch_loss = train(dataloader, model, loss_fn, optimizer)

#     # Calculate average loss for the epoch
#     epoch_loss /= len(dataloader.dataset)

#     # Append the loss values to the lists
#     loss_values.append(epoch_loss)

#     # Print progress
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# # Plot loss function and MSE
# plt.figure(figsize=(5, 5))
# plt.plot(range(1, num_epochs + 1), loss_values, label='loss')
# plt.legend()
# plt.yscale('log')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Synthetic ECoG to Synthetic EEG')

# plt.tight_layout()
# plt.show()
# -----------------------------------------------------------------------------
# aggregate ecog
# agg_ecog = np.zeros((4,319234))
# agg_ecog[0] = ecog_data[9] + ecog_data[10] + ecog_data[16] + ecog_data[17]
# agg_ecog[1] = ecog_data[41] + ecog_data[42] + ecog_data[52] + ecog_data[53]
# agg_ecog[2] = ecog_data[88] + ecog_data[89] + ecog_data[92] + ecog_data[93]
# agg_ecog[3] = ecog_data[108] + ecog_data[94] + ecog_data[109] + ecog_data[123]
# agg_ecog /= 4
# X = torch.tensor(ecog_data, dtype=torch.float32).T
# y = torch.tensor(agg_ecog, dtype=torch.float32).T
# print(y.shape)
# print(X.shape)
# dataset = TensorDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# hidden_size = 50
# # model = LinearRegressionModel(X.shape[1], y.shape[1]).to(device)
# model = MultiLayerPerceptron(X.shape[1], hidden_size, y.shape[1]).to(device)  # hidden_size=50, lr=5e-5, end_loss=0.0556
# loss_fn = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.99))

# loss_values = []
# num_epochs = 10
# for epoch in range(num_epochs):
#     epoch_loss = train(dataloader, model, loss_fn, optimizer)

#     # Calculate average loss for the epoch
#     epoch_loss /= len(dataloader.dataset)

#     # Append the loss values to the lists
#     loss_values.append(epoch_loss)

#     # Print progress
#     print(f'Epoch [{epoch+1}/{num_epochs}], train_Loss: {epoch_loss:.4f}')

# # Plot loss function and MSE
# plt.figure(figsize=(5, 5))
# plt.plot(range(1, num_epochs + 1), loss_values, label='Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('ECoG to Aggregate ECoG')

# plt.tight_layout()
# plt.show()

# -----------------------------------------------------------------------------
# Run with actual data
# downsample eeg to ecog
if eeg_data.shape[1] != ecog_data.shape[1]:
    # # gaussian normalization
    # if eeg_data.shape[1] > ecog_data.shape[1]:
    #     eeg_data = dp.downsample_data(eeg_data, ecog_data.shape[1])
    # elif eeg_data.shape[1] < ecog_data.shape[1]:
    #     ecog_data = dp.downsample_data(ecog_data, eeg_data.shape[1])

    # # truncate the end
    # if eeg_data.shape[1] > ecog_data.shape[1]:
    #     eeg_data = eeg_data[:, :ecog_data.shape[1]]
    # elif eeg_data.shape[1] < ecog_data.shape[1]:
    #     ecog_data = ecog_data[:, :eeg_data.shape[1]]

    # truncate the front
    if eeg_data.shape[1] > ecog_data.shape[1]:
        eeg_data = eeg_data[:, eeg_data.shape[1]-ecog_data.shape[1]:]
    elif eeg_data.shape[1] < ecog_data.shape[1]:
        ecog_data = ecog_data[:, ecog_data.shape[1]-eeg_data.shape[1]:]
print("------------------")
print(eeg_data.shape)
print(ecog_data.shape)
eeg_data = eeg_data.T
ecog_data = ecog_data.T
# Split data into training, validation and test
random_section = random.randint(0,9)
X_train = np.vstack((ecog_data[:ecog_data.shape[0]*random_section//10,:], ecog_data[ecog_data.shape[0]*(random_section+1)//10:,:]))
X_test = ecog_data[ecog_data.shape[0]*random_section//10:ecog_data.shape[0]*(random_section+1)//10,:]
y_train = np.vstack((eeg_data[:eeg_data.shape[0]*random_section//10,:], eeg_data[eeg_data.shape[0]*(random_section+1)//10:,:]))
y_test = eeg_data[eeg_data.shape[0]*random_section//10:eeg_data.shape[0]*(random_section+1)//10,:]
print(X_train.shape)
print(y_train.shape)
# simulating first iteration of 4-fold
X_train_new = np.vstack((X_train[:0*X_train.shape[0]//4,:],X_train[(0+1)*X_train.shape[0]//4:,:]))
X_val = X_train[0*X_train.shape[0]//4:(0+1)*X_train.shape[0]//4,:]
y_train_new = np.vstack((y_train[:0*y_train.shape[0]//4,:],y_train[(0+1)*y_train.shape[0]//4:,:]))
y_val = y_train[0*y_train.shape[0]//4:(0+1)*y_train.shape[0]//4,:]
print(X_train_new.shape)
print(X_val.shape)
print(y_train_new.shape)
print(y_val.shape)

low_bound = 0.5
high_bound = 45
sampling_rate = 1000
X_train_filtered = dp.butter_bandpass_filter(X_train_new.T, low_bound, high_bound, sampling_rate).T
X_val_filtered = dp.butter_bandpass_filter(X_val.T, low_bound, high_bound, sampling_rate).T
# PCA whitening
X_train_w = dp.whitening(X_train_filtered)
X_val_w = dp.whitening(X_val_filtered)

# Normalize X
scaler = StandardScaler()
X_train_new = scaler.fit_transform(X_train_w)
X_val = scaler.fit_transform(X_val_w)

X_val = torch.tensor(X_val.T, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X = torch.tensor(X_train_new.T, dtype=torch.float32)
y = torch.tensor(y_train_new, dtype=torch.float32)
print(X_val.shape)
print(y_val.shape)
print(X.shape)
print(y.shape)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
hidden_size1 = 80
hidden_size2 = 30
# model = LinearRegressionModel(X.shape[1], y.shape[1]).to(device)
# model = MultiLayerPerceptron(X.shape[1], hidden_size1, y.shape[1]).to(device)  #
# model = MultiLayerPerceptron2(X.shape[1], hidden_size1, hidden_size2, y.shape[1]).to(device)  #
model = nn.Sequential(
    SaverModule(nn.Linear(X.shape[1], hidden_size1)),
    nn.ReLU(),
    SaverModule(nn.Linear(hidden_size1, hidden_size2)),
    nn.ReLU(),
    SaverModule(nn.Linear(hidden_size2, y.shape[1])),
).to(device)

model.apply(init_weights)
loss_fn = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9,0.99))
optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9, nesterov=True)

train_losses = []
val_losses = []
accs = []
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = train(dataloader, model, loss_fn, optimizer)

    # Calculate average loss for the epoch
    epoch_loss /= len(dataloader.dataset)

    # Append the loss values to the lists
    train_losses.append(epoch_loss)

    val_loss, acc = test(X_val.to(device), y_val.to(device), model, loss_fn)

    val_losses.append(val_loss)
    accs.append(acc)

    # Print progress
    # if (epoch+1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], train_Loss: {epoch_loss:.4f}, val_loss: {val_loss:.4f}')

# Plot loss function and MSE
plt.figure(figsize=(5, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='train loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='val loss')
plt.legend()
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses over epoch')

plt.tight_layout()
plt.show()

accs = np.array(accs)
print(accs.shape)
plt.figure(figsize=(10, 5))
for i in range(y_val.shape[1]):
    plt.plot(range(1, num_epochs + 1), accs[:,i], label="ch" + str(i+1))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Accuracy per Channel')

plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

# # 4-fold cross-validation
# k = 4
# loss_dict = {}
# for i in range(k):
#     X_train_new = np.vstack((X_train[:i*X_train.shape[0]//k,:],X_train[(i+1)*X_train.shape[0]//k:,:]))
#     X_val = X_train[i*X_train.shape[0]//k:(i+1)*X_train.shape[0]//k,:]
#     # print("X_train_new shape: ",X_train_new.shape) # (215482, 129)
#     # print("X_val shape: ",X_val.shape)             # (71827, 129)
#     y_train_new = np.vstack((y_train[:i*y_train.shape[0]//k,:],y_train[(i+1)*y_train.shape[0]//k:,:]))
#     y_val = y_train[i*y_train.shape[0]//k:(i+1)*y_train.shape[0]//k,:]
#     # print("y_train_new shape: ",y_train_new.shape) # (215482, 19)
#     # print("y_val shape: ",y_val.shape)             # (71827, 19)

#     # possible hyperparameters
#     low_bound = 0.5
#     high_bound = 45
#     sampling_rate = 1000

#     print("filtering...")
#     # bandwidth_butterworth_filter
#     X_train_filtered = dp.butter_bandpass_filter(X_train_new.T, low_bound, high_bound, sampling_rate)
#     X_val_filtered = dp.butter_bandpass_filter(X_val.T, low_bound, high_bound, sampling_rate)

#     print("whitening...")
#     # PCA whitening
#     X_train_w = dp.whitening(X_train_filtered.T)
#     X_val_w = dp.whitening(X_val_filtered.T)

#     # Convert numpy arrays to PyTorch tensors
#     X_tensor = torch.tensor(X_train_w.T, dtype=torch.float32)
#     y_tensor = torch.tensor(y_train_new, dtype=torch.float32)
#     print("X_tensor shape", X_tensor.shape)
#     print("y_tensor shape", y_tensor.shape)

#     # Instantiate the model
#     input_size = X_tensor.shape[1]   # 129
#     output_size = y_tensor.shape[1]  # 19
#     model = LinearRegressionModel(input_size, output_size).to(device)

#     # Define loss function and optimizer
#     loss_fn = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.99))

#     # Create DataLoader for batch processing
#     dataset = TensorDataset(X_tensor, y_tensor)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#     # Keep track of loss and error
#     loss_values = []

#     # Training loop
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         epoch_loss = train(dataloader, model, loss_fn, optimizer)

#         # Calculate average loss for the epoch
#         epoch_loss /= len(dataloader.dataset)

#         # Append the loss values to the lists
#         loss_values.append(epoch_loss)

#         # Print progress
#         if (epoch+1) % 10 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

#     loss_dict[i] = loss_values

#     # # Plot loss function and MSE
#     # plt.figure(figsize=(10, 5))
#     # plt.subplot(1, 2, 1)
#     # plt.plot(range(1, num_epochs + 1), loss_values, label='Loss')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('Loss')
#     # plt.title('Training Loss')

#     # plt.tight_layout()
#     # plt.show()

#     # Evaluate on validation set

#     X_val_tensor = torch.tensor(X_val_w.T, dtype=torch.float32).to(device)
#     y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

#     test(X_val_tensor, y_val_tensor, model, loss_fn)

#     # # Calculate evaluation metrics
#     # mse = mean_squared_error(y_val_tensor.numpy(), y_pred.numpy())
#     # rmse = mean_squared_error(y_val_tensor.numpy(), y_pred.numpy(), squared=False)
#     # mae = mean_absolute_error(y_val_tensor.numpy(), y_pred.numpy())
#     # r2 = r2_score(y_val_tensor.numpy(), y_pred.numpy())

#     # print(f'Mean Squared Error (MSE): {mse:.4f}')
#     # print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
#     # print(f'Mean Absolute Error (MAE): {mae:.4f}')
#     # print(f'R-squared (R2): {r2:.4f}')