# Evaluating Linear regression model with given function

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
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import data_preprocessing as dp
from models import linear_regresion_model2 as lr2
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_path', type=str, default="../Datasets/20110607S2_EEGandECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG05_anesthesia.mat")
    parser.add_argument('--ecog_path', type=str, default="../Datasets/20110607S2_EEGandECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG05_anesthesia.mat")
    parser.add_argument('--optim', type=str, default="sgd")  # [sgd, adam]
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--output_root', type=str, default='output')
    args = parser.parse_args()

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Import data
    print("Loading data...")
    eeg_path = args.eeg_path
    ecog_path = args.ecog_path
    _, eeg_data = dp.loadMatFile(eeg_path)
    _, ecog_data = dp.loadMatFile(ecog_path)

    # downsample eeg to ecog
    if eeg_data.shape[1] != ecog_data.shape[1]:
        print("Downsampling data...")
        # truncate the front
        if eeg_data.shape[1] > ecog_data.shape[1]:
            eeg_data = eeg_data[:, eeg_data.shape[1]-ecog_data.shape[1]:]
        elif eeg_data.shape[1] < ecog_data.shape[1]:
            ecog_data = ecog_data[:, ecog_data.shape[1]-eeg_data.shape[1]:]
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

    eeg_data = eeg_data.T
    ecog_data = ecog_data.T
    # Split data into training, validation and test
    print("Spliting data...")
    random_section = random.randint(0,9)
    X_train = np.vstack((ecog_data[:ecog_data.shape[0]*random_section//10,:], ecog_data[ecog_data.shape[0]*(random_section+1)//10:,:]))
    X_test = ecog_data[ecog_data.shape[0]*random_section//10:ecog_data.shape[0]*(random_section+1)//10,:]
    y_train = np.vstack((eeg_data[:eeg_data.shape[0]*random_section//10,:], eeg_data[eeg_data.shape[0]*(random_section+1)//10:,:]))
    y_test = eeg_data[eeg_data.shape[0]*random_section//10:eeg_data.shape[0]*(random_section+1)//10,:]
    # print(X_train.shape)
    # print(y_train.shape)

    # simulating first iteration of 4-fold (without shuffle)
    X_train_new = np.vstack((X_train[:0*X_train.shape[0]//4,:],X_train[(0+1)*X_train.shape[0]//4:,:]))
    X_val = X_train[0*X_train.shape[0]//4:(0+1)*X_train.shape[0]//4,:]
    y_train_new = np.vstack((y_train[:0*y_train.shape[0]//4,:],y_train[(0+1)*y_train.shape[0]//4:,:]))
    y_val = y_train[0*y_train.shape[0]//4:(0+1)*y_train.shape[0]//4,:]
    # print(X_train_new.shape)
    # print(X_val.shape)
    # print(y_train_new.shape)
    # print(y_val.shape)

    low_bound = 0.5              # low bound freq
    high_bound = 45              # high bound freq
    sampling_rate = 1000         # sample rate

    print("Preprocessing... ")
    X_train_filtered = dp.butter_bandpass_filter(X_train_new.T, low_bound, high_bound, sampling_rate).T
    X_val_filtered = dp.butter_bandpass_filter(X_val.T, low_bound, high_bound, sampling_rate).T

    # PCA whitening
    X_train_w = dp.whitening(X_train_filtered)
    X_val_w = dp.whitening(X_val_filtered)

    # Normalize X
    scaler = StandardScaler()
    X_train_new = scaler.fit_transform(X_train_w)
    X_val = scaler.fit_transform(X_val_w)

    # print(np.max(X_train_new))
    # print(np.max(X_val))
    # print(np.min(X_train_new))
    # print(np.min(X_val))

    X_val = torch.tensor(X_val.T, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X = torch.tensor(X_train_new.T, dtype=torch.float32)
    y = torch.tensor(y_train_new, dtype=torch.float32)
    # print(X_val.shape)
    # print(y_val.shape)
    # print(X.shape)
    # print(y.shape)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    hidden_size1 = 80
    hidden_size2 = 30
    model = lr2.LinearRegressionModel(X.shape[1], y.shape[1]).to(device)
    # model = MultiLayerPerceptron(X.shape[1], hidden_size1, y.shape[1]).to(device)  #
    # model = MultiLayerPerceptron2(X.shape[1], hidden_size1, hidden_size2, y.shape[1]).to(device)  # best loss so far SGD, lr=1e-7, nesterov momentum=0.9
    # model = nn.Sequential(
    #     SaverModule(nn.Linear(X.shape[1], hidden_size1)),
    #     nn.ReLU(),
    #     SaverModule(nn.Linear(hidden_size1, hidden_size2)),
    #     nn.ReLU(),
    #     SaverModule(nn.Linear(hidden_size2, y.shape[1])),
    # ).to(device)

    momentum = 0.9
    learning_rate = args.lr
    model.apply(lr2.init_weights)
    loss_fn = nn.MSELoss()
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.99))
    else:
        if args.nesterov is True:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    num_epochs = args.epoch
    for epoch in range(num_epochs):
        epoch_loss = lr2.train(dataloader, model, loss_fn, optimizer, device)

        # Calculate average loss for the epoch
        epoch_loss /= len(dataloader.dataset)

        # Append the loss values to the lists
        train_losses.append(epoch_loss)

        print("Train error on last batch")
        train_loss, train_acc = lr2.test(X.to(device), y.to(device), model, loss_fn)

        print("Test error")
        val_loss, val_acc = lr2.test(X_val.to(device), y_val.to(device), model, loss_fn)

        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print progress
        # if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], train_Loss: {epoch_loss:.4f}, val_loss: {val_loss:.4f}')

    # Plot loss function and MSE
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='train loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='val loss')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over epoch')

    plt.tight_layout()
    # plt.savefig(f'output/Losses_LR_SGD_lr{learning_rate}_epo_{num_epochs}_neterove_{momentum}.png')
    plt.savefig(os.path.join(args.output_root, f'Losses_LR_{args.optim}_lr{learning_rate}_epo_{num_epochs}.png'))
    plt.show()

    train_accs = np.array(train_accs)
    val_accs = np.array(val_accs)

    plt.figure(figsize=(10, 5))
    for i in range(y_val.shape[1]):
        plt.plot(range(1, num_epochs + 1), val_accs[:,i], label="ch" + str(i+1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title('Validation Accuracy per Channel')

    plt.tight_layout()
    # plt.savefig(f'output/Val_Accuracy_LR_SGD_lr{learning_rate}_epo_{num_epochs}_neterove_{momentum}.png')
    plt.savefig(os.path.join(args.output_root, f'Val_Accuracy_LR_{args.optim}_lr{learning_rate}_epo_{num_epochs}.png'))
    plt.show()

    plt.figure(figsize=(10, 5))
    for i in range(y_val.shape[1]):
        plt.plot(range(1, num_epochs + 1), train_accs[:,i], label="ch" + str(i+1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.title('Train Accuracy per Channel')

    plt.tight_layout()
    # plt.savefig(f'output/Train_Accuracy_LR_SGD_lr{learning_rate}_epo_{num_epochs}_neterove_{momentum}.png')
    plt.savefig(os.path.join(args.output_root, f'Train_Accuracy_LR_{args.optim}_lr{learning_rate}_epo_{num_epochs}.png'))
    plt.show()

    train_acc = np.mean(train_accs, axis=1)
    val_acc = np.mean(val_accs, axis=1)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_acc, label='train acc')
    plt.plot(range(1, num_epochs + 1), val_acc, label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Average Accuracy overtime')

    plt.tight_layout()
    # plt.savefig(f'output/Avg_Accuracy_LR_SGD_lr{learning_rate}_epo_{num_epochs}_neterove_{momentum}.png')
    plt.savefig(os.path.join(args.output_root, f'Avg_Accuracy_LR_{args.optim}_lr{learning_rate}_epo_{num_epochs}.png'))
    plt.show()

# -----------------------------------------------------------------------------
# Legacy code
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
