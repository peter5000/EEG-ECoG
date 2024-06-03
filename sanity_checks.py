# Tests and example usage of our models

import sys
sys.path.append('../EEG-ECoG') # adding path for packages
from utils import data_preprocessing as dp
from visualizing import graphs as gp
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models import linear_regresion_model2 as lr2
import random

# PCA whitening sanity check
def SanityCheckWhitening():
    print("start testing PCA whitening")

    # Generate random data
    np.random.seed(0)
    data = np.random.randn(100, 5)

    # Apply whitening
    whitened_data = dp.whitening(data)

    # Calculate covariance matrix of the whitened data
    cov_whitened = np.cov(whitened_data)

    # Check if the covariance matrix is close to identity matrix
    is_identity = np.allclose(cov_whitened, np.eye(cov_whitened.shape[0]))

    print("covariance whitened", cov_whitened)

    if is_identity:
        print("The whitened data has covariance matrix close to identity matrix, PCA whitening is successful!")
    else:
        print("The whitened data does not have covariance matrix close to identity matrix, PCA whitening might not be successful.")

    # # Compute reconstruction loss
    # loss = np.mean(np.square(recon_data - sum_signal))
    # print("Reconstruction loss ", loss)

def SanityCheckFiltering():
    print("start testing 4th order butterworth filter")
    # f = number of phases, Fs = frequency, x = sampling rate
    signal1 = generateSineWave(0, 5, 100, 10)
    signal2 = generateSineWave(0, 5, 100, 5)
    output = dp.butter_bandpass_filter(signal1 + signal2, 4, 6, 100, 4)
    fig, ax = plt.subplots(1,4, figsize= (15, 15))
    ax[0].plot(signal1)
    ax[1].plot(signal2)
    ax[2].plot(signal1 + signal2)
    ax[3].plot(output)
    plt.show()

# Generate sinewave
def generateSineWave(start_time, end_time, sample_rate, frequency, amplitude=1, offset=0):
    time = np.arange(start_time, end_time, 1/sample_rate)
    return amplitude * np.sin(2 * np.pi * frequency * time + offset)

# Synthetic data test
def sineToSine(device):
    print("start testing synthetic ecog to synthetic eeg")
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

    print("synthetic ecog shape: ", synth_ecog.shape)
    print("synthetic eeg shape: ", synth_eeg.shape)
    X = torch.tensor(synth_ecog, dtype=torch.float32).T
    y = torch.tensor(synth_eeg, dtype=torch.float32).T
    # print(y.shape)
    # print(X.shape)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = lr2.LinearRegressionModel(X.shape[1], y.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.99))

    loss_values = []
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = lr2.train(dataloader, model, loss_fn, optimizer, device)

        # Calculate average loss for the epoch
        epoch_loss /= len(dataloader.dataset)

        # Append the loss values to the lists
        loss_values.append(epoch_loss)

        # Print progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Plot loss function and MSE
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, num_epochs + 1), loss_values, label='loss')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Synthetic ECoG to Synthetic EEG')

    plt.tight_layout()
    plt.show()

    # Plot synthetic data
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

# -----------------------------------------------------------------------------
# aggregate ecog
def ecogToAggEcog(device, ecog_data):
    print("start testing ecog to aggregated ecog")
    agg_ecog = np.zeros((4,319234))
    agg_ecog[0] = ecog_data[9] + ecog_data[10] + ecog_data[16] + ecog_data[17]
    agg_ecog[1] = ecog_data[41] + ecog_data[42] + ecog_data[52] + ecog_data[53]
    agg_ecog[2] = ecog_data[88] + ecog_data[89] + ecog_data[92] + ecog_data[93]
    agg_ecog[3] = ecog_data[108] + ecog_data[94] + ecog_data[109] + ecog_data[123]
    agg_ecog /= 4
    X = torch.tensor(ecog_data, dtype=torch.float32).T
    y = torch.tensor(agg_ecog, dtype=torch.float32).T
    # print(y.shape)
    # print(X.shape)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    hidden_size = 50
    model = lr2.LinearRegressionModel(X.shape[1], y.shape[1]).to(device)    # Adam lr=5e-5, betas=(0.9, 0.99) end_loss=0.0278
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9,0.99))

    loss_values = []
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = lr2.train(dataloader, model, loss_fn, optimizer, device)

        # Calculate average loss for the epoch
        epoch_loss /= len(dataloader.dataset)

        # Append the loss values to the lists
        loss_values.append(epoch_loss)

        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}], train_Loss: {epoch_loss:.4f}')

    # Plot loss function and MSE
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, num_epochs + 1), loss_values, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ECoG to Aggregate ECoG')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--test', type=str, default='whitening')    # whitening, filtering, sinetosine, ecogtoagg
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
    ecog_path = "data/ECoG05_anesthesia.mat"
    _, ecog_data = dp.loadMatFile(ecog_path)

    if args.all is True:
        SanityCheckWhitening()
        SanityCheckFiltering()
        sineToSine(device)
        ecogToAggEcog(device, ecog_data)
    elif args.test == "whitening":
        SanityCheckWhitening() #
    elif args.test == "filtering":
        SanityCheckFiltering()
    elif args.test == "sinetosine":
        sineToSine(device)
    elif args.test ==  "ecogtoagg":
        ecogToAggEcog(device, ecog_data)
    else:
        print("Please set --all or choose one for --test [whitening, filtering, sinetosine, ecogtoagg]")

# ---------------------------------------------------
# LEGACY CODE
# ---------------------------------------------------

# # PCA sanity check
# def SanityCheckPCA(Fs=1000, f=5, sample=1000):
#     x = np.arange(sample)
#     signal = generateSineWave(0, 3, 10, 2)
#     signal2 = np.arange(30)

#     sum_signal = np.vstack((signal, signal2))
#     pca_result, eig_vec, mean, std = dp.pca(sum_signal.T)

#     recon_data = pca_result[:,1:].dot(eig_vec[:,1:].T) * std + mean
#     fig, ax = plt.subplots(2,3, figsize= (15, 15))
#     ax[0][0].plot(range(pca_result.shape[0]), pca_result[:,0], color='blue', marker='.')
#     ax[1][0].plot(range(pca_result.shape[0]), pca_result[:,1], color='blue', marker='.')
#     ax[0][1].plot(range(sum_signal.T.shape[0]), sum_signal.T[:,0], color='green', marker='.')
#     ax[1][1].plot(range(sum_signal.T.shape[0]), sum_signal.T[:,1], color='green', marker='.')
#     ax[0][2].plot(range(recon_data.shape[0]), recon_data[:,0], color='blue', marker='.')
#     ax[1][2].plot(range(recon_data.shape[0]), recon_data[:,1], color='blue', marker='.')
#     plt.show()
        # SanityCheckPCA()       # passed
    # else:
    #     # example usage
    #     ecog_fp = '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/ECoG_rest.mat'
    #     eeg_fp = '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/EEG_rest.mat'
    #     _, ecog_data = dp.loadMatFile(ecog_fp)
    #     _, eeg_data = dp.loadMatFile(eeg_fp)

    #     # print(eeg_data.shape) (17, 300000)
    #     # print(ecog_data.shape) (256, 300000)
    #     ecog_fs = 1000
    #     eeg_fs = 1000
    #     window_length = 3
    #     ecog_channel = ecog_data.shape[0]
    #     eeg_channel = eeg_data.shape[0]

    #     pca_data, eig_vec, mean, std = dp.pca(eeg_data[:,:eeg_fs*window_length].T, 17)
    #     recon_data = pca_data[:,1:].dot(eig_vec[:,1:].T) * std + mean

    #     print(recon_data.shape)
    #     print(pca_data.shape)
    #     gp.graphAllChannels(eeg_data[:,:eeg_fs*window_length], 17, eeg_fs*window_length)
    #     gp.graphAllChannels(pca_data.T, 17, eeg_fs*window_length)
    #     gp.graphAllChannels(recon_data.T, 17, eeg_fs*window_length)

    #     pca_data_2, eig_vec_2, mean_2, std_2 = dp.pca(ecog_data[:,:ecog_fs*window_length].T, 200)
    #     recon_data_2 = pca_data_2[:,1:].dot(eig_vec_2[:,1:].T) * std_2 + mean_2
    #     gp.graphAllChannels(ecog_data[:,:ecog_fs*window_length], 20, eeg_fs*window_length)
    #     gp.graphAllChannels(pca_data_2.T, 20, ecog_fs*window_length)
    #     gp.graphAllChannels(recon_data_2.T, 20, ecog_fs*window_length)

    #     # print(eeg_backto_pca)
    #     # print("--------------------------")
    #     # print(eeg_post_pca)
    #     pass