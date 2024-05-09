import sys
sys.path.append('C:/Users/chans/Documents/UW/2023_2024/SP24/CSE_481F/EEG-ECoG') # adding path for packages
from utils import data_preprocessing as dp
from visualizing import graphs as gp
import matplotlib.pyplot as plt
import numpy as np

# PCA sanity check
def SanityCheckPCA(Fs=1000, f=5, sample=1000):
    x = np.arange(sample)
    signal = np.sin(2 * np.pi * f * x / Fs)
    signal2 = np.sin(2 * np.pi * f * x / Fs)
    signal2[0] = signal2[1]*300

    sum_signal = np.vstack((signal, signal2))
    pca_result, eig_vec, mean, std = dp.pca(sum_signal.T)

    recon_data = pca_result[:,:1].dot(eig_vec[:,:1].T) * std + mean
    fig, ax = plt.subplots(2,3, figsize= (15, 15))
    ax[0][0].plot(range(pca_result.shape[0]), pca_result[:,0], color='blue', marker='.')
    ax[1][0].plot(range(pca_result.shape[0]), pca_result[:,1], color='blue', marker='.')
    ax[0][1].plot(range(sum_signal.T.shape[0]), sum_signal.T[:,0], color='green', marker='.')
    ax[1][1].plot(range(sum_signal.T.shape[0]), sum_signal.T[:,1], color='green', marker='.')
    ax[0][2].plot(range(recon_data.shape[0]), recon_data[:,0], color='blue', marker='.')
    ax[1][2].plot(range(recon_data.shape[0]), recon_data[:,1], color='blue', marker='.')
    plt.show()

# PCA whitening sanity check
def SanityCheckWhitening():# Generate random data
    np.random.seed(0)
    data = np.random.randn(100, 5)

    # Apply whitening
    whitened_data = dp.whitening(data)

    # Calculate covariance matrix of the whitened data
    cov_whitened = np.cov(whitened_data)

    # Check if the covariance matrix is close to identity matrix
    is_identity = np.allclose(cov_whitened, np.eye(cov_whitened.shape[0]))

    print(cov_whitened)

    if is_identity:
        print("The whitened data has covariance matrix close to identity matrix, PCA whitening is successful!")
    else:
        print("The whitened data does not have covariance matrix close to identity matrix, PCA whitening might not be successful.")

    # # Compute reconstruction loss
    # loss = np.mean(np.square(recon_data - sum_signal))
    # print("Reconstruction loss ", loss)

# def SanityCheckFiltering(data):
#     signal = np.sin(2 * np.pi * f * x / Fs)
#     output = dp.butter_bandpass_filter(data, lowcut, highcut, fs, order)
#     plt.plot()
#     plt.show()

# ecog_fp = '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/ECoG_rest.mat'
# eeg_fp = '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/EEG_rest.mat'
# ecog_data = dp.loadMatFile(ecog_fp)[1]
# eeg_data = dp.loadMatFile(eeg_fp)[1]

# # print(eeg_data.shape) (17, 300000)
# # print(ecog_data.shape) (256, 300000)
# ecog_channel = ecog_data.shape[0]
# eeg_channel = eeg_data.shape[0]
# ecog_fs = 1000
# eeg_fs = 1000

# data, recon_data = dp.pca(eeg_data.T, 2)

# print(recon_data.shape)
# print(data.shape)
# gp.graphAllChannels(data.T, 17, eeg_data.shape[1])
# # eeg_data_z = dp.z_score(eeg_data)
# # gp.graphOneChannel(eeg_data, eeg_fs, 0, 1)
# # gp.graphAllChannels(eeg_data, 17, eeg_fs)
# # PCA
# # eeg_post_pca = np.zeros_like(eeg_data.T)
# # for time in range(eeg_data.shape[1]):
# # eeg_post_pca, pca = dp.pca(eeg_data, 10)
# # gp.graphAllChannels(eeg_post_pca, 17, eeg_fs)
# # print(eeg_post_pca)
# # print('-------------')
# # print(pca.components_@eeg_data)
# # eeg_backto_pca = eeg_pca.inverse_transform(eeg_post_pca)

# gp.graphAllChannels(recon_data.T, 17, eeg_data.shape[1])

# ecog_data_pca, recon_ecog_data = dp.pca(ecog_data.T, 200)
# gp.graphAllChannels(ecog_data_pca.T, 20, ecog_data.shape[1])
# gp.graphAllChannels(recon_ecog_data.T, 20, ecog_data.shape[1])

# # print(eeg_backto_pca)
# # print("--------------------------")
# # print(eeg_post_pca)

if __name__ == "__main__":
    # SanityCheckPCA()       # passed
    SanityCheckWhitening() # passed
