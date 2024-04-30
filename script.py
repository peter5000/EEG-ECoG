import sys
sys.path.append('C:/Users/chans/Documents/UW/2023_2024/SP24/CSE_481F/EEG-ECoG')
from utils import data_preprocessing as dp
from visualizing import graphs as gp
import numpy as np

ecog_fp = '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/ECoG_rest.mat'
eeg_fp = '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/EEG_rest.mat'
ecog_data = dp.loadMatFile({"ecog_fp": [ecog_fp]})['20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/ECoG_rest.mat']
eeg_data = dp.loadMatFile({"eeg_fp": [eeg_fp]})['20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/EEG_rest.mat']

# print(eeg_data.shape) (17, 300000)
# print(ecog_data.shape) (256, 300000)
ecog_channel = ecog_data.shape[0]
eeg_channel = eeg_data.shape[0]
ecog_fs = 1000
eeg_fs = 1000

data, recon_data = dp.pca(eeg_data.T, 10)

print(recon_data.shape)
print(data.shape)
gp.graphAllChannels(data.T, 17, eeg_data.shape[1])
# eeg_data_z = dp.z_score(eeg_data)
# gp.graphOneChannel(eeg_data, eeg_fs, 0, 1)
# gp.graphAllChannels(eeg_data, 17, eeg_fs)
# PCA
# eeg_post_pca = np.zeros_like(eeg_data.T)
# for time in range(eeg_data.shape[1]):
# eeg_post_pca, pca = dp.pca(eeg_data, 10)
# gp.graphAllChannels(eeg_post_pca, 17, eeg_fs)
# print(eeg_post_pca)
# print('-------------')
# print(pca.components_@eeg_data)
# eeg_backto_pca = eeg_pca.inverse_transform(eeg_post_pca)

gp.graphAllChannels(recon_data.T, 17, eeg_data.shape[1])


# print(eeg_backto_pca)
# print("--------------------------")
# print(eeg_post_pca)