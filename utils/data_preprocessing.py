# data loading and data preprocessing files

import scipy.io
import mat73
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Input format: {name1: [path1, path2, ...], name2: [path1, path2, ...], ...}
# output format: {name1/matfilename1: numpy.ndarray, name1/matfilename2: numpy.ndarray,}
def loadMatFile(paths):
    data = {}
    for path in paths.keys():
        for matfile in paths[path]:
            name = matfile.split("/")
            key = name[-2] + "/" + name[-1]
            # Leveraging the fact that the data is always at the end
            try:
                temp = scipy.io.loadmat(matfile)
                data[key] = temp[list(temp.keys())[-1]]
                # print(list(temp.keys())[-1])

            except NotImplementedError:
                temp = mat73.loadmat(matfile)
                data[key] = temp[list(temp.keys())[-1]]
            except:
                ValueError('could not read at all...')
    return data

# PCA: data = (sample, features)
def pca(data, n_comp):
    mean = np.mean(data, axis=0)
    print("Mean ", mean.shape)
    mean_data = data - mean
    print("Data after subtracting mean ", data.shape, "\n")

    cov = np.cov(mean_data.T)
    cov = np.round(cov, 2)
    print("Covariance matrix ", cov.shape, "\n")

    eig_val, eig_vec = np.linalg.eig(cov)

    indices = np.arange(0,len(eig_val), 1)
    indices = ([x for _,x in sorted(zip(eig_val, indices))])[::-1]
    eig_val = eig_val[indices]
    eig_vec = eig_vec[:,indices]

    # Get explained variance
    sum_eig_val = np.sum(eig_val)
    explained_variance = eig_val/ sum_eig_val
    print("Explained variance ", explained_variance)
    cumulative_variance = np.cumsum(explained_variance)
    print("Cumulative variance ", cumulative_variance)

    # Plot explained variance
    plt.plot(np.arange(0, len(explained_variance), 1), cumulative_variance)
    plt.title("Explained variance vs number of components")
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance")
    plt.show()

    ## We will 2 components
    n_comp = 17
    eig_vec = eig_vec[:,:n_comp]
    print(eig_vec.shape)

    # Take transpose of eigen vectors with data
    pca_data = mean_data.dot(eig_vec)
    print("Transformed data ", pca_data.shape)

    # Plot data

    fig, ax = plt.subplots(1,3, figsize= (15,15))
    # Plot original data
    ax[0].scatter(data[:,0], data[:,1], color='blue', marker='.')

    # Plot data after subtracting mean from data
    ax[1].scatter(mean_data[:,0], mean_data[:,1], color='red', marker='.')

    # Plot data after subtracting mean from data
    ax[2].scatter(pca_data[:,0], pca_data[:,1], color='red', marker='.')

    # Set title
    ax[0].set_title("Scatter plot of original data")
    ax[1].set_title("Scatter plot of data after subtracting mean")
    ax[2].set_title("Scatter plot of transformed data")

    # Set x ticks
    ax[0].set_xticks(np.arange(-8, 1, 8))
    ax[1].set_xticks(np.arange(-8, 1, 8))
    ax[2].set_xticks(np.arange(-8, 1, 8))

    # Set grid to 'on'
    ax[0].grid('on')
    ax[1].grid('on')
    ax[2].grid('on')

    major_axis = eig_vec[:,0].flatten()
    xmin = np.amin(pca_data[:,0])
    xmax = np.amax(pca_data[:,0])
    ymin = np.amin(pca_data[:,1])
    ymax = np.amax(pca_data[:,1])

    plt.show()
    plt.close('all')

    # Reverse PCA transformation
    recon_data = pca_data[:,2:].dot(eig_vec[:,2:].T) + mean
    print(recon_data.shape)

    # Plot reconstructed data

    fig, ax = plt.subplots(1,3, figsize= (15, 15))
    ax[0].scatter(data[:,0], data[:,1], color='blue', marker='.')
    ax[1].scatter(mean_data[:,0], mean_data[:,1], color='red', marker='.')
    ax[2].scatter(recon_data[:,0], recon_data[:,1], color='red', marker='.')
    ax[0].set_title("Scatter plot of original data")
    ax[1].set_title("Scatter plot of data after subtracting mean")
    ax[2].set_title("Scatter plot of reconstructed data")
    ax[0].grid('on')
    ax[1].grid('on')
    ax[2].grid('on')
    plt.show()

    # Compute reconstruction loss
    loss = np.mean(np.square(recon_data - data))
    print("Reconstruction loss ", loss)

    # pca = PCA(n_components = n_channel)

    # data = pca.fit_transform(data.T).T

    # some preprocessing

    # inverse transform
    # train = pca.inverse_transform
    # test = pca.inverse_transform

    return data, recon_data

def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X.T).T
    return Xz


# FFT


# Bandpass filter



# Example usages
if __name__ == "__main__":
  # Paths to datasets
  PATHS = {
    "20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat" :
    [
      '../Datasets/20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG01.mat',
      '../Datasets/20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG02.mat',
      '../Datasets/20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG01.mat',
      '../Datasets/20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG02.mat',
    ],
    "20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat" :
    [
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG01_anesthesia.mat',
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG02_anesthesia.mat',
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG03_anesthesia.mat',
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG04_anesthesia.mat',
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG05_anesthesia.mat',
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG01_anesthesia.mat',
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG02_anesthesia.mat',
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG03_anesthesia.mat',
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG04_anesthesia.mat',
      '../Datasets/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG05_anesthesia.mat',
    ],
    "20110607S3_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat" :
    [
      '../Datasets/20110607S3_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S3_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG06_anesthesia.mat',
      '../Datasets/20110607S3_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S3_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG06_anesthesia.mat',
    ],
    "20110607S11_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat" :
    [
      '../Datasets/20110607S11_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S11_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG01.mat',
      '../Datasets/20110607S11_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S11_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG02.mat',
      '../Datasets/20110607S11_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S11_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG01.mat',
      '../Datasets/20110607S11_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S11_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG02.mat',
    ],
    "20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat" :
    [
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG01_anesthesia.mat',
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG02_anesthesia.mat',
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG03_anesthesia.mat',
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG04_anesthesia.mat',
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG05_anesthesia.mat',
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG01_anesthesia.mat',
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG02_anesthesia.mat',
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG03_anesthesia.mat',
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG04_anesthesia.mat',
      '../Datasets/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18/20110607S12_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG05_anesthesia.mat',
    ],
    "20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat" :
    [
      '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/ECoG_rest.mat',
      '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/EEG_rest.mat',
      '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/ECoG_low-anesthetic.mat',
      '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/EEG_low-anesthetic.mat',
      '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/ECoG_deep-anesthetic.mat',
      '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/EEG_deep-anesthetic.mat',
      '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/ECoG_recovery.mat',
      '../Datasets/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17/20120123S11_EEGECoG_Su_Oosugi_ECoG256-EEG17_mat/EEG_recovery.mat',
    ],
    "20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat" :
    [
      '../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/ECoG_rest.mat',
      '../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/EEG_rest.mat',
      '../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/ECoG_low-anesthetic.mat',
      '../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/EEG_low-anesthetic.mat',
      '../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/ECoG_deep-anesthetic.mat',
      '../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/EEG_deep-anesthetic.mat',
      '../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/ECoG_recovery.mat',
      '../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/EEG_recovery.mat',
    ],
  }

  # data format: {name1/matfilename1:
  #               {'__header__': string, '__version__': string, '__globals__': list, ['EEG' or 'ECoG']: numpy.ndarray},
  #                name1/matfilename2: ...}}
  data = loadMatFile(PATHS)
  # print(data["20110607S1_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG01.mat"])
  for key in data.keys():
    # print(key, data[key].keys())
    print(key, data[key])