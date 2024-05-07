# data loading and data preprocessing files

import scipy.io
import mat73
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Input format: string (file path)
# output format: (file name, signal)
def loadMatFile(file_path):
    name = file_path.split("/")
    key = name[-1]
    # Leveraging the fact that the data is always at the end
    try:
        temp = scipy.io.loadmat(file_path)
        return (key, temp[list(temp.keys())[-1]])
    except NotImplementedError:
        temp = mat73.loadmat(file_path)
        return (key, temp[list(temp.keys())[-1]])
    except:
        ValueError('could not read at all...')
        return None

# PCA
# data: (|samples|, |features|)
def pca(data, n_comp=2):
    mean = np.mean(data, axis=0)   # (|features|, )
    mean_data = data - mean
    std = np.std(data, axis=0)
    z_score = mean_data / std

    cov = np.cov(z_score.T)      # (x, x) s.t. x = num_components = min(|sample|, |features|)
    # cov = np.round(cov, 2)     # This line is probably better for runtime, but decreases the performance

    # eig_val: (|num_components|, )
    # eig_vec: (|features|, |num_components|)
    eig_val, eig_vec = np.linalg.eig(cov)

    # Sorting eig_vecs from the biggest eiv_val to the smallest
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

    ## We will {n_comp} components
    eig_vec = eig_vec[:,:n_comp]
    print(eig_vec.shape)

    # Take transpose of eigen vectors with data
    pca_data = mean_data.dot(eig_vec)
    print("Transformed data ", pca_data.shape)

    return pca_data, eig_vec, mean, std

# PCA whitening
# data: (|samples|, |features|)
def whitening(data):
    mean = np.mean(data, axis=0)   # (|features|, )
    mean_data = data - mean

    cov = np.cov(mean_data.T)      # (x, x) s.t. x = num_components = min(|sample|, |features|)

    # eig_val: (|num_components|, )
    # eig_vec: (|features|, |num_components|)
    # A: diagonal matrix with eig_val
    eig_val, eig_vec = np.linalg.eig(cov)

    # Sorting eig_vecs from smallest eiv_val to largest
    indices = np.argsort(eig_val)[::-1]
    eig_val = eig_val[indices]
    eig_vec = eig_vec[:,indices]

    return (np.diag(1/eig_val) ** (0.5))@eig_vec.T@data.T

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