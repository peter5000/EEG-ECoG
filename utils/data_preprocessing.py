# data loading and data preprocessing files
import scipy.io
import mat73
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, convolve2d

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
def pca(data, n_comp=2, std=False):
    mean = np.mean(data, axis=0)   # (|features|, )
    mean_data = data - mean
    std = np.std(data, axis=0)
    z_score = mean_data
    # if std != 0 and std performs better
    if std is True:
        z_score /= std

    cov = np.cov(z_score.T)      # (x, x) s.t. x = num_components = min(|sample|, |features|)
    # cov = np.round(cov, 2)     # Including this line is probably better for the runtime, but decreases the performance

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

    # # Compute reconstruction loss
    # loss = np.mean(np.square(recon_data - sum_signal))
    # print("Reconstruction loss ", loss)

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
    # print(eig_val)
    # print(eig_vec)
    eig_vec = eig_vec[:,indices]

    return (np.diag(1/eig_val) ** (0.5))@eig_vec.T@data.T

# Filters

# returns denominator and numerator of IIR filter
# lowcut: low-bound, highcut: highbound, fs: sampling frequency, order: order
def butter_bandpass(lowcut, highcut, fs, order=4):
    return butter(order, [lowcut, highcut], fs=fs, btype='bandpass')

# returns filtered data from butterworth
# data: data, lowcut: low-bound, highcut: highbound, fs: sampling frequency, order: order
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)    # Create Filter
    y = lfilter(b, a, data)                                     # Filter Data
    return y

# size: size of gaussian window, sigma: higher the value, smoother the gaussian, data: data, new_length: new_length
def downsample_data(data, new_length, size=15, sigma=1):
    kernel = np.exp(-0.5 * (np.arange(size) - size // 2)**2 / sigma**2)
    kernel = kernel / np.sum(kernel)
    # smoothed_data = np.convolve(data, kernel, mode='same')
    smoothed_data = np.array([np.convolve(row, kernel, mode='same') for row in data])
    x_original = np.linspace(0, 1, smoothed_data.shape[1])
    x_new = np.linspace(0, 1, new_length)
    print(x_original.shape)
    print(smoothed_data.shape)
    return np.array([np.interp(x_new, x_original, row) for row in smoothed_data])
    '''
    # Define the kernel size and create a sample kernel (for example, a simple averaging kernel)
    # Using mode='valid' to reduce size
    kernel_size = 4029
    kernel = np.ones(kernel_size) / kernel_size

    # Convolve each row with the kernel
    convolved_data = np.array([np.convolve(row, kernel, mode='valid') for row in data])
    # Create a 2D kernel that has 1 row and `kernel_size` columns (e.g., a simple averaging kernel)
    kernel = np.ones((1, kernel_size)) / kernel_size

    # Perform the convolution
    convolved_data = convolve2d(data, kernel, mode='valid', boundary='wrap')
    # Check the shape of the result
    print(convolved_data.shape)  # Expected output: (19, 319234)
    return convolved_data
    '''


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

    # data format: {name1: {matfilename1: signal, matfilename2: signal, ...}, name2: {}, ...}
    data = {}
    for key in PATHS.keys():
        data[key] = {}
        for file in PATHS[key]:
            filename, signal = loadMatFile(file)
            data[key][filename] = signal
    for key in data.keys():
        print(key)
        # print(key, data[key])