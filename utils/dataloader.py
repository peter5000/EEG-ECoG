import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# only return train dataset for now
def dataloader(batch_size=32, shuffle=False):
    # Import data
    data_X = scipy.io.loadmat('../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/ECoG_rest.mat')
    data_y = scipy.io.loadmat('../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/EEG_rest.mat')

    X = data_X['EEG']
    y = data_y['ECoG']
    X = X.T
    y = y.T

    # Split data into training, validation and test
    if (shuffle == True):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.9, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.8, random_state=42)
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.9, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.8, shuffle=False)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)

    '''
    X_train = X_train.T
    X_val = X_val.T
    X_test = X_test.T
    y_train = y_train.T
    y_val = y_val.T
    y_test = y_test.T
    '''

    print("Train shapes:", X_train.shape, y_train.shape)
    print("Validation shapes:", X_val.shape, y_val.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create DataLoader for batch processing
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size, shuffle=shuffle)
