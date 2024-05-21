# Linear Regression model
import sys
sys.path.append('../EEG-ECoG') # adding path for packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from utils.dataloader import dataloader
from utils import data_preprocessing as dp

# Import data
'''
data_X = scipy.io.loadmat('data/20120904S11_EEGECoG_Chibi_Oosugi-Naoya+Nagasaka-Yasuo+Hasegawa+Naomi_ECoG128-EEG16_mat\EEG_rest.mat')
data_y = scipy.io.loadmat('data/20120904S11_EEGECoG_Chibi_Oosugi-Naoya+Nagasaka-Yasuo+Hasegawa+Naomi_ECoG128-EEG16_mat\ECoG_rest.mat')
'''
data_X = dp.loadMatFile('../Datasets/20110607S2_EEGandECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/ECoG05_anesthesia.mat')
data_y = dp.loadMatFile('../Datasets/20110607S2_EEGandECoG_Su_Oosugi_ECoG128-EEG18/20110607S2_EEGECoG_Su_Oosugi_ECoG128-EEG18_mat/EEG05_anesthesia.mat')
'''
data_X = scipy.io.loadmat('../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/ECoG_rest.mat')
data_y = scipy.io.loadmat('../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/EEG_rest.mat')
'''

X = data_X[1]
y = data_y[1]

print(X.shape)
print(y.shape)
# downsample if mismatch in size of datasets
if data_X[1].shape[1] != data_y[1].shape[1]:
    # gaussian normalization
    if X.shape[1] > y.shape[1]:
        X = dp.downsample_data(X, y.shape[1])
    elif X.shape[1] < y.shape[1]:
        y = dp.downsample_data(y, X.shape[1])

X = X.T
y = y.T

print(X.shape)
print(y.shape)
# Split data into training, validation and test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Normalize X
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)

# '''
# X_train = X_train.T
# X_val = X_val.T
# X_test = X_test.T
# y_train = y_train.T
# y_val = y_val.T
# y_test = y_test.T
# '''

print("Train shapes:", X_train.shape, y_train.shape)
print("Validation shapes:", X_val.shape, y_val.shape)
print("Test shapes:", X_test.shape, y_test.shape)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        '''
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        '''

    def forward(self, x):
        '''
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        '''
        return self.linear(x)

# Instantiate the model
input_size = X_tensor.size(1)  # 129 (ecog)
output_size = y_tensor.size(1)  # 19 (eeg)
print(input_size)
print(output_size)
model = LinearRegressionModel(input_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Create DataLoader for batch processing
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# dataloader = dataloader(batch_size=32, shuffle=True)

# Keep track of loss and error
loss_values = []
mse_values = []

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    epoch_loss = 0.0
    epoch_mse = 0.0
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Calculate MSE
        mse = torch.mean((outputs - labels) ** 2)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        epoch_mse += mse.item() * inputs.size(0)

    # Calculate average loss and mse for the epoch
    epoch_loss /= len(dataloader.dataset)
    epoch_mse /= len(dataloader.dataset)

    # Append the loss values to the lists
    loss_values.append(epoch_loss)
    mse_values.append(epoch_mse)

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # 4-fold cross-validation
# k = 4
# loss_dict = {}
# mse_dict = {}
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
#     X_train_filtered = dp.butter_bandpass_filter(X_train_new.T, low_bound, high_bound, sampling_rate)
#     X_val_filtered = dp.butter_bandpass_filter(X_val.T, low_bound, high_bound, sampling_rate)

#     # PCA whitening
#     X_train_w = dp.whitening(X_train_filtered.T)
#     X_val_w = dp.whitening(X_val_filtered.T)

#     # Convert numpy arrays to PyTorch tensors
#     X_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_tensor = torch.tensor(y_train, dtype=torch.float32)

#     # Instantiate the model
#     input_size = X_tensor.size(1)   # 129
#     output_size = y_tensor.size(1)  # 19
#     model = LinearRegressionModel(input_size, output_size)

#     # Define loss function and optimizer
#     loss_fn = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.99))

# Plot loss function and MSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), mse_values, label='MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Mean Squared Error')

plt.tight_layout()
plt.savefig('loss_mse_linreg.png')

# Evaluate on validation set

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Make predictions on validation set
model.eval()
with torch.no_grad():
    y_pred = model(X_val_tensor)

# Calculate evaluation metrics
mse = mean_squared_error(y_val_tensor.numpy(), y_pred.numpy())
rmse = mean_squared_error(y_val_tensor.numpy(), y_pred.numpy(), squared=False)
mae = mean_absolute_error(y_val_tensor.numpy(), y_pred.numpy())
r2 = r2_score(y_val_tensor.numpy(), y_pred.numpy())

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R2): {r2:.4f}')