# Linear Regression model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.dataloader import dataloader

# Import data
'''
data_X = scipy.io.loadmat('data/20120904S11_EEGECoG_Chibi_Oosugi-Naoya+Nagasaka-Yasuo+Hasegawa+Naomi_ECoG128-EEG16_mat\EEG_rest.mat')
data_y = scipy.io.loadmat('data/20120904S11_EEGECoG_Chibi_Oosugi-Naoya+Nagasaka-Yasuo+Hasegawa+Naomi_ECoG128-EEG16_mat\ECoG_rest.mat')
'''

# data_X = scipy.io.loadmat('../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/ECoG_rest.mat')
# data_y = scipy.io.loadmat('../Datasets/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16/20120904S11_EEGECoG_Chibi_Oosugi_ECoG128-EEG16_mat/EEG_rest.mat')

# X = data_X['EEG']
# y = data_y['ECoG']
# X = X.T
# y = y.T

# # Split data into training, validation and test
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.9, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.8, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.fit_transform(X_val)

# '''
# X_train = X_train.T
# X_val = X_val.T
# X_test = X_test.T
# y_train = y_train.T
# y_val = y_val.T
# y_test = y_test.T
# '''

# print("Train shapes:", X_train.shape, y_train.shape)
# print("Validation shapes:", X_val.shape, y_val.shape)
# print("Test shapes:", X_test.shape, y_test.shape)

# # Convert numpy arrays to PyTorch tensors
# X_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_tensor = torch.tensor(y_train, dtype=torch.float32)

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
input_size = X_tensor.size(1)  # 16
output_size = y_tensor.size(1)  # 128
model = LinearRegressionModel(input_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Create DataLoader for batch processing
# dataset = TensorDataset(X_tensor, y_tensor)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
dataloader = dataloader(batch_size=32, shuffle=True)

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
plt.show()

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