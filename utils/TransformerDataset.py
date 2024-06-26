# remember first "pip install positional-encodings[pytorch]"
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import torch
from torch.utils.data import Dataset

class TransformerDataset(Dataset):
    ## data and target input size --> (channel, time)
    def __init__(self, data, targets, device, normalize=True):
        # Move data to CUDA
        data = data.to(device)

        # Positional encoding across *location*: this lib encodes along the last dimension
        summer = Summer(PositionalEncoding1D(data.size(1))).to(device)
        pre_data = summer(data.unsqueeze(1))

        # Transformer implementation expects (batch, electrode, time) for electrode encoding
        pre_data = pre_data.permute(0, 2, 1)

        # Normalize data and targets if required
        if normalize:
            # Normalize data by the mean and std of the whole dataset
            self.data_mean = pre_data.mean()
            self.data_std = pre_data.std()
            self.data = (pre_data - self.data_mean) / (self.data_std + 1e-6)

            # Compute and store mean and std for targets along the second dimension
            self.target_mean = targets.mean(dim=0, keepdim=True)
            self.target_std = targets.std(dim=0, keepdim=True)
            self.targets = (targets - self.target_mean) / (self.target_std + 1e-6)
            self.targets = targets.to(device)
        else:
            self.data = pre_data.to(device)
            self.targets = targets.to(device)
            self.data_mean = None
            self.data_std = None
            self.target_mean = None
            self.target_std = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def denormalize_targets(self, normalized_targets):
        """Denormalize the targets using stored mean and std."""
        if self.target_mean is not None and self.target_std is not None:
            return normalized_targets * self.target_std + self.target_mean
        else:
            raise ValueError("Targets were not normalized or mean and std were not saved.")