# Transformers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io

import math
import numpy as np

from utils.dataloader import dataloader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MAX_SEQ_LEN = 3000

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TransformerModel, self).__init__()
        self.embedding_src = nn.Linear(input_size, 256)
        self.positional_encoder = PositionalEncoding(dim_model=256, dropout_p=0.1, max_len=MAX_SEQ_LEN)
        self.transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
        self.fc = nn.Linear(256, output_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, tgt_mask):
        src = self.embedding_src(src)
        src = self.positional_encoder(src)
        output = self.transformer(src, tgt, tgt_mask)
        output = self.fc(output)
        return output
        

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # input size： [seq_len, dim_model]. seq_len < max_len
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


train_loader = dataloader(batch_size=MAX_SEQ_LEN) # each batch (seq_len=300, 1, channel#)

input_size = 128  # ecog channel#
output_size = 18  # eeg channel

# Model, Loss, Optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerModel(input_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Beginning training model...")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        # print(x.size())
        # print(y.size())
        x, y = x.to(device), y.to(device)

        # Now we shift the tgt by one
        y_input = y[:-1, :, :]
        y_expected = y[1:, :, :]
        
        # Get mask to mask out the next words
        sequence_length = y.size(0)
        tgt_mask = model.generate_square_subsequent_mask(sequence_length).to(device)
        
        # print(y_input.size())
        # print(y_expected.size())
        # Standard training except we pass in y_input and tgt_mask
        pred = model(x, y_input, tgt_mask)    
        loss = criterion(pred, y_expected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")


def predict(model, input, max_length=MAX_SEQ_LEN):
    model.eval()
    
    y_input = torch.ones(1, 1, 18, dtype=torch.float, device=device)
    # print(y_input.size())
    # print(input.size())
    
    tgt_mask = model.generate_square_subsequent_mask(max_length).to(device)

    for i in range(max_length):
        pred = model(input, y_input[i].unsqueeze(0), tgt_mask)
        y_input = torch.cat((y_input, pred), dim=0)
        pred.cpu()
        # print(y_input.size())

    return y_input[1:]

print("Beginning predicting...")
# predict loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = predict(model, x)    
            loss = criterion(pred, y)
            running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")