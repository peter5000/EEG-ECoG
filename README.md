# EEG-ECoG
Project on Converting between EEG and ECoG

## Setup
pip install -r requirements.txt

## Data Preprocess
utils/data_preprocessing.py: PCA and whitening

## ML models
models/linear_regression_model.py: Linear regression training on raw data from ecog to eeg
models/transformer.py: transformer model training on raw data from ecog to eeg

## Visualizing
Visualizing/graphs.py: Graphing the signals for preprocessing
Visualizing/heatmap.py: Graphing the signals in heatmap for bigger view windows

## GUI
gui/gui.py
