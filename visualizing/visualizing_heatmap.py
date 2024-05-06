import h5py
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load data from .mat file
mat_data = h5py.File('ECoG06_anesthesia.mat', 'r') # replace with your path of the dataset

# Check the variables present in the .mat file
print('the variables present in the .mat file is ' + list(mat_data.keys()))
# Extract the data and convert it to a DataFrame
data = mat_data['WaveData'][:, :10000]

# Convert the data to a DataFrame
ECoG06_anesthesia_1k_csv_df = pd.DataFrame(data)

# Remove the time intervals beyond the first 1000
ECoG06_anesthesia_1k_csv_df = ECoG06_anesthesia_1k_csv_df.iloc[:10000, :]

plt.figure(figsize=(15, 10))
sns.heatmap(ECoG06_anesthesia_1k_csv_df, cmap='coolwarm', cbar_kws={'label': 'Signal Amplitude'}, vmin=-1900, vmax=1600)
plt.title('Heatmap of Brain Signal Amplitude Across All Channels', fontsize=16)
plt.xlabel('Channel')
plt.ylabel('Time (ns)')

# Save the heatmap figure
heatmap_path = 'ECog_heatmap_10k.png'
plt.savefig(heatmap_path)
plt.close()
heatmap_path

