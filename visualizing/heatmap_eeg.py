import scipy.io
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load data from .mat file
mat_data = scipy.io.loadmat('C:/All in it/CSE 481F/20120123S11_EEGECoG_Su_Oosugi-Naoya+Nagasaka-Yasuo+Hasegawa+Naomi_ECoG256-EEG17_mat/20120123S11_EEGECoG_Su_Oosugi-Naoya+Nagasaka-Yasuo+Hasegawa+Naomi_ECoG256-EEG17_mat/EEG_deep-anesthetic.mat') # replace with your path of the dataset

# Extract the data and convert it to a DataFrame
data = mat_data['EEG'][:, :1000]

# Convert the data to a DataFrame
df = pd.DataFrame(data)
df = df.transpose()

plt.figure(figsize=(15, 10))
sns.heatmap(df, cmap='coolwarm', cbar_kws={'label': 'Signal Amplitude'}, vmin=-1000, vmax=1000)
plt.title('Heatmap of Brain Signal Amplitude Across All Channels', fontsize=16)
plt.xlabel('Channel')
plt.ylabel('Time (ns)')

# Save the heatmap figure
heatmap_path = 'C:/All in it/CSE 481F/Eeg_heatmap_1k.png'
plt.savefig(heatmap_path)
plt.close()
heatmap_path

