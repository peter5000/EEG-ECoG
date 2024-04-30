# Graphing datas
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('C:/Users/chans/Documents/UW/2023_2024/SP24/CSE_481F/EEG-ECoG')
from utils import data_preprocessing as dp

# print(np.linspace(0,10,ecog_fs*10).shape)
def graphOneChannel(data, sample_rate, channel, length=1, figsize=[10,4]):
    plt.figure(figsize=figsize)
    plt.plot(np.linspace(0,length,sample_rate*length),data.T[:sample_rate*length,channel])
    plt.xlabel('time (s)')
    plt.ylabel('potential (uV)')
    plt.title(f'{length} seconds of ECoG data')

    plt.show()

def graphAllChannels(data, channel_size, sample_rate):
    plt.figure(figsize=[50,8])
    for ch in range(channel_size):
        plt.subplot(2,int((channel_size+1)/2),ch+1)
        plt.plot(data.T[:sample_rate,ch])
        plt.xlabel('samples')
        plt.ylabel('potential(uV)')
        plt.title(ch)

    plt.tight_layout()
    plt.show()
