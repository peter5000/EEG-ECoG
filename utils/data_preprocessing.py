# Load a mat file

import scipy.io

# Input format: {name1: [path1, path2, ...], name2: [path1, path2, ...], ...}
# output format: {name1/matfilename1:
#                  {'__header__': string, '__version__': string, '__globals__': list, ['EEG' or 'ECoG']: numpy array},
#                 name1/matfilename2: ...}
def loadMatFile(paths):
  data = {}
  for path in paths.keys():
    for matfile in paths[path]:
      name = matfile.split("/")
      key = name[-2] + "/" + name[-1]
      data[key] = scipy.io.loadmat(matfile)
  return data

# Example usages
if __name__ == "__main__":
  # Paths to datasets
  PATHS = {
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

  # output format: {name1/matfilename1:
  #                  {'__header__': string, '__version__': string, '__globals__': list, ['EEG' or 'ECoG']: numpy array},
  #                 name1/matfilename2: ...}
  data = loadMatFile(PATHS)

  print(data)