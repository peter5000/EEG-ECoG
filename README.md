# EEG-ECoG
Project on Converting between EEG and ECoG

## Assumption
Python 3.10.x or higher\

Miniconda enviornment [link](https://docs.anaconda.com/free/miniconda/miniconda-install/)

Example:
`conda create -n "env_name" python=3.10.0`

## Setup

environment: `pip install -r requirements.txt`\
Dataset [link](http://www.www.neurotycho.org/expdatalist/listview?task=45)

We specifically used `20110607S2/EEG05_anesthesia.mat` and `20110607S2/ECoG05_anesthesia.mat` to train our model

## Run Code

### Train our linear model
`python model_eval.py --eeg_path "your relative path" --ecog_path "your relative path" --output_root "root path for graph"`

other optional arguments:\
--optim sgd or --optim adam (for different optimizer functions)\
--epoch `int` (for desired number of epochs)\
--lr `float` (for desired learning rate)\
--nesterov (set the flag to enable nesterov momentum with value=0.9)

Output graphs will be stored in output_root folder that is inputed

### Sanity Check Codes
`python sanity_checks.py --all`

or

`python sanity_checks.py --test whitening`

other options for `--test` argument:\
[whitening, filtering, sinetosine, ecogtoagg]

### Train/Predict our Transformer model
Follow the instructions in models/transformer_EECoG_20min_with_accuracy.ipynb.\
This notebook provides the best example of how to use the models/transformer_model.py, utils/TransformerDataset.py\
You will need to download the dataset from the website (http://www.www.neurotycho.org/expdatalist/listview?task=45) if you are training the model. (Our GitHub repo has limited storage to hold all these data.)\
You can also restore the fine-tuned weights from output\transfomer_weights to predict outputs, without the need of training model.

Check out models\transformer_EECoG_20min_twoMonkey.ipynb and models\transformer_EECoG_synth.ipynb for the transformer model performance on the other monkey's datasets and synthetic dataset, along with hyperparameter tuning on the validation set.

### GUI
`python gui/gui/py`

You can run a model by importing csv file and save the result by clicking save file.

## File Structure
- `data/`: Contains the data for sanity checks.
- `gui/`: Directory for the graphical user interface.
- `models/`: Contains different versions and types of models.
- `output/`: Primary output directory for generated graphs.
- `utils/`: Contains code for data loading and preprocessing.
- `visualizing/`: Directory for graphing and visualizations.

## Scripts

- `model_eval.py`: Script for training our linear model.
- `sanity_checks.py`: Script for running test files.
- `models/transformer_EECoG_20min_with_accuracy.ipynb`: Example Script for training, testing, and predicting transformer model
