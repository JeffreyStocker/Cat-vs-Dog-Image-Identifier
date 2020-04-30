import torch
from PIL import Image

from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path
from get_input_args import get_input_args_for_training as get_input_args
import Timer
import model_train
import model_import from model_import

'''
    Training a network 	      train.py successfully trains a new network on a dataset of images
    Training validation log 	The training loss, validation loss, and validation accuracy are printed out as a network trains
    Model architecture 	      The training script allows users to choose from at least two different architectures available from torchvision.models
    Model hyperparameters 	  The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
    Training with GPU 	      The training script allows users to choose training the model on a GPU
    Predicting classes 	      The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
    Top K classes 	          The predict.py script allows users to print out the top K classes along with associated probabilities
    Displaying class names 	  The predict.py script allows users to load a JSON file that maps the class values to other category names
    Predicting with GPU 	    The predict.py script allows users to use the GPU to calculate the predictions
 '''

is_cuda_available = torch.cuda.is_available
device = "cuda" if is_cuda_available() else 'cpu'


'''
Commandline arguments
  arch = pick architecture structure
  device = cpu or gpu
  checkpoint = path to checkpoint
  learing_rate = set learining rate
  number of hidden unites =
  epochs = number of epochs to train with
  save_location = location to save training data
  image_path = path to images
 '''
arguments = get_input_args().parse_args()
print(arguments)

if arguments['checkpoint']:


start_time = Timer.Timer('Start Training')