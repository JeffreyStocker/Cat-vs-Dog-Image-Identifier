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
import torch
from PIL import Image

from torch import nn, optim
from torch.utils.data import DataLoader
import datetime
from pathlib import Path

from get_input_args import get_input_args


start_time = datetime.datetime.now()

is_cuda_available = torch.cuda.is_available
device = "cuda" if is_cuda_available() else 'cpu'

'''
Commandline arguments
  arch = pick architecture structure
  device = cpu or gpu
  checkpoint = path to checkpoint
 '''
arguments = get_input_args().parse_args()
