import torch
from PIL import Image

from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path
from get_input_args import get_input_args_for_training as get_input_args
import Timer
import model_train
from model_import import load_universal_model as load_checkpoint
from model_build import build_universal_model as build_model

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

model_output_name_and_input_count = {
    "alexnet": ('classifier', 9216),

    "vgg11": ('classifier', 25088),
    "vgg13": ('classifier', 25088),

    "vgg11_bn": ('classifier', 25088),
    "vgg13_bn": ('classifier', 25088),
    "vgg16": ('classifier', 25088),
    "vgg16_bn": ('classifier', 25088),
    "vgg19_bn": ('classifier', 25088),

    "resnet50": ('fc', 2048),
    "resnet101": ('fc', 2048),
    "resnet152": ('fc', 2048),

    "squeezenet1_0": ('classifier', 512),
    "squeezenet1_1": ('classifier', 512),

    "densenet121": ('classifier', 1024),
    "densenet169": ('classifier', 1664),
    "densenet161": ('classifier', 2208),
    "densenet201": ('classifier', 1920),

    "shufflenet_v2_x0_5": ('fc', 1024),
    "shufflenet_v2_x1_0": ('fc', 1024),
    "shufflenet_v2_x1_5": ('fc', 1024),
    "shufflenet_v2_x2_0": ('fc', 2048),

    "mobilenet_v2": ('classifier', 1280),
  }

def get_model_info(model_name):
  try:
    layer_name, n_inputs = model_output_name_and_input_count[model_name.lower()]
  except:
    errortext = f"Model name should be one of these {f'Model name should be one of these {}'.keys()}"
    raise errortext

  return layer_name, n_inputs


layers_output_n = 0 #TODO



if arguments['checkpoint']:
  model, save_data, idx_to_classes = load_checkpoint(arguments['checkpoint'], dropout=arguments['dropout'])
else:
  model_name = arguments['model_name']
  percent_dropouts = arguments['dropouts']
  n_hidden_layers = arguments['n_hidden_layers']

  layer_name, n_inputs = get_model_info(model_name)

  layers = [int(n_inputs)]

  for n in n_hidden_layers:
    layers.append(int(n))
  else:
    layers.append(layers_output_n)

  model, save_data = build_model(model_name, layers, layer_name, dropout=percent_dropouts, pretrained=True)




start_time = Timer.Timer('Start Training')