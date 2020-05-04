import torch
from torch import nn, optim
from get_input_args import get_input_args_for_training as get_input_args
import Timer
from model_import import load_universal_model as load_checkpoint
from model_build import build_universal_model as build_model
from model_train import train as train_model
import model_save
from torch.utils.data import DataLoader
from torchvision import datasets
from convert_class_to_idx import convert_class_to_idx
import transforms

#not using yet
from PIL import Image
from pathlib import Path
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

ImageFolder = datasets.ImageFolder

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

    "mobilenet_v2": ('classifier', 1280),
  }

def get_model_info(model_name):
  try:
    layer_name, n_inputs = model_output_name_and_input_count[model_name.lower()]
  except:
    errortext = f"Model name should be one of these {model_output_name_and_input_count.keys()}"
    raise errortext

  return layer_name, n_inputs


dropout_percent = float(arguments.dropout)
learning_rate = float(arguments.learning_rate)

epochs = int(arguments.epochs)


#check if device is available
device = 'cuda' if arguments.gpu else 'cpu'
is_cuda_available = torch.cuda.is_available

if device =='cuda' and not torch.cuda.is_available():
  raise 'There is no CUDA on this computer'

save_dir = arguments.save_dir
model_save.set_path(save_dir)

#load images
data_dir = arguments.data_dir

train_images_folder = ImageFolder (data_dir + 'train', transform=transforms.data_transforms_train)
test_images_folder = ImageFolder(data_dir + 'test', transform=transforms.data_transforms)
class_to_idx = train_images_folder.class_to_idx

train_images_dataloader = DataLoader(train_images_folder, batch_size=32, shuffle=True)
test_images_dataloader = DataLoader(test_images_folder, batch_size=32)

#load checkpoint
checkpoint = arguments.checkpoint
if checkpoint:
  model, save_data, idx_to_classes = load_checkpoint(checkpoint, dropout=dropout_percent)
  model_name = save_data['model_name']
  layer_name, n_inputs = get_model_info(model_name)
else:
  layers_output_n = len(class_to_idx)
  model_name = arguments.arch
  layer_name, n_inputs = get_model_info(model_name)

  n_hidden_layers = arguments.hidden_units

  layers = [int(n_inputs)]
  layers.extend([int(layer.strip()) for layer in n_hidden_layers.split(',')])
  layers.append(int(layers_output_n))

  model, save_data = build_model(model_name, layers, layer_name, dropout=dropout_percent, pretrained=True)

if not save_data.get('idx_to_class'):
  save_data["idx_to_class"] = convert_class_to_idx(class_to_idx)

#train model
timer = Timer.Timer('Start Training')

criterion = nn.NLLLoss()
optimizer = optim.Adam(getattr(model, layer_name).parameters(), learning_rate)

train_model(model, criterion, optimizer, train_images_dataloader, test_images_dataloader, save_data=save_data, epochs=epochs, device=device)
timer.stop('end time')

