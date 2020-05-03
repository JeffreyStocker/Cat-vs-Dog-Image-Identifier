#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#
# PROGRAMMER: Jeffrey Stocker
# DATE CREATED: 4-11-2020
# REVISED DATE: 4-11-2020
# PURPOSE: Create a function that retrieves the following 3 command line inputs
#          from the user using the Argparse Python module. If the user fails to
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image Folder as --dir with default value 'pet_images'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#
##
# Imports python modules
import argparse

# TODO 1: Define get_input_args function below please be certain to replace None
#       in the return statement with parser.parse_args() parsed argument
#       collection that you created with this function
#
def get_input_args():
  """
  Retrieves and parses the 3 command line arguments provided by the user when
  they run the program from a terminal window. This function uses Python's
  argparse module to created and defined these 3 command line arguments. If
  the user fails to provide some or all of the 3 arguments, then the default
  values are used for the missing arguments.
  Command Line Arguments:
    1. Image Folder as --dir with default value 'pet_images'
    2. CNN Model Architecture as --arch with default value 'vgg'
    3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
  This function returns these arguments as an ArgumentParser object.
  Parameters:
    None - simply using argparse module to create & store command line arguments
  Returns:
    parse_args() -data structure that stores the command line arguments object
  """
  # Replace None with parser.parse_args() parsed argument collection that
  # you created with this function
  parser = argparse.ArgumentParser(description='Help')
  parser.add_argument('--dir', type=str, default='pet_images/')
  parser.add_argument('--arch', default='vgg')
  parser.add_argument('--dogfile', default='dognames.txt')

  return parser

def get_input_args_for_training():
  """
  Retrieves and parses the 3 command line arguments provided by the user when
  they run the program from a terminal window. This function uses Python's
  argparse module to created and defined these 3 command line arguments. If
  the user fails to provide some or all of the 3 arguments, then the default
  values are used for the missing arguments.
  Command Line Arguments:
    1. Image Folder as --dir with default value 'pet_images'
    2. CNN Model Architecture as --arch with default value 'vgg'
    3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
  This function returns these arguments as an ArgumentParser object.
  Parameters:
    None - simply using argparse module to create & store command line arguments
  Returns:
    parse_args() -data structure that stores the command line arguments object
  """

  '''
  Commandline arguments
  arch = pick architecture structure
  device = cpu or gpu
  checkpoint = path to checkpoint

  learning_rate = set learining rate
  n_hidden_units = i'm assuming it the hidden layers,
  epochs = number of epochs to train with

  save_location = location to save training data
  images_path = path to images
  idx_to_names = json that has index to names dictionary
 '''

  # Replace None with parser.parse_args() parsed argument collection that
  # you created with this function

  parser = argparse.ArgumentParser(description='Help')

  parser.add_argument('data_dir', type=str)
  parser.add_argument('--arch', default='resnet101', type=str)
  parser.add_argument('--device', default='cpu', type=str, choices=['cpu', 'cuda'])
  parser.add_argument('--checkpoint', default=None)

  parser.add_argument('--learning_rate', default=0.003, help="Should be a string with each layer sharing before and after, IE 3 layers would have '5, 3, 2, 6'") #
  parser.add_argument('--dropout', default=0.02)
  parser.add_argument('--hidden_units', type=str, default='512')
  parser.add_argument('--epochs', default=1)

  return parser


def get_input_args_for_predict():
  """
  Retrieves and parses the 3 command line arguments provided by the user when
  they run the program from a terminal window. This function uses Python's
  argparse module to created and defined these 3 command line arguments. If
  the user fails to provide some or all of the 3 arguments, then the default
  values are used for the missing arguments.
  Command Line Arguments:
    1. Image Folder as --dir with default value 'pet_images'
    2. CNN Model Architecture as --arch with default value 'vgg'
    3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
  This function returns these arguments as an ArgumentParser object.
  Parameters:
    None - simply using argparse module to create & store command line arguments
  Returns:
    parse_args() -data structure that stores the command line arguments object
  """

  '''
  Commandline arguments
 '''

  # Replace None with parser.parse_args() parsed argument collection that
  # you created with this function
  parser = argparse.ArgumentParser(description='Help')

  parser.add_argument('--device', default='cpu')
  parser.add_argument('--checkpoint')

  parser.add_argument('--image_path', type=str)
  parser.add_argument('--class_values', default=None)
  parser.add_argument('--topk', default=1, type=int)

  return parser

if __name__ == '__main__':
  input_args = get_input_args()
  args = input_args.parse_args('--dir x y --arch 1 2, --dogfile "test"'.split())
  # input_args.parse_args()
  print('main', args)
  pass

