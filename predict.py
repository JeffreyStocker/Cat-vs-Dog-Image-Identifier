'''
    Predicting classes 	      The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
    Top K classes 	          The predict.py script allows users to print out the top K classes along with associated probabilities
    Displaying class names 	  The predict.py script allows users to load a JSON file that maps the class values to other category names
    Predicting with GPU 	    The predict.py script allows users to use the GPU to calculate the predictions
 '''
import torch
import model_validate
import model_import
import transforms

from PIL import Image
import json

from torch import nn, optim
from torch.utils.data import DataLoader
import datetime
from pathlib import Path

from get_input_args import get_input_args_for_predict


start_time = datetime.datetime.now()

is_cuda_available = torch.cuda.is_available

arguments = get_input_args_for_predict().parse_args()

checkpoint = arguments.checkpoint
image_path = arguments.image_path
class_values = arguments.category_names
topk = arguments.topk

#check if device is available
device = 'cuda' if arguments.gpu else 'cpu'
if device =='cuda' and not is_cuda_available():
  raise 'There is no CUDA on this computer'

if class_values:
  with open(class_values, 'r') as f:
      class_to_names = json.load(f)
else:
  class_to_names = None

model, save_data, idx_to_class = model_import.load_universal_model(checkpoint)


#evaluate image with no gradients o& model in eval
model.eval()
with torch.no_grad():
  model.to(device)
  pil_image = Image.open(image_path)
  tensor_image = transforms.data_transforms(pil_image)
  tensor_image.unsqueeze_(0)
  log_prob = model(tensor_image)

prob_topk = torch.exp(log_prob).topk(topk)

# print(prob_topk.items())
output = []

for prob, idx in zip(prob_topk[0][0].numpy(), prob_topk[1][0].numpy()):
  class_idx = idx_to_class.get(str(idx))
  ind_prob_info = [prob, idx, class_idx]

  if class_to_names:
    name = class_to_names.get(str(class_idx))
    ind_prob_info.append(name)

  output.append(ind_prob_info)


#print names
for names_probs in output:
  print(names_probs[-1], names_probs[0])
