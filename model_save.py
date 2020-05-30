import torch
import os
import datetime
from os import path

time = datetime.datetime.now()
save_dir = 'checkpoint/'

def set_path(newPath):
    if path.isdir(newPath):
        save_dir = newPath
    else:
        raise "Save path is not valid directory. Create it or double check your path"

def save_universal_model(model, save_data, epochs = None):
    checkpoint = save_data
    checkpoint["state_dict"] = model.state_dict()

    if epochs:
        checkpoint["epochs"] = epochs

    file_name = f'{save_data["model_name"]}_checkpoints_{time}.pth'.replace(":", "-").replace(" ", "_")
    file_path =  path.join(save_dir, file_name)

    try:
        torch.save(checkpoint, file_path)
    except:
        os.mkdir('checkpoint')
        try:
            torch.save(checkpoint, 'checkpoint/' + file_name)
        except:
            torch.save(checkpoint, file_name)
            print('unable to save')
