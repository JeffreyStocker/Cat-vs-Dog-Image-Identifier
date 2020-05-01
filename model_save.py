import torch
import os
import datetime

time = datetime.datetime.now()

def save_universal_model(model, save_data, epochs = None, extra_text=None, include_epoch_in_title=None):
    checkpoint = save_data
    checkpoint["state_dict"] = model.state_dict()

    if epochs:
        if epochs: checkpoint["epochs"] = epochs

    file_name = f'{save_data["model_name"]}_checkpoints_{time}{f"_e{epochs}" if include_epoch_in_title else ""}{"-" + extra_text if extra_text else ""}.pth'.replace(":", "-")

    try:
        torch.save(checkpoint, 'checkpoint/' + file_name)
    except:
        os.mkdir('checkpoint')
        try:
            torch.save(checkpoint, 'checkpoint/' + file_name)
        except:
            torch.save(checkpoint, file_name)
            print('unable to save')
