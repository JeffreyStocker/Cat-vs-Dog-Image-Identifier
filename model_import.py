import torch
from model_build import build_universal_model
from convert_class_to_idx import convert_class_to_idx

def load_universal_model(filename, map_location='cpu', dropout = 0):
  # def build_universal_model(model_name, layers, layer_target_name = 'classifier', dropout=0.2):
    save_data = torch.load(filename, map_location=map_location)
    model, model_save_data = build_universal_model(
                                            save_data["model_name"],
                                            save_data["layers"],
                                            save_data["layer_target_name"],
                                            dropout if dropout else save_data["dropout"])

    model.load_state_dict(save_data["state_dict"])

    idx_to_class = save_data.get("idx_to_class")
    class_to_idx = save_data.get("class_to_idx") #left over from previous runs

    if not idx_to_class and class_to_idx:
      idx_to_class = convert_class_to_idx(class_to_idx)
      save_data['idx_to_class'] = idx_to_class
      save_data.pop('class_to_idx')

    save_data['dropout'] = dropout

    print('Loaded: ', filename)
    return model, save_data, idx_to_class