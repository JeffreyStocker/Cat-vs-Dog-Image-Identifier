import torch
from model_build import build_universal_model
from convert_class_to_idx import convert_class_to_idx

def load_universal_model(filename, map_location='cpu', dropout = None):
  # def build_universal_model(model_name, layers, layer_target_name = 'classifier', dropout=0.2):
    data = torch.load(filename, map_location=map_location)
    model, model_save_data = build_universal_model(
                                            data["model_name"],
                                            data["layers"],
                                            data["layer_target_name"],
                                            dropout if dropout else data["dropout"])

    model.load_state_dict(data["state_dict"])

    idx_to_class = data.get("idx_to_class")

    print('Loaded: ', filename)
    return model, data, idx_to_class