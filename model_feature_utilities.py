from torchvision import models

def get_in_features_from_model(model):
  key, modules = get_last_module(model)

  in_features = getattr(modules, 'in_features', None)
  in_channels = getattr(modules, 'in_channels', None)

  if in_features:
    return key, in_features
  elif in_channels:
    return key, in_channels

  for mod_key, module in modules._modules.items():
    in_features = getattr(module, 'in_features', None)
    in_channels = getattr(module, 'in_channels', None)
    if in_features:
      return key, in_features
    elif in_channels:
      return key, in_channels

  raise "Can not find in_features or in_channels for that model"


def replace_feature_in_model_(model, key, module):
  modules = model._modules._modules
  modules[key] = module

  return model


def get_last_module(model):
  last_modules = None
  for module in model._modules.items():
    last_modules = module

  return last_modules