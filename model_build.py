from torchvision import models
from torch import nn

def build_universal_model(model_name, layers, layer_target_name = 'classifier', dropout=0.2, initial_epoch = 0):
    model = getattr(models, model_name)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    number_of_inputs = layers[0]
    number_of_Classifications = layers[-1]

    layer_count = zip(layers[:-2], layers[1:-1])

    build_sequence = []

    for lower, upper in layer_count:
        build_sequence.append(nn.Linear(lower, upper))
        build_sequence.append(nn.ReLU())
        build_sequence.append(nn.Dropout(p=dropout))
    else:
        build_sequence.append(nn.Linear(layers[-2], layers[-1]))
        build_sequence.append(nn.LogSoftmax(dim=1))

    classifier = nn.Sequential(*build_sequence)

    save_data = {
        "layers": layers,
        "model_name": model_name,
        'layer_target_name': layer_target_name,
        'dropout': dropout,
        'epochs': initial_epoch
    }

    setattr(model, layer_target_name, classifier)

    return model, save_data