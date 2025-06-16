from collections import OrderedDict
import torch.nn as nn

def insert_layer(model, layer_name, new_layer, position='after'):
    new_layers = OrderedDict()
    for name, layer in model._modules.items():
        if position == 'before' and name == layer_name:
            new_layers['injected_' + name] = new_layer
        new_layers[name] = layer
        if position == 'after' and name == layer_name:
            new_layers['injected_' + name] = new_layer
    return nn.Sequential(new_layers)


model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(3, 16, 3, padding=1)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(16, 32, 3, padding=1)),
]))

# Insert BatchNorm after 'conv1'
new_model = insert_layer(model, 'conv1', nn.BatchNorm2d(16), position='after')
print(new_model)
