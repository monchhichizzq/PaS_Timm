import torch
import torch.nn as nn
import torch.nn.functional as F
from core_sparse.layers.spa_block import SparseConv2d
from core_sparse.layers.spa_block import SparseLinear
from core_sparse.layers.spa_block import SparseConvTranspose2d
from core_sparse.layers.prune_block import PaSAct

def insert_sparse_modules(network, inplace=False):
    """
    Recursively traverse the network and replace Conv2d layers with SparseConv2d layers.
    """
    for module_name, module in network.named_modules():
        if isinstance(module, nn.Conv2d):
            # Replace Conv2d with SparseConv2d
            new_module = SparseConv2d(module, threshold=2**-25, inplace=inplace)
            _swap_module(module_name=module_name, network=network, new_module=new_module)
            print(f"""\t - \"{module_name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")

        if isinstance(module, nn.ConvTranspose2d):
            # Replace ConvTranspose2d with SparseConvTranspose2d
            new_module = SparseConvTranspose2d(module, threshold=2**-25, inplace=inplace)
            _swap_module(module_name=module_name, network=network, new_module=new_module)
            print(f"""\t - \"{module_name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")
        
        if isinstance(module, nn.Linear):   
            # Replace Linear with SparseLinear
            new_module = SparseLinear(module, threshold=2**-25, inplace=inplace)
            _swap_module(module_name=module_name, network=network, new_module=new_module)
            print(f"""\t - \"{module_name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")

def insert_pas_modules(network, inplace=False):
    """
    Recursively traverse the network and replace Conv2d layers with SparseConv2d layers.
    """
    in_ch_list = []
    for module_name, module in network.named_modules():
        if isinstance(module, nn.Conv2d):
            # Replace Conv2d with SparseConv2d
            new_module = SparseConv2d(module, threshold=2**-25, inplace=inplace)
            _swap_module(module_name=module_name, network=network, new_module=new_module)
            print(f"""\t - \"{module_name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")
            
            # hack_in (resnet)
            ou_ch = module.out_channels
            in_ch_list.append(ou_ch)

        if isinstance(module, nn.ConvTranspose2d):
            # Replace ConvTranspose2d with SparseConvTranspose2d
            new_module = SparseConvTranspose2d(module, threshold=2**-25, inplace=inplace)
            _swap_module(module_name=module_name, network=network, new_module=new_module)
            print(f"""\t - \"{module_name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")
        
        if isinstance(module, nn.Linear):   
            # Replace Linear with SparseLinear
            new_module = SparseLinear(module, threshold=2**-25, inplace=inplace)
            _swap_module(module_name=module_name, network=network, new_module=new_module)
            print(f"""\t - \"{module_name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")

        if isinstance(module, nn.ReLU):
            # Replace ReLU with PaSAct
            # (for resnet, skip act1)
            if module_name != "act1":
                new_module = PaSAct(module, in_chs = in_ch_list[-1])
                _swap_module(module_name=module_name, network=network, new_module=new_module)
                print(f"""\t - \"{module_name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")

        if isinstance(module, nn.MaxPool2d):
            # Replace ReLU with PaSAct
            new_module = PaSAct(module, in_chs = in_ch_list[-1])
            _swap_module(module_name=module_name, network=network, new_module=new_module)
            print(f"""\t - \"{module_name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")


def _swap_module(module_name, network, new_module):
    """
    Swap the specified module in the network with a new module.
    """
    # Nested modueles are identified 
    module_names = module_name.split('.')
    # Outer_module_names is an empty list if the module is not nested
    outer_module_names = module_names[:-1]
    inner_module_name = module_names[-1]
    # Traverse the network to find the module
    module = network
    for outer_name in outer_module_names:
        module = getattr(module, outer_name)
  
    # Swap in the new module
    setattr(module, inner_module_name, new_module)


def replace_gelu_to_relu(network):
    for name, module in network.named_modules():
        if isinstance(module, nn.GELU):
            new_module = nn.ReLU(inplace=False)
            _swap_module(name, network, new_module)
            print(f"""\t - \"{name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")
        else:
            print(f"""\t - \"{name}\": {module.__class__.__name__} -> """)


def replace_hswish_to_relu(network):
    for name, module in network.named_modules():
        if "backbone" in name:
            if isinstance(module, nn.Hardswish):
                new_module = nn.ReLU(inplace=False)
                _swap_module(name, network, new_module)
                print(f"""\t - \"{name}\": {module.__class__.__name__} -> {new_module.__class__.__name__}""")
            else:
                print(f"""\t - \"{name}\": {module.__class__.__name__} -> """)