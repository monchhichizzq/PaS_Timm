import torch
from core_sparse.layers.prune_staic_layer import BinaryConv2d

class Convergence(object):
    """Computes and stores the average and current value"""

    def __init__(self, model):
        self.channels = torch.tensor([])
        for name, module in model.named_modules():
            # if 'scale' in name or isinstance(module, BinaryConv2d):
            if isinstance(module, BinaryConv2d):
                w = module.weight.detach().cpu()
                binary_w = (w > 0.5).float()
                self.channels = torch.cat((self.channels, torch.sum(torch.squeeze(binary_w), dim=0, keepdim=True)), dim=0)
                # print(name, torch.sum(torch.squeeze(binary_w)))
        self.channels = self.channels.reshape(1, self.channels.size(0))
        print("self.channels: ", self.channels.shape) # (1, 49)

    def update(self, model):
        channel_list = torch.tensor([])
        # for name, module in model.module.named_modules():
        for name, module in model.named_modules():
            # if 'scale' in name or isinstance(module, BinaryConv2d):
            if isinstance(module, BinaryConv2d):
                w = module.weight.detach().cpu()
                binary_w = (w > 0.5).float()
                channel_list = torch.cat((channel_list, torch.sum(torch.squeeze(binary_w), dim=0, keepdim=True)), dim=0)
        channel_list = channel_list.reshape(1, channel_list.size(0))
        self.channels = torch.cat((self.channels, channel_list), dim=0)
        # print("channel_list:", channel_list)

    def save(self, config_path='convergence.pt'):
        print("(update) self.channels: ", self.channels.shape, self.channels[-1])
        torch.save(self.channels, config_path)


# def apply_pas_loss(model, prune_ratio=0.5, pas_coeff=5, arch="resnets"):
#     """
#     Apply PaS loss to the model.
#     Args:
#         model: The model to apply PaS loss.
#         prune_ratio: The ratio of channels to prune.
#         pas_coeff: The coefficient for PaS loss.
#         arch: The architecture of the model.
#     Returns:
#         The PaS loss value.
#     """
#     if arch == "resnets":
#         Branches = torch.tensor([]).cuda()

#         for name, module in model.named_modules():
#             if isinstance(module, BinaryConv2d):
#                 w = module.weight.detach()
#                 binary_w = (w > 0.5).float()
#                 residual = w - binary_w
#                 branch_out = module.weight - residual
#                 Branches = torch.cat((Branches, torch.sum(torch.squeeze(branch_out), dim=0, keepdim=True)), dim=0)
        
#     if arch == "resnets":
#         return pas_coeff * torch.sum(convergence.channels[-1] * (1 - prune_ratio))
#     else:
#         raise NotImplementedError(f"Architecture {arch} is not implemented.")