import torch
import torch.nn as nn
import numpy as np
from core_sparse.layers.spa_layer import Input_Placeholder
from core_sparse.layers.spa_block import SparseConv2d
from core_sparse.layers.prune_layer import BinaryConv2d


def induce_sparsity(x, star_coeff=0.0, thresh=-23, loss_mode='l1'):
    """
    Applies regularization to feature maps (x) based on the specified loss mode.

    Args:
        x (torch.Tensor): Input tensor.
        alpha (float): Regularization strength.
        thresh (float): Threshold used for 'partial_l1' mode (interpreted as power of 2).
        loss_mode (str): Type of regularization ('l1', 'partial_l1', 'count', 'hoyer').

    Returns:
        torch.Tensor: Computed regularization loss.
    """
    axis_ = 1 if x.dim() < 4 else [1, 2, 3]

    if loss_mode == 'partial_l1':
        float_thresh = torch.pow(torch.tensor(2.0, device=x.device), thresh)
        mask = torch.abs(x) < float_thresh
        partial_values = torch.abs(x) * mask
        loss = torch.mean(torch.sum(partial_values, dim=axis_)) * star_coeff

    elif loss_mode == 'l1':
        loss = torch.mean(torch.sum(torch.abs(x), dim=axis_)) * star_coeff

    elif loss_mode == 'count':
        spike_approx = torch.sigmoid(100 * x)
        loss = torch.mean(torch.sum(spike_approx, dim=axis_)) * star_coeff

    elif loss_mode == 'hoyer':
        abs_sum = torch.sum(torch.abs(x), dim=axis_)
        sq_sum = torch.sum(x ** 2, dim=axis_)
        # Prevent division by zero
        epsilon = 1e-8
        ratio = (abs_sum ** 2) / (sq_sum + epsilon)
        loss = torch.mean(ratio) * star_coeff

    else:
        print(f"[Warning] Loss mode '{loss_mode}' is not supported.")
        loss = torch.tensor(0.0, device=x.device)

    return loss


def apply_opt_loss(model, star_coeff=0, pas_coeff=0, loss_mode="l1"):
    """
    Apply optimization loss to the model.
    Args:
        model: The model to apply optimization loss.
        star_coeff: The coefficient for Star loss.
        pas_coeff: The coefficient for PaS loss.
        loss_mode: The mode of the loss function ('l1' or 'l2').
    Returns:
        The optimization loss value.
    """
    total_loss = 0

    if star_coeff > 0:
        for name, module in model.named_modules():
            if isinstance(module, Input_Placeholder):
                sparse_acts = module.sig_x
                reg_loss = induce_sparsity(sparse_acts, star_coeff, -23, loss_mode)
                total_loss += reg_loss

    return total_loss




def apply_pas_loss(model, prune_ratio=0.0, pas_coeff=5, arch="resnets"):
    """
    Apply PaS loss to the model.
    Args:
        model: The model to apply PaS loss.
        prune_ratio: The ratio of channels to prune.
        pas_coeff: The coefficient for PaS loss.
        arch: The architecture of the model.
    Returns:
        The PaS loss value.
    """

    total_dense_macs = 0
    if arch == "resnets":
        layer_mac_dict = {}
        layer_mac_list = []
        for name, module in model.named_modules():
            if isinstance(module, SparseConv2d): # notice input size
                feature_shape = module.input_shape
                sigma_evts = module.sigma_events
                spati_evts = module.spati_events
                dense_evts = module.dense_events
                in_feat = module.sig_x # batch, in_chs, h, w

                c_out, c_in, k_h, k_w, s, g, params = get_conv_params(module)
                _, h_in, w_in = module.input_shape
         
                c_out, h_out, w_out = c_out, h_in // s, w_in // s
                
                if g != 1:
                    dense_macs = h_out * w_out * c_out * c_in * k_h * k_w / g
                else:
                    dense_macs = h_out * w_out * c_out * c_in * k_h * k_w

                if g != 1:
                    dense_params = h_out * w_out * k_h * k_w / g
                else:
                    dense_params = h_out * w_out * k_h * k_w
                
                axis_ = 0 if in_feat.dim() < 4 else [0, 2, 3]

                if len(in_feat.shape) < 4:
                    in_chs = torch.sum(in_feat, dim=axis_)
                else:
                    in_chs = torch.sum(in_feat, dim=axis_)

                binary_chs = (in_chs > 0.0).float()
                c_in_act = torch.sum(binary_chs)
  
                # if isinstance(c_in_act, torch.Tensor):
                #     c_in_act = c_in_act.cpu().numpy()

                total_dense_macs += dense_macs
 
                # important
                if "downsample" not in name:
                    layer_mac_dict[name] = dense_params
                    layer_mac_list.append(float(dense_params))
          
        in_chs_list = torch.tensor([]).cuda()

        for name, module in model.named_modules():
            if isinstance(module, BinaryConv2d):
                w = module.weight.detach()
                binary_w = (w > 0.5).float()
                residual = w - binary_w
                branch_out = module.weight - residual

                # skipping relu before maxpooling
                if name != "act1":
                # if name != "act1.binary_conv":
                    in_chs_list = torch.cat((in_chs_list, torch.sum(torch.squeeze(branch_out), dim=0, keepdim=True)), dim=0)

        # add input channel 3
        ou_chs_list = in_chs_list
        in_chs_list = torch.cat((torch.tensor([3]).cuda(), in_chs_list[:-1]), dim=0)

        total_prune_macs = torch.sum(torch.tensor(in_chs_list) * torch.tensor(layer_mac_list).cuda() * ou_chs_list)

    else:
        raise NotImplementedError(f"Architecture {arch} is not implemented.")

    criterion = nn.MSELoss()
    total_dense_macs = total_dense_macs / 1e9
    target_macs = total_dense_macs * (1 - prune_ratio)
    target_macs = torch.tensor(target_macs, dtype=torch.float32).cuda()
    total_prune_macs = total_prune_macs / 1e9
    prune_loss = criterion(total_prune_macs, target_macs)
    pas_loss = pas_coeff * prune_loss
    # print(total_prune_macs, target_macs, pas_loss, prune_ratio)
    # print("")
    return pas_loss, total_dense_macs, total_prune_macs, target_macs


def get_conv_params(module):
    """
    Get the number of parameters for each layer in the module.
    """
    g, s = module.groups, module.stride[0]
    params = np.prod(module.weight.shape)

    # TODO : Add support for groups
    c_out, c_in, k_h, k_w = module.weight.shape

    if g != 1 and c_in == 1:
        c_in = c_out
    
    # print(f"groups: {g}, c_out: {c_out}, c_in: {c_in}, k_h: {k_h}, k_w: {k_w}")

    return c_out, c_in, k_h, k_w, s, g, params
