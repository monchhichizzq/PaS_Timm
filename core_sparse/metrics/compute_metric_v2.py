import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from core_sparse.layers.spa_layer import Input_Placeholder
from timm.models.vision_transformer import Attention

# color codes
R = "\033[31m" # Red
G = "\033[32m" # Green
B = "\033[34m" # Blue
Y = "\033[33m" # Yellow
C = "\033[36m" # Cyan
M = "\033[35m" # Magenta
BOLD = "\033[1m" # Bold
UNDERLINE = "\033[4m" # Underline
N = "\033[0m" # Reset color


def create_layer_metric_dict(model):

    avg_metric_dict = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

            if isinstance(m, nn.LayerNorm):
                original_name = name.replace(".norm_layer", "")
            else:
                original_name = name

            if "act" not in original_name and "maxpool" not in original_name: # skip binary dw (resnet act)
                avg_metric_dict[original_name] = {}
                avg_metric_dict[original_name]['sigma evts'] = []
                avg_metric_dict[original_name]['spati evts'] = []
                avg_metric_dict[original_name]['delta evts'] = []
                avg_metric_dict[original_name]['dense evts'] = []

                avg_metric_dict[original_name]['sigma macs'] = []
                avg_metric_dict[original_name]['spati macs'] = []
                avg_metric_dict[original_name]['delta macs'] = []
                avg_metric_dict[original_name]['dense macs'] = []

                avg_metric_dict[original_name]['sigma cycs'] = []
                avg_metric_dict[original_name]['spati cycs'] = []
                avg_metric_dict[original_name]['delta cycs'] = []
                avg_metric_dict[original_name]['dense cycs'] = []

    return avg_metric_dict


def calculate_events(feature):
    dyn_evts = torch.mean((feature != 0).float(), dim=0)
    zeo_evts = torch.mean((feature == 0).float(), dim=0)
    sta_evts = dyn_evts + zeo_evts

    batch_dyn_evts = torch.sum(dyn_evts)
    batch_zeo_evts = torch.sum(zeo_evts)
    batch_sta_evts = torch.sum(sta_evts)
    return batch_dyn_evts, batch_sta_evts


def calculate_macs(net):
    op_infos = {}
    for name, module in net.named_modules():
        if isinstance(module, Input_Placeholder):
            feature_shape = module.input_shape
            sigma_evts = module.sigma_events
            spati_evts = module.spati_events
            dense_evts = module.dense_events

            float_thr  = module.threshold
            trans_policy = module.trans_policy

            _data_list = [feature_shape, sigma_evts, spati_evts, dense_evts, float_thr, trans_policy]
            op_infos[name] = _data_list
    return op_infos


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

def get_linear_params(module):
    """
    Get the number of parameters for each layer in the module.
    """
    params = np.prod(module.weight.shape)
    c_out, c_in = module.weight.shape
    return c_out, c_in, 1, 1, 1, 1, params

def get_layernorm_params(module):
    """
    Get the number of parameters for each layer in the module.
    """
    params = np.prod(module.weight.shape)
    c_out = c_in = module.weight.shape[0]
    return c_out, c_in, 1, 1, 1, 1, params


def export_thr_exp(thr):
    p2_thr = np.log(thr + 1e-7) / np.log(2)
    p2_thr = np.round(p2_thr)
    return p2_thr


def modify_name(name):
    # modify the name
    ori_name = name
    new_name = name.replace("module.", "")
    new_name = new_name.replace("backbone.", "")
    return ori_name, new_name


def measure_network_compute(net, arch, avg_metric_dit, xlsx_file_path=None, verbose=False):
    dw_scale = 5
    op_infs = calculate_macs(net)

    # Create a PrettyTable object
    _result_table = PrettyTable()
    _result_table.field_names = [
        "name", "policy", "thr", 
        "sig-evt", "spa-evt", "del-evt", "den-evt",
        "sig-mac", "spa-mac", "del-mac", "den-mac",
        "sig-cyc", "spa-cyc", "del-cyc", "den-cyc",
        "sig-den", "spa-den", "del-den", "params", "height"
    ]
    _result_table.align = "l"  # Align to the left
    _result_table.padding_width = 1  # One space between column edges and contents
    _result_table.float_format = ".2f"  # Set float format to 2 decimal places

    # Create excel
    excel_col_names = [
        "name", "policy", "thr",
        "sig-evt", "spa-evt", "del-evt", "den-evt",
        "sig-mac", "spa-mac", "del-mac", "den-mac",
        "sig-den", "spa-den", "del-den", "params"
    ]

    excel_layer_names = []
    excel_layer_content = []

    # Initialize the layerwise data
    total_sigma_evts = 0
    total_spati_evts = 0
    total_delta_evts = 0
    total_dense_evts = 0

    total_sigma_macs = 0
    total_spati_macs = 0
    total_delta_macs = 0
    total_dense_macs = 0

    total_sigma_cycs = 0
    total_spati_cycs = 0
    total_delta_macs = 0
    total_dense_cycs = 0

    total_sigma_delta_evts = 0
    total_sigma_delta_macs = 0
    total_sigma_spati_evts = 0
    total_sigma_spati_macs = 0

    total_params = 0

    # Specific network metrics
    # fusev*
    if arch == "fuse":
        total_exp_sigma_macs = 0
        total_con_sigma_macs = 0
        total_exp_dense_macs = 1e-7
        total_con_dense_macs = 1e-7

        total_exp_sigma_cycs = 0
        total_con_sigma_cycs = 0
        total_exp_dense_cycs = 1e-7
        total_con_dense_cycs = 1e-7

    # transformer
    if arch == "transformer":
        total_fc1_sigma_macs = 0
        total_fc2_sigma_macs = 0
        total_fc1_dense_macs = 0
        total_fc2_dense_macs = 0
        total_qkv_sigma_macs = 0
        total_qkv_dense_macs = 0
        total_pro_sigma_macs = 0
        total_pro_dense_macs = 0
        total_att_sigma_macs = 0
        total_att_dense_macs = 0

    for name, m in net.named_modules():

        # self-attention in transformer
        if isinstance(m, Attention):
            # get macs
            qk_dyn_macs = m.qk_dyn_macs.cpu().numpy()
            qk_sta_macs = m.qk_sta_macs.cpu().numpy()
            kv_dyn_macs = m.kv_dyn_macs.cpu().numpy()
            kv_sta_macs = m.kv_sta_macs.cpu().numpy()

            sigma_evts, spati_evts, dense_evts = 0, 0, 0

            sigma_macs = (qk_dyn_macs + kv_dyn_macs)
            dense_macs = (qk_sta_macs + kv_sta_macs)

            mean_sigma_density = sigma_macs / dense_macs

            # get thr
            qkv_thr = m.qkv_thr
            att_thr = m.att_thr

            # fullfill _result_table
            _result_table.add_row([
                R + name.replace(".qkv", "") + N,
                "{}, {}".format(qkv_thr, att_thr),
                "",  # mean sigma evts
                "",  # mean spati evts
                "/", # mean delta evts
                "{:.2f}".format( 0 / 10**3), # mean dense evts
                "{:.2f}".format(sigma_macs / 10**6), # mean sigma macs
                "",  # mean spati macs
                "/", # mean delta macs
                "{:.2f}".format(dense_macs / 10**6), # mean dense macs
                np.round(mean_sigma_density, 2), # mean sigma density
                "", # mean spati density
                "/", # mean delta density
                "", # params
                "", # h_in
            ])

            # qkv macs
            total_sigma_evts += 0
            total_dense_evts += 0

            total_sigma_macs += sigma_macs
            total_dense_macs += dense_macs

            # attention macs
            if arch == "transformer":
                total_att_sigma_macs += sigma_macs
                total_att_dense_macs += dense_macs

        if (
            isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d, nn.LayerNorm))
            and "act" not in name
            and "maxpool" not in name # important change (resnets)
            ):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): # skip binary dw:
                c_out, c_in, k_h, k_w, s, g, params = get_conv_params(m)
                # feature_shape, sigma_evts, spati_evts, dense_evts, float_thr, trans_policy
                fm_shape, sigma_evts, spati_evts, dense_evts, threshold, trans_policy = op_infs[name]
                c_in, h_in, w_in = fm_shape
                dense_evts = c_in * h_in * w_in
                seq_len = 1

                if isinstance(m, nn.ConvTranspose2d):
                    c_out, h_out, w_out = c_out, h_in * s, w_in * s
                else:
                    c_out, h_out, w_out = c_out, h_in // s, w_in // s

                if g != 1:
                    dense_macs = h_out * w_out * c_out * c_in * k_h * k_w / g
                else:
                    dense_macs = h_out * w_out * c_out * c_in * k_h * k_w
                
                # scale mac for compute
                num_per_g = c_in // g
                # print(name, num_per_g, c_in, g)
                dw_factor = 1 if (num_per_g >= dw_scale) or (g == 1) else dw_scale
                # dw_factor = 1 if g == 1 else dw_scale
                dense_cycs = dense_macs * dw_factor
             
            if isinstance(m, nn.Linear):
                c_out, c_in, k_h, k_w, s, g, params = get_linear_params(m)
                fm_shape, sigma_evts, spati_evts, dense_evts, threshold, trans_policy = op_infs[name]

                if len(fm_shape) == 1:
                    seq_len = 1
                    model_dim = fm_shape
                    dense_evts = model_dim[0]
                elif len(fm_shape) == 2:
                    seq_len, model_dim = fm_shape
                    dense_evts = np.prod(fm_shape)

                c_out, h_out, w_out = c_out, 1, 1

                dense_macs = c_in * c_out * seq_len
                dense_cycs = dense_macs
            
            if isinstance(m, nn.LayerNorm):
                name = name.replace(".norm_layer", "")
                c_out, c_in, k_h, k_w, s, g, params = get_layernorm_params(m)
                fm_shape, sigma_evts, spati_evts, dense_evts, threshold, trans_policy = op_infs[name]

                dense_evts = dense_evts.cpu().numpy()
                dense_macs = 4 * params
                dense_cycs = dense_macs * dw_scale

            # sum up num of parameters
            total_params += params / 10**6

            # convert torch tensor to float
            # if not isinstance(sigma_evts, float):
            if isinstance(sigma_evts, torch.Tensor):
                sigma_evts = sigma_evts.cpu().numpy()

            # if not isinstance(spati_evts, float):
            if isinstance(spati_evts, torch.Tensor):
                spati_evts = spati_evts.cpu().numpy()

            # calculate mean
            sigma_evts = sigma_evts / 10**6
            spati_evts = spati_evts / 10**6
            dense_evts = dense_evts / 10**6
            sigma_density = sigma_evts / dense_evts
            spati_density = spati_evts / dense_evts

            # get dense macs
            dense_macs = dense_macs / 10**6
            m.static_macs = dense_macs
            sigma_macs = sigma_density * dense_macs
            spati_macs = spati_density * dense_macs

            # get dense cycs
            dense_cycs = dense_cycs / 10**6
            sigma_cycs = sigma_density * dense_cycs
            spati_cycs = spati_density * dense_cycs

            # get macs based on layer type
            if arch == "transformer":
                if "fc2" in name:
                    total_fc2_sigma_macs += sigma_macs
                    total_fc2_dense_macs += dense_macs
                if "fc1" in name:
                    total_fc1_sigma_macs += sigma_macs
                    total_fc1_dense_macs += dense_macs
                if "qkv" in name:
                    total_qkv_sigma_macs += sigma_macs
                    total_qkv_dense_macs += dense_macs
                if "pro" in name:
                    total_pro_sigma_macs += sigma_macs
                    total_pro_dense_macs += dense_macs

            if arch == "fuse":
                cond_1 = "conv_pw" in name and "conv_pwl" not in name
                cond_2 = "conv.0" in name
                cond_3 = not "dw_3x3" in name
                
                if cond_1 or cond_2: #or cond_3:
                    total_exp_sigma_macs += sigma_macs
                    total_exp_dense_macs += dense_macs

                    total_exp_sigma_cycs += sigma_cycs
                    total_exp_dense_cycs += dense_cycs
                else:
                    total_con_sigma_macs += sigma_macs
                    total_con_dense_macs += dense_macs

                    total_con_sigma_cycs += sigma_cycs
                    total_con_dense_cycs += dense_cycs
                
            # modify the name
            ori_name, new_name = modify_name(name)

            # get threshold
            p2_thr = export_thr_exp(threshold)

            # get density
            if sigma_density > spati_density:
                total_sigma_spati_evts += spati_evts
                total_sigma_spati_macs += spati_macs
            else:
                total_sigma_spati_evts += sigma_evts
                total_sigma_spati_macs += sigma_macs

            # append the data to dict
            avg_metric_dit[ori_name]['sigma evts'].append(sigma_evts / 10**0)
            avg_metric_dit[ori_name]['spati evts'].append(spati_evts / 10**0)
            avg_metric_dit[ori_name]['delta evts'].append(0)
            avg_metric_dit[ori_name]['dense evts'].append(dense_evts / 10**0)
            avg_metric_dit[ori_name]['sigma macs'].append(sigma_macs / 10**0)
            avg_metric_dit[ori_name]['spati macs'].append(spati_macs / 10**0)
            avg_metric_dit[ori_name]['delta macs'].append(0)
            avg_metric_dit[ori_name]['dense macs'].append(dense_macs / 10**0)
            avg_metric_dit[ori_name]['sigma cycs'].append(sigma_cycs / 10**0)
            avg_metric_dit[ori_name]['spati cycs'].append(spati_cycs / 10**0)
            avg_metric_dit[ori_name]['delta cycs'].append(0)
            avg_metric_dit[ori_name]['dense cycs'].append(dense_cycs / 10**0)

            # average list
            mean_sigma_evts = np.mean(avg_metric_dit[ori_name]['sigma evts'])
            mean_spati_evts = np.mean(avg_metric_dit[ori_name]['spati evts'])
            mean_dense_evts = np.mean(avg_metric_dit[ori_name]['dense evts'])
            mean_sigma_macs = np.mean(avg_metric_dit[ori_name]['sigma macs'])
            mean_spati_macs = np.mean(avg_metric_dit[ori_name]['spati macs'])
            mean_dense_macs = np.mean(avg_metric_dit[ori_name]['dense macs'])
            mean_sigma_cycs = np.mean(avg_metric_dit[ori_name]['sigma cycs'])
            mean_spati_cycs = np.mean(avg_metric_dit[ori_name]['spati cycs'])
            mean_dense_cycs = np.mean(avg_metric_dit[ori_name]['dense cycs'])

            std_sigma_evts = np.std(avg_metric_dit[ori_name]['sigma evts'])
            std_spati_evts = np.std(avg_metric_dit[ori_name]['spati evts'])
            std_dense_evts = np.std(avg_metric_dit[ori_name]['dense evts'])
            std_sigma_macs = np.std(avg_metric_dit[ori_name]['sigma macs'])
            std_spati_macs = np.std(avg_metric_dit[ori_name]['spati macs'])
            std_dense_macs = np.std(avg_metric_dit[ori_name]['dense macs'])
            std_sigma_cycs = np.std(avg_metric_dit[ori_name]['sigma cycs'])
            std_spati_cycs = np.std(avg_metric_dit[ori_name]['spati cycs'])
            std_dense_cycs = np.std(avg_metric_dit[ori_name]['dense cycs'])

            # calculate mean density
            mean_sigma_density = mean_sigma_macs / mean_dense_macs
            mean_spati_density = mean_spati_macs / mean_dense_macs

            std_sigma_density = std_sigma_macs / mean_dense_macs
            std_spati_density = std_spati_macs / mean_dense_macs

            # color code
            spati_color = B if mean_sigma_density > mean_spati_density else G

            # add row to table
            _result_table.add_row([
                new_name, 
                int(p2_thr), 
                trans_policy[:3],
                "{:.2f}".format(mean_sigma_evts),
                "{:.2f}".format(mean_spati_evts),
                "/", # mean delta evts
                "{:.2f}".format(mean_dense_evts),
                "{:.2f}".format(mean_sigma_macs),
                "{:.2f}".format(mean_spati_macs),
                "/", # mean delta macs
                "{:.2f}".format(mean_dense_macs),
                "{:.2f}".format(mean_sigma_cycs),
                "{:.2f}".format(mean_spati_cycs),
                "/", # mean delta cycs
                "{:.2f}".format(mean_dense_cycs),
                "{:.2f}".format(mean_sigma_density * 100),
                spati_color + "{:.2f}".format(mean_spati_density * 100) + N,
                "/", # mean delta density
                "{:.2f}".format(params / 10**6),
                h_in
            ])

            # add row to excel
            excel_layer_names.append(ori_name)
            excel_layer_content.append(
                [
                    trans_policy, int(p2_thr),
                    "{:.4f}".format(mean_sigma_evts),
                    "{:.4f}".format(mean_spati_evts),
                    0, # mean delta evts
                    "{:.4f}".format(mean_dense_evts),
                    "{:.4f}".format(mean_sigma_macs),
                    "{:.4f}".format(mean_spati_macs),
                    0, # mean delta macs
                    "{:.4f}".format(mean_dense_macs),
                    "{:.4f}".format(mean_sigma_density * 100),
                    "{:.4f}".format(mean_spati_density * 100),
                    0, # mean delta density
                    "{:.4f}".format(params / 10**3),
                    ""
                ])

            # sum up
            total_sigma_evts += sigma_evts
            total_spati_evts += spati_evts
            total_dense_evts += dense_evts
            total_sigma_macs += sigma_macs
            total_spati_macs += spati_macs
            total_dense_macs += dense_macs
            total_sigma_cycs += sigma_cycs
            total_spati_cycs += spati_cycs
            total_dense_cycs += dense_cycs

    # sum up the total
    total_sigma_evt = total_sigma_evts / total_dense_evts
    total_spati_evt = total_spati_evts / total_dense_evts
    total_sigma_spati_evt = total_sigma_spati_evts / total_dense_evts

    total_sigma_mac = total_sigma_macs / total_dense_macs
    total_spati_mac = total_spati_macs / total_dense_macs
    total_sigma_spati_mac = total_sigma_spati_macs / total_dense_macs

    total_sigma_cyc = total_sigma_cycs / total_dense_cycs
    total_spati_cyc = total_spati_cycs / total_dense_cycs

    # add row to table
    _result_table.add_row([
        "sum",
        "",
        "",
        "{:.2f}".format(total_sigma_evts),
        "{:.2f}".format(total_spati_evts),
        "", # mean delta evts
        "{:.2f}".format(total_dense_evts),
        "{:.2f}".format(total_sigma_macs),
        "{:.2f}".format(total_spati_macs),
        "", # mean delta macs
        "{:.2f}".format(total_dense_macs),
        "{:.2f}".format(total_sigma_cycs),
        "{:.2f}".format(total_spati_cycs),
        "", # mean delta cycs
        "{:.2f}".format(total_dense_cycs),
        "",
        "",
        "", # mean delta density
        "{:.2f}".format(total_params),
        "",
    ])

    _result_table.add_row([
        "density",
        "",
        "",
        "{:.2f}".format(total_sigma_evt * 100),
        "{:.2f}".format(total_spati_evt * 100),
        "", # mean delta evts
        "{:.2f}".format(100),
        "{:.2f}".format(total_sigma_mac * 100),
        "{:.2f}".format(total_spati_mac * 100),
        "", # mean delta macs
        "{:.2f}".format(100),
        "{:.2f}".format(total_sigma_cyc * 100),
        "{:.2f}".format(total_spati_cyc * 100),
        "", # mean delta cycs
        "{:.2f}".format(100),
        "",
        "",
        "", # mean delta density
        "",
        ""
    ])

    metrics = [
        total_sigma_evt * 100,
        total_spati_evt * 100,
        total_sigma_spati_evt * 100,
        total_sigma_mac * 100,
        total_spati_mac * 100,
        total_sigma_spati_mac * 100,
        total_sigma_cyc * 100,
        total_spati_cyc * 100,
    ]

    # add row to excel
    df = pd.DataFrame(
        excel_layer_content,
        columns=excel_col_names,
        index=excel_layer_names
    )
    
    if xlsx_file_path is not None:
        df.to_excel(xlsx_file_path, sheet_name=arch)
        print("Excel file saved to {}".format(xlsx_file_path))

    if verbose:
        print(_result_table)

    if arch == "transformer":
        _mac_dist_table = PrettyTable()
        _mac_dist_table.field_names = [
            "name", "sigma macs", "dense macs", "density"
        ]
        _mac_dist_table.add_row(
            ["qkv", total_qkv_sigma_macs / 10**3, total_qkv_dense_macs / 10**3, np.round(total_qkv_sigma_macs / total_qkv_dense_macs * 100, 2)])
        _mac_dist_table.add_row(
            ["pro", total_pro_sigma_macs / 10**3, total_pro_dense_macs / 10**3, np.round(total_pro_sigma_macs / total_pro_dense_macs * 100, 2)])
        _mac_dist_table.add_row(
            ["fc1", total_fc1_sigma_macs / 10**3, total_fc1_dense_macs / 10**3, np.round(total_fc1_sigma_macs / total_fc1_dense_macs * 100, 2)])
        _mac_dist_table.add_row(
            ["fc2", total_fc2_sigma_macs / 10**3, total_fc2_dense_macs / 10**3, np.round(total_fc2_sigma_macs / total_fc2_dense_macs * 100, 2)])
        _mac_dist_table.add_row(
            ["att", total_att_sigma_macs / 10**3, total_att_dense_macs / 10**3, np.round(total_att_sigma_macs / total_att_dense_macs * 100, 2)])

        if verbose:
            print(_mac_dist_table)  
    
    if arch == "fuse":
        _mac_dist_table = PrettyTable()
        _mac_dist_table.field_names = [
            "name", "sigma", "dense", "density", "protion"
        ]
        _mac_dist_table.add_row(
            [
                "exp macs", 
                "{:.2f}".format(total_exp_sigma_macs / 10**3), 
                "{:.2f}".format(total_exp_dense_macs / 10**3), 
                "{:.2f}".format(total_exp_sigma_macs / total_exp_dense_macs * 100, 2),
                "{:.2f}".format(total_exp_dense_macs / total_dense_macs * 100, 2)
            ]
        )
        _mac_dist_table.add_row(
            [
                "relu macs", 
                "{:.2f}".format(total_con_sigma_macs / 10**3), 
                "{:.2f}".format(total_con_dense_macs / 10**3), 
                "{:.2f}".format(total_con_sigma_macs / total_con_dense_macs * 100, 2),
                "{:.2f}".format(total_con_dense_macs / total_dense_macs * 100, 2)
            ]
        )

        _mac_dist_table.add_row(
            [
                "exp cycs", 
                "{:.2f}".format(total_exp_sigma_cycs / 10**3), 
                "{:.2f}".format(total_exp_dense_cycs / 10**3), 
                "{:.2f}".format(total_exp_sigma_cycs / total_exp_dense_cycs * 100, 2),
                "{:.2f}".format(total_exp_dense_cycs / total_dense_cycs * 100, 2)
            ]
        )

        _mac_dist_table.add_row(
            [
                "relu cycs", 
                "{:.2f}".format(total_con_sigma_cycs / 10**3), 
                "{:.2f}".format(total_con_dense_cycs / 10**3), 
                "{:.2f}".format(total_con_sigma_cycs / total_con_dense_cycs * 100, 2),
                "{:.2f}".format(total_con_dense_cycs / total_dense_cycs * 100, 2)
            ]
        )


        if verbose:
            print(_mac_dist_table)
    
    return df, metrics, avg_metric_dit, total_dense_macs / 10**9