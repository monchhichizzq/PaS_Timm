import torch
import torch.nn as nn
from torch.autograd import Function


class Identity_In(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # self.sig_x = torch.tensor(x, dtype=torch.float32)
        self.sig_x = x
        print("sig x:", self.sig_x.shape)
        return x

class ThrFunction(nn.Module):
    def forward(self, x, thresh):
        x_fw = x * torch.greater(torch.abs(x), thresh)
        x_bw = x
        x_out = x_bw + x_fw.detach() - x_bw.detach() 
        return x_out

class ThrInpFunction(Function):
    @staticmethod
    def forward(ctx, x, thresh):
        # Create mask and apply thresholding inplace
        mask = torch.abs_(x) > thresh  # inplace abs
        x.mul_(mask)                   # inplace multiplication
        return x
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None       # no gradient for threshold

class FloInpFunction(Function):
    @staticmethod
    def forward(ctx, x, thresh):
        # Create mask and apply thresholding inplace
        x_flo = torch.floor_(x / thresh)
        x_flo.mul_(thresh)              # inplace multiplication
        return x
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None       # no gradient for threshold


class RegThrInpFunction(Function):
    @staticmethod
    def forward(ctx, x, thresh, l1_lambda=0.01):
        # Save input tensor for backward pass
        ctx.save_for_backward(x)
        ctx.l1_lambda = l1_lambda

        # Create mask and apply thresholding inplace
        mask = torch.abs_(x) > thresh  # inplace abs
        x.mul_(mask)                   # inplace multiplication
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        l1_lambda = ctx.l1_lambda
        x = ctx.saved_tensors[0]
        
        # Gradient of L1 penalty: d(|x|)/dx = sign(swish(x)) * swish'(x)
        grad_l1 = l1_lambda * torch.sum(torch.abs(x)) * torch.sign(x) * grad_output
        
        grad_x = grad_output + grad_l1
        print("grad_1:", grad_l1.shape, torch.min(grad_l1), torch.max(grad_l1))
        print("grad_o:", grad_output.shape, torch.min(grad_output), torch.max(grad_output))
        print("grad_x:", grad_x.shape, torch.min(grad_x), torch.max(grad_x))
        print("")
        return grad_x, None, None


def else_suppression(x):
    if len(x.shape) == 4:
        _, sc, sh, sw = x.shape
        row_sup_loss = 0
        x_1 = x[:, :, :sh-1]
        x_2 = x[:, :, 1:]
        del_x = x_2 - x_1
        fir_x = torch.unsqueeze(x[:, :, 0:1], dim=2)
        del_map = torch.cat((fir_x, del_x), dim=2)

        sig_map = x_1 + del_map[:, :, 1:]
        del_fire = del_map
        sig_fire = torch.cat((fir_x, del_fire), dim=2)
    
    if len(x.shape) == 3:
        _, n_token, token_dim = x.shape
        row_sup_loss = 0
        x_1 = x[:, :n_token-1]
        x_2 = x[:, 1:]
        del_x = x_2 - x_1
        fir_x = torch.unsqueeze(x[:, 0], dim=1)
        del_map = torch.cat((fir_x, del_x), dim=1)

        sig_map = x_1 + del_map[:, 1:]
        del_fire = del_map
        sig_fire = torch.cat((fir_x, del_fire), dim=1)

    else:
        _, sc = x.shape
        del_fire = x
        sig_fire = x
    return del_fire, sig_fire


class Input_Threshold(nn.Module):
    def __init__(self, thresh=-25, inplace=False):
        nn.Module.__init__(self)
        self.threshold = thresh
        self.inplace = inplace
        self.l1_lambda = 0 # 1e-7
        # print("threshold: {}, inplace: {}, l1 coeff: {}".format(self.threshold, self.inplace, self.l1_lambda))

        self.input_shape = 0.
        self.trans_policy = "thresholding"
        self.static_macs = 0.
        self.sigma_events = 0.
        self.spati_events = 0.
        self.dense_events = 1.

    def forward(self, x):
        self.input_shape = x.shape[1:]
        self.sig_x = x # torch.tensor(x, dtype=torch.float32)
        # self.sig_loss = torch.sum(torch.abs(x)) 

        if self.inplace:
            if self.l1_lambda > 0:
                x = RegThrInpFunction.apply(x, self.threshold, self.l1_lambda)
            else:
                if self.trans_policy == "thresholding":
                    x = ThrInpFunction.apply(x, self.threshold) # wrong
                elif self.trans_policy == "flooring":
                    x = FloInpFunction.apply(x, self.threshold)
                    del_fire, x = else_suppression(x)
                    self.spati_events, self.dense_events = calculate_events(del_fire)
                else:
                    print(f"{self.trans_policy} not support")
        else:
            x = ThrFunction()(x, self.threshold)
        
        self.sigma_events, self.dense_events = calculate_events(x)
        return x
 

class Input_Placeholder(nn.Module):
    def __init__(self, thresh=-25, inplace=False):
        nn.Module.__init__(self)
        self.threshold = thresh
        self.inplace = inplace
        self.l1_lambda = 0 # 1e-7
        # print("threshold: {}, inplace: {}, l1 coeff: {}".format(self.threshold, self.inplace, self.l1_lambda))

        self.input_shape = 0.
        self.trans_policy = "thresholding"
        self.static_macs = 0.
        self.sigma_events = 0.
        self.spati_events = 0.
        self.dense_events = 1.

    def forward(self, x):
        self.input_shape = x.shape[1:]
        self.sig_x = x # torch.tensor(x, dtype=torch.float32)
        # self.sig_loss = torch.sum(torch.abs(x)) 

        if self.inplace:
            if self.l1_lambda > 0:
                x = RegThrInpFunction.apply(x, self.threshold, self.l1_lambda)
            else:
                if self.trans_policy == "thresholding":
                    x = ThrInpFunction.apply(x, self.threshold) # wrong
                elif self.trans_policy == "flooring":
                    x = FloInpFunction.apply(x, self.threshold)
                    del_fire, x = else_suppression(x)
                    self.spati_events, self.dense_events = calculate_events(del_fire)
                else:
                    print(f"{self.trans_policy} not support")
        else:
            x = ThrFunction()(x, self.threshold)
        
        self.sigma_events, self.dense_events = calculate_events(x)
        return x


def calculate_events(feature):
    dyn_evts = torch.mean((feature != 0).float(), dim=0)
    zeo_evts = torch.mean((feature == 0).float(), dim=0)
    sta_evts = dyn_evts + zeo_evts

    batch_dyn_evts = torch.sum(dyn_evts)
    batch_zeo_evts = torch.sum(zeo_evts)
    batch_sta_evts = torch.sum(sta_evts)
    return batch_dyn_evts, batch_sta_evts
