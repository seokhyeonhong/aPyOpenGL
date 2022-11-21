import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU(), activation_at_last=False):
        super(MLP, self).__init__()
        self.activation = activation
        self.activation_at_last = activation_at_last

        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(activation)
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))
        if activation_at_last:
            layers.append(activation)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class PhaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=F.elu, activation_at_last=False):
        super(PhaseMLP, self).__init__()

        self.params_w = nn.ParameterList()
        self.params_b = nn.ParameterList()
        for h_dim in hidden_dims:
            self.params_w.append(Parameter(torch.Tensor(4, input_dim, h_dim)))
            self.params_b.append(Parameter(torch.Tensor(4, 1, h_dim)))
            self.register_parameter("w_{}".format(len(self.params_w) - 1), self.params_w[-1])
            self.register_parameter("b_{}".format(len(self.params_b) - 1), self.params_b[-1])
            input_dim = h_dim
        self.params_w.append(Parameter(torch.Tensor(4, input_dim, output_dim)))
        self.params_b.append(Parameter(torch.Tensor(4, 1, output_dim)))
        self.register_parameter("w_{}".format(len(self.params_w) - 1), self.params_w[-1])
        self.register_parameter("b_{}".format(len(self.params_b) - 1), self.params_b[-1])
    
        for i in range(len(self.params_w)):
            nn.init.kaiming_uniform_(self.params_w[i])
            nn.init.zeros_(self.params_b[i])
        
        self.activation = activation
        self.activation_at_last = activation_at_last
    
    def forward(self, x, phase):
        x = x.unsqueeze(1)
        w, idx_0, idx_1, idx_2, idx_3 = self.phase_idx(phase)
        for i in range(0, len(self.params_w)):
            param_w, param_b = self.params_w[i], self.params_b[i]
            weight = self.cubic(param_w[idx_0], param_w[idx_1], param_w[idx_2], param_w[idx_3], w)
            bias = self.cubic(param_b[idx_0], param_b[idx_1], param_b[idx_2], param_b[idx_3], w)
            x = torch.bmm(x, weight) + bias
            if i < len(self.params_w) - 1 or self.activation_at_last:
                x = self.activation(x)
        return x.squeeze(1)
    
    def cubic(self, a0, a1, a2, a3, w):
        return\
            a1\
            +w*(0.5*a2 - 0.5*a0)\
            +w*w*(a0 - 2.5*a1 + 2*a2 - 0.5*a3)\
            +w*w*w*(1.5*a1 - 1.5*a2 + 0.5*a3 - 0.5*a0)

    def phase_idx(self, phase):
        w = 4 * phase
        idx_1 = torch.remainder(w.floor().long(), 4)
        idx_0 = torch.remainder(idx_1 - 1, 4)
        idx_2 = torch.remainder(idx_1 + 1, 4)
        idx_3 = torch.remainder(idx_1 + 2, 4)
        w = torch.fmod(w, 1)
        return w.view(-1, 1, 1), idx_0.view(-1), idx_1.view(-1), idx_2.view(-1), idx_3.view(-1)

class MultiLinear(nn.Module):
    def __init__(self, num_layers, in_features, out_features, bias=True):
        super(MultiLinear, self).__init__()
        
        self.weight = nn.Parameter(torch.Tensor(num_layers, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(num_layers, 1, out_features)) if bias == True else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias != None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        return x if self.bias == None else x + self.bias