import torch
import torch.nn as nn
from model.mlp import MLP, PhaseMLP

class RTN(nn.Module):
    def __init__(self,
        dof,
        hidden_dims={
            "hidden_init": [512],
            "frame_encoder": [512],
            "offset_encoder": [128],
            "target_encoder": [128],
            "lstm": 512,
            "decoder": [256, 128],
        },
        output_dims={
            "hidden_init": 512,
            "frame_encoder": 512,
            "offset_encoder": 128,
            "target_encoder": 128,
        }
    ):
        super(RTN, self).__init__()
        self.dof = dof

        self.h_initializer = MLP(dof, hidden_dims["hidden_init"], output_dims["hidden_init"], activation=nn.LeakyReLU(), activation_at_last=True)
        self.c_initializer = MLP(dof, hidden_dims["hidden_init"], output_dims["hidden_init"], activation=nn.LeakyReLU(), activation_at_last=True)
        self.frame_encoder = MLP(dof, hidden_dims["frame_encoder"], output_dims["frame_encoder"], activation=nn.LeakyReLU(), activation_at_last=True)
        self.offset_encoder = MLP(dof, hidden_dims["offset_encoder"], output_dims["offset_encoder"], activation=nn.LeakyReLU(), activation_at_last=True)
        self.target_encoder = MLP(dof, hidden_dims["target_encoder"], output_dims["target_encoder"], activation=nn.LeakyReLU(), activation_at_last=True)
        self.lstm = nn.LSTM(input_size=output_dims["frame_encoder"] + output_dims["offset_encoder"] + output_dims["target_encoder"], hidden_size=hidden_dims["lstm"], num_layers=1, batch_first=True)

        self.decoder = MLP(input_dim=hidden_dims["lstm"], hidden_dims=hidden_dims["decoder"], output_dim=dof)
    
    def set_target(self, target):
        self.target = target
        self.enc_target = self.target_encoder(target)

    def init_hidden(self, batch_size, init_frame):
        self.h = self.h_initializer(init_frame).view(1, batch_size, -1)
        self.c = self.c_initializer(init_frame).view(1, batch_size, -1)

    def forward(self, x):
        enc_frame = self.frame_encoder(x)
        enc_offset = self.offset_encoder(self.target - x)
        h_in = torch.cat([enc_frame, enc_offset, self.enc_target], dim=-1).unsqueeze(1)
        output, (self.h, self.c) = self.lstm(h_in, (self.h, self.c))
        return self.decoder(output.squeeze(1)) + x

class PhaseRTN(nn.Module):
    def __init__(self,
        dof,
        hidden_dims={
            "hidden_init": [512],
            "frame_encoder": [512],
            "offset_encoder": [128],
            "target_encoder": [128],
            "lstm": 512,
            "decoder": [256, 128],
        },
        output_dims={
            "hidden_init": 512,
            "frame_encoder": 512,
            "offset_encoder": 128,
            "target_encoder": 128,
        }
    ):
        super(PhaseRTN, self).__init__()
        self.dof = dof

        self.h_initializer = PhaseMLP(dof, hidden_dims["hidden_init"], output_dims["hidden_init"], activation_at_last=True)
        self.c_initializer = PhaseMLP(dof, hidden_dims["hidden_init"], output_dims["hidden_init"], activation_at_last=True)
        self.frame_encoder = PhaseMLP(dof, hidden_dims["frame_encoder"], output_dims["frame_encoder"], activation_at_last=True)
        self.offset_encoder = PhaseMLP(dof, hidden_dims["offset_encoder"], output_dims["offset_encoder"], activation_at_last=True)
        self.target_encoder = PhaseMLP(dof, hidden_dims["target_encoder"], output_dims["target_encoder"], activation_at_last=True)
        self.lstm = nn.LSTM(input_size=output_dims["frame_encoder"] + output_dims["offset_encoder"] + output_dims["target_encoder"], hidden_size=hidden_dims["lstm"], num_layers=1, batch_first=True)

        self.decoder = PhaseMLP(input_dim=hidden_dims["lstm"], hidden_dims=hidden_dims["decoder"], output_dim=dof+1)
    
    def set_target(self, target, phase):
        self.target = target
        self.enc_target = self.target_encoder(target, phase)

    def init_hidden(self, batch_size, init_frame, phase):
        self.h = self.h_initializer(init_frame, phase).view(1, batch_size, -1)
        self.c = self.c_initializer(init_frame, phase).view(1, batch_size, -1)

    def forward(self, x, phase):
        enc_frame = self.frame_encoder(x, phase)
        enc_offset = self.offset_encoder(self.target - x, phase)
        h_in = torch.cat([enc_frame, enc_offset, self.enc_target], dim=-1).unsqueeze(1)
        output, (self.h, self.c) = self.lstm(h_in, (self.h, self.c))
        output = self.decoder(output.squeeze(1), phase)
        return output[..., :-1] + x, output[..., -1]