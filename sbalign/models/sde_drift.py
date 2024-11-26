import torch
import torch.nn as nn

from sbalign.models.common import build_mlp
from sbalign.utils.sb_utils import get_timestep_embedding


class SDEDrift(nn.Module):

    def __init__(self,
                 timestep_emb_dim: int = 64, 
                 n_layers: int = 3,
                 in_dim: int = 2,
                 out_dim: int = 2,
                 h_dim: int = 64,
                 activation: int = 'relu',
                 dropout_p: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.x_enc = build_mlp(in_dim=in_dim, h_dim=h_dim, n_layers=n_layers,
                             out_dim=h_dim, dropout_p=dropout_p,
                             activation=activation)
        
        self.t_enc = build_mlp(in_dim=timestep_emb_dim, h_dim=h_dim, n_layers=2,
                             out_dim=h_dim, dropout_p=dropout_p,
                             activation=activation)

        self.mlp = build_mlp(in_dim=2*h_dim, h_dim=h_dim, n_layers=n_layers,
                             out_dim=out_dim, dropout_p=dropout_p,
                             activation=activation)
        if timestep_emb_dim > 1:
            self.timestep_emb_fn = get_timestep_embedding('sinusoidal', embedding_dim=timestep_emb_dim)
        else:
            self.timestep_emb_fn = lambda x: x 
    
    def forward(self, x, t):
        #print('t shape ', t.shape) #[32, 1]
        t_encoded = self.t_enc(self.timestep_emb_fn(t))
        #print('t_encoded shape ', t_encoded.shape) # [32, 64]
        #print('t_encoded shape ', t_encoded[:, None, :].shape) # [32, 1, 64]
        x_encoded = self.x_enc(x) 
        #print('x_encoded shape is ', x_encoded.shape) # [32, 2, 64]

        # I just duplicated time tensor across the channel dimension here:
        #t_encoded = t_encoded[:, None, :].repeat(1,2,1)
        #print('t_encoded shape after reshape ', t_encoded.shape) # [32, 2, 64]
        inputs = torch.cat([x_encoded, t_encoded], dim=-1)
        return self.mlp(inputs)
