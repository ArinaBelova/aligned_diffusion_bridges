import torch
import yaml
import numpy as np
from argparse import Namespace
import copy
from typing import Callable

from proteins.conf.models import build_model_from_args
from sbalign.utils.sb_utils import get_t_schedule
from sbalign.training.diffusivity import get_diffusivity_schedule
from sbalign.utils.ops import to_numpy
from sbalign.utils.definitions import DEVICE

from sbalign.training.diffusivity import matrix_vector_mp

def rmsd(y_pred, y_true):
    se = (y_pred - y_true)**2
    mse = se.sum(axis=1).mean()
    return np.sqrt(mse)


class ConfEngine:

    def __init__(self,
                 samples_per_protein: int,
                 inference_steps: int,
                 model_file: str = None,
                 config_file: str = None,
                 model: torch.nn.Module = None,
                 g_fn: Callable = None,
            ):
        self.samples_per_protein = samples_per_protein
        
        if model is None:
            #print("You did not supply existing model, creating a new model from scratch!!!")
            with open(config_file) as f:
                model_args = Namespace(**yaml.full_load(f))

            model = build_model_from_args(model_args)
            model_dict = torch.load(model_file, map_location='cpu')
            model.load_state_dict(model_dict)
        
        self.model = model.to(DEVICE)
        self.model.eval()
        print(f'Sampling with steps={inference_steps}')
        self.inference_steps = inference_steps
        t_schedule = get_t_schedule(inference_steps=inference_steps)
        self.t_schedule = torch.from_numpy(t_schedule)
        self.dt = self.t_schedule[1] - self.t_schedule[0]

        if g_fn is None:
            g_fn = get_diffusivity_schedule(model_args.diffusivity_schedule,
                                            g_max=model_args.max_diffusivity,
                                            K = args.K,
                                            H = args.H,
                                            norm = args.norm)
        self.dif = g_fn


    def generate_conformation(self, data):

        data.pos_T = None
        data.pos_t = data.pos_0
        data.pos_orig = data.pos_0.clone().to(DEVICE)

        if self.dif.K > 0:        
            pos = torch.cat([data.pos_orig[:,:,None],torch.zeros(data.pos_orig.shape[0], data.pos_orig.shape[1], self.dif.K, device=DEVICE)],dim=-1)
        else:
            pos = data.pos_orig.clone().to(DEVICE)

        trajectory = []
        with torch.no_grad():
            for t_idx in range(self.inference_steps):
            #for t_idx in range(self.inference_steps+1):
                if self.dif.K > 0:
                    t = self.t_schedule[t_idx].float()
                    data.t = (t * data.x.new_ones(data.num_nodes))#.float()
                    t = t[None,None].to(DEVICE)
                    T = self.dif.T.to(DEVICE)

                    x = pos[:,:,0]
                    Y = pos[:,:,1:]
                    F = self.dif.F_t[None,None,:,:].to(DEVICE)
                    G = self.dif.G_t[None,None,:].to(DEVICE)
                    GG = self.dif.G_t[None,None,:,None].to(DEVICE) * self.dif.G_t[None,None,None,:].to(DEVICE)
                    dw = torch.sqrt(self.dt) * torch.randn_like(x)[:,:,None]

                    data.pos_t = self.dif.input_transform(x,Y,t,T,self.dif.omega.to(DEVICE), self.dif.gamma.to(DEVICE),self.dif.g_max.to(DEVICE))
                    # print('for K>0 - data.t:',data.t.dtype,flush=True)
                    # print('for K>0 - data.pos_t:',data.pos_t.dtype,flush=True)
                    drift_pos_x = self.model.run_drift(data)

                    drift_pos = self.dif.score(drift_pos_x.to(DEVICE),t,T,self.dif.omega.to(DEVICE), self.dif.gamma.to(DEVICE),self.dif.g_max.to(DEVICE))
                    dpos = (matrix_vector_mp(F, pos) + matrix_vector_mp(GG, drift_pos))*self.dt + G * dw
                    
                    pos = pos + dpos
                    trajectory.append(pos)
                else:    
                    t = self.t_schedule[t_idx].to(DEVICE)

                    data.t = t * data.x.new_ones(data.num_nodes).to(DEVICE)
                    g_t = data.x.new_tensor(self.dif.g(t)).float().to(DEVICE)
                    # print('for K=0 - data.t:',data.t.dtype,flush=True)
                    # print('for K=0 - data.pos_t:',data.pos_t.dtype,flush=True)
                    std = 1.0 #orignal
                    #std = torch.sqrt((self.dif.g_max**2)*(1-t)) if t<1 else 1.0
                    # print('data.t',data.t.dtype)
                    # print('data.pos_t.',data.pos_t.dtype)
                    drift = self.model.run_drift(data.to(DEVICE)) / std
                    diffusion = g_t * torch.randn_like(data.pos_t,device=DEVICE)* torch.sqrt(self.dt).to(DEVICE)

                    dpos = torch.square(g_t) * drift * self.dt + diffusion
                    pos_t = data.pos_t  + dpos
                    data.pos_t = pos_t
                    trajectory.append(pos_t)

        trajectory = torch.stack(trajectory, dim=0)

        if self.dif.K>0:
            return trajectory[-1,:,:,0], trajectory[:,:,0]
        else:
            return trajectory[-1], trajectory

    
    def generate_conformations(self, data, apply_mean: bool = True):
        data = data.to(DEVICE)

        conformations, trajectories = [], []
        metrics = {'rmsd': [], "init_rmsd": []}

        for sample_id in range(self.samples_per_protein):
            data_copy = copy.deepcopy(data)

            conformation, trajectory = self.generate_conformation(data=data_copy)
            conformations.append(conformation)
            trajectories.append(trajectory)

            init_rmsd = rmsd(to_numpy(data.pos_0), to_numpy(data.pos_T))
            final_rmsd = rmsd(to_numpy(conformation), to_numpy(data.pos_T))

            metrics['init_rmsd'].append(init_rmsd)
            metrics['rmsd'].append(final_rmsd)
        
        trajectories = to_numpy(torch.stack(trajectories, dim=0).mean(dim=0))
        
        if apply_mean:
            for metric, metric_list in metrics.items():
                metrics[metric] = np.round(np.mean(metric_list), 4)

        return trajectories, metrics
