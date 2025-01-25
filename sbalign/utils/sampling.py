import torch
import numpy as np

from sbalign.utils.definitions import DEVICE
from sbalign.training.diffusivity import fractional_input_transform, matrix_vector_mp


def sampling(pos_0, model, diffusivity, inference_steps, t_schedule, apply_score=False, return_traj: bool=False):

    model.eval()

    if diffusivity.K > 0:
        pos = torch.cat([pos_0[:,:,None],torch.zeros(pos_0.shape[0], pos_0.shape[1], diffusivity.K)],dim=-1)
        trajectory = np.zeros((inference_steps+1, *pos_0.shape, diffusivity.K+1))

    else:
        pos = pos_0.clone()
        trajectory = np.zeros((inference_steps+1, *pos_0.shape))   

    trajectory[0] = pos.cpu()

    dt = t_schedule[1] - t_schedule[0]

    with torch.no_grad():
        for t_idx in range(1, inference_steps+1):

            if diffusivity.K > 0:
                
                t = t_schedule[t_idx][None,None]
                T = diffusivity.T

                x = pos[:,:,0]
                Y = pos[:,:,1:]
                F = diffusivity.F_t[None,None,:,:]
                G = diffusivity.G_t[None,None,:]
                GG = diffusivity.G_t[None,None,:,None] * diffusivity.G_t[None,None,None,:]
                dw = torch.sqrt(dt) * torch.randn_like(x)[:,:,None]

                pos_transform = diffusivity.input_transform(x,Y,t,T,diffusivity.omega, diffusivity.gamma,diffusivity.g_max)
                drift_pos_x = model.run_drift(pos_transform, torch.ones(pos_transform.shape[0]).to(DEVICE)* t[0,0])
                drift_pos = diffusivity.score(drift_pos_x,t,T,diffusivity.omega, diffusivity.gamma,diffusivity.g_max)
                dpos = (matrix_vector_mp(F, pos) + matrix_vector_mp(GG, drift_pos))*dt + G * dw
            else:
                t = t_schedule[t_idx]
                g = diffusivity.g
                drift_pos = model.run_drift(pos, torch.ones(pos.shape[0]).to(DEVICE)* t)
                diffusion = g(t) * torch.randn_like(pos) * torch.sqrt(dt)
                dpos = np.square(g(t)) * drift_pos * dt + diffusion
            
            pos = pos + dpos
            
            trajectory[t_idx] = pos.cpu()

    if return_traj:
        return trajectory
    else:
        return trajectory[-1]

