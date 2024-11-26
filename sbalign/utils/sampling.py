import torch
import numpy as np

from sbalign.utils.definitions import DEVICE
from sbalign.training.diffusivity import fractional_input_transform, matrix_vector_mp


def sampling(pos_0, model, diffusivity, inference_steps, t_schedule, apply_score=False, return_traj: bool=False):

    #pos_0 = pos_0.clone()
    print('pos_0 shape',pos_0.shape[0])
    pos = torch.cat([pos_0[:,:,None],torch.zeros(pos_0.shape[0], pos_0.shape[1], diffusivity.K)],dim=-1)
    g = diffusivity.g

    model.eval()
    y_T = diffusivity.sample(diffusivity.T*torch.ones(pos_0.shape[0],1), c=2, h=1, w=1)[:,:,1:]
    print('K in sampling',diffusivity.K)
    trajectory = np.zeros((inference_steps+1, *pos_0.shape, diffusivity.K+1))
    trajectory[0] = pos.cpu()

    dt = t_schedule[1] - t_schedule[0]

    with torch.no_grad():
        for t_idx in range(1, inference_steps+1):
            t = t_schedule[t_idx]
            print('t in sampling',t.shape)
            if diffusivity.K > 0:

                nn_input, cond_var = fractional_input_transform(pos, t[None,None], y_T, diffusivity)
                cond_std = torch.sqrt(cond_var)
            # _, _, _, _, eta_Tt, sig_Tt, tau_Tt = diffusivity.marginal_stats(1.0 - t)

            #data.cond_var_t =  (sig_Tt - tau_Tt)[:,0,0]
            #varphi = dif.g(data.t) * dif.omega * dif.gamma * (1.0 - data.t) + torch.exp(-dif.gamma * (1.0 - data.t))
            #data.pos_t = data.pos_t[:,:,0] + torch.sum(eta_Tt[:,0,0] * data.aug_pos_T[:,:,1:] + varphi[:,None,:] * data.pos_t[:,:,1:], dim=-1)
            
                drift_pos_x = model.run_drift(nn_input, torch.ones(nn_input.shape[0]).to(DEVICE)* t)/cond_std
                varphi = diffusivity.g(t[None,None]) * diffusivity.omega * diffusivity.gamma * (1.0 - t[None,None]) + torch.exp(-diffusivity.gamma * (1.0 - t[None,None]))
                cond_score = torch.cat([drift_pos_x, -varphi * drift_pos_x], dim=-1)

                zeros = torch.zeros_like(nn_input)[:, :, None]
                _,_, _, cov_yy, _, _, _ = diffusivity.marginal_stats(t[None,None])
                y_t = pos[:,:,1:]
                nabla_y_t = torch.linalg.solve(cov_yy, torch.squeeze(y_t), left=False)[:, :, None, None, :]
                score_y_t = -torch.cat([zeros, nabla_y_t], dim=-1)
                drift_pos = cond_score + score_y_t
            else:
                drift_pos = model.run_drift(pos, torch.ones(pos.shape[0]).to(DEVICE)* t)

            if apply_score:
                assert False, "Must pass x_T as parameter of the function"
                # torch.stack([torch.ones(pos.shape[0], device=DEVICE)*5, pos[:,1]], axis=1)
                # drift_pos = drift_pos + model.run_doobs_score(pos, ..., torch.ones(pos.shape[0]).to(DEVICE)* t)
            print('dif.K',diffusivity.K)
            if diffusivity.K > 0:
                G = diffusivity.G(t[None])[:,:,0,0,:]
                GG = (G[:,:,:,None] * G[:,:,None,:])
                print('pos',pos.shape)
                diffusion = G * torch.randn_like(pos[:,:,0])[:,:,None] * torch.sqrt(dt)
                print('diffusion',diffusion.shape)
                print('drift_pos',drift_pos.shape)
                print('GG',GG.shape)
                dpos = matrix_vector_mp(GG, drift_pos) * dt + diffusion

            else:
                diffusion = g(t) * torch.randn_like(pos) * torch.sqrt(dt)

                dpos = np.square(g(t)) * drift_pos * dt + diffusion
            
            pos = pos + dpos

            trajectory[t_idx] = pos.cpu()

    if return_traj:
        return trajectory
    else:
        return trajectory[-1]

