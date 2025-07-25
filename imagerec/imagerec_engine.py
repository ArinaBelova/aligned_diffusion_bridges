import torch
import yaml
import numpy as np
from argparse import Namespace
import copy
from typing import Callable

from sbalign.training.diffusivity import get_diffusivity_schedule
from sbalign.utils.sb_utils import get_t_schedule
from sbalign.utils.definitions import DEVICE
from sbalign.utils.ops import to_numpy



class ImageRecEngine:
    def __init__(self,
                 inference_steps: int,
                 model: torch.nn.Module = None,
                 g_fn: Callable = None):

        self.inference_steps = inference_steps

        if model is None:
            raise ValueError("Model must be provided")
            
        self.model = model.to(DEVICE)
        self.model.eval()

        self.t_schedule = torch.from_numpy(get_t_schedule(inference_steps=inference_steps))
        self.dt = self.t_schedule[1] - self.t_schedule[0]

        if g_fn is None:
            #raise ValueError("Diffusivity function must be provided")
            g_fn = get_diffusivity_schedule(model_args.diffusivity_schedule,
                                            g_max=model_args.max_diffusivity,
                                            K = args.K,
                                            H = args.H,
                                            norm = args.norm)
        self.g_fn = g_fn

    def generate_image(self, pos_0):
        pos_orig = pos_0.clone().to(DEVICE)
        pos_t = pos_0.clone().to(DEVICE)

        if self.g_fn.K > 0:        
            pos = torch.cat([pos_orig[:,:,:,:,None],torch.zeros(pos_orig.shape[0], pos_orig.shape[1], pos_orig.shape[2], pos_orig.shape[3], self.g_fn.K, device=DEVICE)],dim=-1)
        else:
            pos = pos_orig.clone().to(DEVICE)

        # generate the image
        trajectory = []

        with torch.no_grad():
            for t_idx in range(self.inference_steps):
                t = self.t_schedule[t_idx].float().to(DEVICE)
                T = self.g_fn.T.to(DEVICE)
            
                if self.g_fn.K > 0:
                    pass
                else:
                    g_t = self.g_fn.g(t).to(DEVICE)
                    std = torch.sqrt(((self.g_fn.g(t)**2)*(1-t)))
                    drift = self.model(pos_0, pos_t, t) #/ std
                    if t_idx == (self.inference_steps - 1):
                        diffusion = 0
                    else:
                        diffusion = g_t * torch.randn_like(pos_t, device=DEVICE) * torch.sqrt(self.dt).to(DEVICE)

                    dpos = torch.square(g_t) * drift * self.dt + diffusion
                    pos_t = pos_t + dpos
                    trajectory.append(pos_t)

                if t_idx == self.inference_steps // 2:
                    half_time_image = pos_t   

        if self.g_fn.K>0:
            return half_time_image[:,:,:,:,0], trajectory[-1,:,:,:,:,0], trajectory[:,:,:,:,0] # not sure here about the dimensions...
        else:
            return half_time_image, trajectory[-1], trajectory

    def compute_psnr(self, inferred_image, pos_T):
        mse = torch.mean((inferred_image - pos_T) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        return psnr

    def generate_images(self, data):
        pos_T, pos_0 = data['GT'], data['LQ']
        pos_T = pos_T.to(DEVICE)
        pos_0 = pos_0.to(DEVICE)
        #metrics = {}

        half_time_image, inferred_image, trajectory = self.generate_image(pos_0 = pos_0)
        psnr = self.compute_psnr(inferred_image * 255, pos_T * 255)
        #metrics['psnr'] = psnr.item()

        return half_time_image, inferred_image, psnr.item()
