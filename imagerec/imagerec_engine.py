import torch
import yaml
import numpy as np
from argparse import Namespace
import copy
from typing import Callable

from sbalign.training.diffusivity import get_diffusivity_schedule, matrix_vector_mp
from sbalign.utils.sb_utils import get_t_schedule
from sbalign.utils.definitions import DEVICE
from sbalign.utils.ops import to_numpy

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

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
                    t = self.t_schedule[t_idx].float()
                   # data.t = (t * data.x.new_ones(data.num_nodes))#.float()
                    t = t[None,None].to(DEVICE)
                    T = self.g_fn.T.to(DEVICE)

                    x = pos[:,:,:,:,0]
                    Y = pos[:,:,:,:,1:]
                    F = self.g_fn.F_t[None,None,None,None,:,:].to(DEVICE)
                    G = self.g_fn.G_t[None,None,None,None,:].to(DEVICE)
                    GG = self.g_fn.G_t[None,None,None,None,:,None].to(DEVICE) * self.g_fn.G_t[None,None,None,None,None,:].to(DEVICE)
                    dw = torch.sqrt(self.dt) * torch.randn_like(x)[:,:,:,:,None]

                    pos_t = self.g_fn.input_transform(x,Y,t,T,self.g_fn.omega.to(DEVICE), self.g_fn.gamma.to(DEVICE),self.g_fn.g_max.to(DEVICE))
                    # print('for K>0 - data.t:',data.t.dtype,flush=True)
                    # print('for K>0 - data.pos_t:',data.pos_t.dtype,flush=True)
                    drift_pos_x = self.model(pos_0, pos_t, t)

                    drift_pos = self.g_fn.score(drift_pos_x.to(DEVICE),t,T,self.g_fn.omega.to(DEVICE), self.g_fn.gamma.to(DEVICE),self.g_fn.g_max.to(DEVICE))
                    dpos = (matrix_vector_mp(F, pos) + matrix_vector_mp(GG, drift_pos))*self.dt + G * dw

                    # print("pos ", pos.shape)
                    # print("dpos shape ", dpos.shape)

                    # print("matrix_vector_mp(GG, drift_pos) ", matrix_vector_mp(GG, drift_pos).shape)
                    # print("matrix_vector_mp(F, pos) ", matrix_vector_mp(F, pos).shape)
                    # print("self.dt ", self.dt.shape)
                    # print("drift_pos_x ", drift_pos_x.shape)
                    # print("drift_pos ", drift_pos.shape)
                    # print("G * dw ", (G * dw).shape)
                    
                    pos = pos + dpos
                    trajectory.append(pos)
                else:
                    pos_t = pos_0.clone().to(DEVICE)
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
                    if self.g_fn.K > 0:
                        half_time_image = pos
                    else:                        
                        half_time_image = pos_t   

        trajectory = torch.stack(trajectory, dim=0)

        # print("half_time_image shape ", half_time_image.shape)
        # print("half_time_image[:,:,:,:,0] shape ", half_time_image[:,:,:,:,0].shape)
        # print("trajectory[-1,:,:,:,:,0] shape ", trajectory[-1,:,:,:,:,0].shape)
        # print("trajectory[:,:,:,:,0] shape ", trajectory[:,:,:,:,0].shape)

        if self.g_fn.K>0:
            return half_time_image[:,:,:,:,0], trajectory[-1,:,:,:,:,0], trajectory[0,:,:,:,:,0] # dimensions: [1, 3, 321, 481], [1, 3, 321, 481], ([1, 3, 321, 6])
        else:
            return half_time_image, trajectory[-1], trajectory

    def compute_psnr(self, inferred_image, pos_T):
        mse = torch.mean((inferred_image - pos_T) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        return psnr

    def compute_ssim(self, inferred_image, pos_T):
        ssim = StructuralSimilarityIndexMeasure(data_range=255.0).to(DEVICE)
        return ssim(inferred_image, pos_T)

    # def update_fid(self, inferred_image, pos_T):
    #     fid = FrechetInceptionDistance()
    #     return fid.update(inferred_image, pos_T)

    def bgr2ycbcr(self, img, only_y=True):
        '''bgr version of rgb2ycbcr
        only_y: only return Y channel
        Input:
            uint8, [0, 255]
            float, [0, 1]
        '''
        in_img_type = img.dtype
        print(f"Wile transferring to YCBCR colourscheme image type is {in_img_type}")
        print(f"image dimension is {img.shape}")
        device = img.device
        
        img = img.to(device=device, dtype=torch.float32)

        if in_img_type != torch.uint8:
            img *= 255.
        # convert
        if only_y:
            coeffs = torch.tensor([24.966, 128.553, 65.481], device=device)
            rlt = torch.tensordot(img, coeffs, dims=([1], [0])) / 255.0 + 16.0
            #rlt = (img @ coeffs) / 255.0 + 16.0
        else:            
            # Full conversion matrix for BGR to YCbCr
            matrix = torch.tensor([
                [24.966, 112.0,   -18.214],
                [128.553, -74.203, -93.786],
                [65.481,  -37.797, 112.0]
            ], device=device)
            offset = torch.tensor([16, 128, 128], device=device)
            # img shape: (..., 3), matrix: (3, 3)
            rlt = torch.tensordot(img, matrix, dims=([1], [0])) / 255.0 + offset

        # If input was uint8, round result
        if in_img_type == torch.uint8:
            rlt = torch.round(rlt)
        else:
            rlt = rlt / 255.

        # Cast back to original dtype
        rlt = rlt.to(dtype=in_img_type, device=device)
        return rlt

    def generate_images(self, data):
        pos_T, pos_0 = data['GT'], data['LQ']
        pos_T = pos_T.to(DEVICE)
        pos_0 = pos_0.to(DEVICE)
        metrics = {}

        half_time_image, inferred_image, trajectory = self.generate_image(pos_0 = pos_0)

        print("max value of generated image: ", torch.max(inferred_image))
        print("min value of generated image: ", torch.min(inferred_image))
        #assert ((torch.max(inferred_image) <= 1).all() and (torch.min(inferred_image) >= -1).all()).item(), "generated image is not normalised in [-1,1]"

        psnr = self.compute_psnr(inferred_image * 255, pos_T * 255)

        inferred_image_ycbcr = self.bgr2ycbcr(inferred_image * 255)
        pos_T_ycbcr = self.bgr2ycbcr(pos_T * 255)
        psnr_y = self.compute_psnr(inferred_image_ycbcr, pos_T_ycbcr)
        ssim = self.compute_ssim(inferred_image_ycbcr[:,None,:,:], pos_T_ycbcr[:,None,:,:]) # Expected `preds` and `target` to have BxCxHxW or BxCxDxHxW shape. Got preds: torch.Size([1, 321, 481]) and target: torch.Size([1, 321, 481]).
        
        metrics['psnr'] = psnr.item()
        metrics['psnr_y'] = psnr_y.item()
        metrics['ssim'] = ssim.item()

        return half_time_image, inferred_image, metrics #psnr.item()
