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
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from imagerec.models.vae_inference import decode
from imagerec.models.vae_training import DEFAULT_MODEL_DEF
from diffusers.models import AutoencoderKL


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
        #trajectory = []

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
                    with torch.inference_mode():
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
                    #trajectory.append(pos)
                else:
                    g_t = self.g_fn.g(t).to(DEVICE)
                    std = torch.sqrt(((self.g_fn.g(t)**2)*(1-t)))
                    drift = self.model(pos_0, pos_t, t) #/ std
                    if t_idx == (self.inference_steps - 1):
                        diffusion = 0
                    else:
                        diffusion = g_t * torch.randn_like(pos_t, device=DEVICE) * torch.sqrt(self.dt).to(DEVICE)

                    #print(f"MODEL RETURNED DRIFT {drift} and calculated diffusion is {diffusion}", flush=True)
                    dpos = torch.square(g_t) * drift * self.dt + diffusion
                    pos_t = pos_t + dpos

                    #print(f"max pos_t value is {torch.max(pos_t)} and min pos_t value is {torch.min(pos_t)}")
                    #trajectory.append(pos_t)

                if t_idx == self.inference_steps // 2:
                    if self.g_fn.K > 0:
                        half_time_image = pos
                    else:                        
                        half_time_image = pos_t   

        #trajectory = torch.stack(trajectory, dim=0)

        # print("half_time_image shape ", half_time_image.shape)
        # print("half_time_image[:,:,:,:,0] shape ", half_time_image[:,:,:,:,0].shape)
        # print("trajectory[-1,:,:,:,:,0] shape ", trajectory[-1,:,:,:,:,0].shape)
        # print("trajectory[:,:,:,:,0] shape ", trajectory[:,:,:,:,0].shape)

        torch.cuda.empty_cache()
        
        # if self.g_fn.K>0:
        #     return half_time_image[:,:,:,:,0], trajectory[-1,:,:,:,:,0]#, trajectory[0,:,:,:,:,0] # dimensions: [1, 3, 321, 481], [1, 3, 321, 481], ([1, 3, 321, 6])
        # else:
        #     return half_time_image, trajectory[-1]#, trajectory

        if self.g_fn.K > 0:
            return half_time_image[:,:,:,:,0], pos[:,:,:,:,0]
        else:
            return half_time_image, pos_t

    def generate_images(self, data, original_dataset=None, args=None): #, wandb):
        pos_T, pos_0 = data['GT'], data['LQ']
        pos_T = pos_T.to(DEVICE)
        pos_0 = pos_0.to(DEVICE)
        metrics = {}

        #half_time_image, inferred_image, trajectory = self.generate_image(pos_0 = pos_0)
        half_time_image, inferred_image = self.generate_image(pos_0 = pos_0)

        # for the latent diffusion case
        if args is not None and hasattr(args, "latent") and args.latent:
            # Need to decode the image here!!! 
            # def decode(
            #     vae: AutoencoderKL,
            #     latents: torch.Tensor,
            #     device: torch.device = torch.device('cpu')
            vae = AutoencoderKL(**DEFAULT_MODEL_DEF)
            vae.load_state_dict(torch.load(args.vae_checkpoint_path, map_location=DEVICE)['model'])

            half_time_image = original_dataset.postprocess(half_time_image)
            inferred_image = original_dataset.postprocess(inferred_image)
            pos_T = original_dataset.postprocess(pos_T)

            half_time_image = decode(vae, half_time_image, device=DEVICE)
            inferred_image = decode(vae, inferred_image, device=DEVICE)
            pos_T = decode(vae, pos_T, device=DEVICE)
            

        # To have a sliding sclae of the images generated during the diffusion process
        # for i in range(len(trajectory)):
        #     wandb.log({
        #                 "trajectory_image": wandb.Image(trajectory[i], caption=f"Step {i+1}"),
        #                 "timestep": i,
        #             })

        # print("max value of generated image before normalisation: ", torch.max(inferred_image))
        # print("min value of generated image before normalisation: ", torch.min(inferred_image))


        # print("max value of pos_T image ", torch.max(pos_T)) #1
        # print("min value of pos_T image ", torch.min(pos_T)) #0

        #assert ((torch.max(inferred_image) <= 1).all() and (torch.min(inferred_image) >= -1).all()).item(), "generated image is not normalised in [-1,1]"
        # print("inferred image dtype is ", inferred_image.dtype)
        # print("pos T image dtype is ", pos_T.dtype)

        # inferred_image = normalise_image(inferred_image)
        # pos_T_orig = pos_T.clone().to(DEVICE)
        # pos_T = normalise_image(pos_T)
        #assert torch.equal(pos_T_orig, pos_T), "pos_T supposed to be normalised a priori to [0,1]"
        
        # print("max value of generated image after normalisation: ", torch.max(inferred_image))
        # print("min value of generated image after normalisation: ", torch.min(inferred_image))

        inferred_image_clone = inferred_image.clone()
        pos_T_clone = pos_T.clone()

        psnr = compute_psnr(inferred_image_clone * 255, pos_T_clone * 255) # we want here uint8 or torch.float32?
        inferred_image_ycbcr = bgr2ycbcr(inferred_image_clone) # * 255)
        pos_T_ycbcr = bgr2ycbcr(pos_T_clone) # * 255)

        # print("max value of generated image AFTER bgr2ycbcr ", torch.max(inferred_image_ycbcr))
        # print("min value of generated image AFTER bgr2ycbcr ", torch.min(inferred_image_ycbcr))


        psnr_y = compute_psnr(inferred_image_ycbcr * 255, pos_T_ycbcr * 255)
        ssim = compute_ssim(inferred_image_ycbcr[:,None,:,:], pos_T_ycbcr[:,None,:,:]) # Expected `preds` and `target` to have BxCxHxW or BxCxDxHxW shape. Got preds: torch.Size([1, 321, 481]) and target: torch.Size([1, 321, 481]).
                
        lpips = compute_lpips(inferred_image_clone, pos_T_clone)

        metrics['psnr'] = psnr.item()
        metrics['psnr_y'] = psnr_y.item()
        metrics['ssim'] = ssim.item()
        metrics['lpips'] = lpips.item()
        
        return pos_0, half_time_image, inferred_image, metrics #psnr.item()

def compute_psnr(inferred_image, pos_T):
        mse = torch.mean((inferred_image - pos_T) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        return psnr

def compute_ssim(inferred_image, pos_T):
    ssim = StructuralSimilarityIndexMeasure(data_range=None).to(DEVICE) # determine data range from the data itself, was 255.0 before
    return ssim(inferred_image, pos_T)

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype # torch.float32
    #print(f"Wile transferring to YCBCR colourscheme image type is {in_img_type}")
    #print(f"image dimension is {img.shape}")
    device = img.device
    
    img = img.to(device=device, dtype=torch.float32)

    if in_img_type != torch.uint8:
        img *= 255.

    # print("max value of generated image in bgr2ycbcr before transform: ", torch.max(img))
    # print("min value of generated image in bgr2ycbcr before transform: ", torch.min(img))    
    # convert
    if only_y:
        coeffs = torch.tensor([24.966, 128.553, 65.481], device=device)
        rlt = torch.tensordot(img, coeffs, dims=([1], [0])) / 255.0 + 16.0 # we squeezed the image in normalise_image()
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

def compute_lpips(generated_image, true_image):
    # here we assume that our images are in [0,1] range
    normalised_generatd_image = normalise_image(generated_image)
    normalised_true_image = normalise_image(true_image)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="mean", normalize=True).to(DEVICE)
    return lpips(normalised_generatd_image, normalised_true_image)

def normalise_image(image, min_max=(0,1)):
    # image is given in torch.float32 
    image = image.float().clamp_(*min_max)  # clamp
    image = (image - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

    # image = (image - torch.min(image)) / torch.max(image) - torch.min(image)
    # image = 1 / (1 + torch.exp(-image))
    return image