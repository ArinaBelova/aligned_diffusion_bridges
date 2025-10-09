import yaml
import torch
import wandb
import numpy as np
import copy
import math
from torchvision.utils import save_image as torch_save_image
import os
os.chdir("/data/cluster/users/belova/projects/sbalign/aligned_diffusion_bridges") 
import sys
sys.path.append(os.getcwd())



from options import parse, dict_to_nonedict
from sbalign.training.epoch_fns import inference_epoch_imagerec 
from imagerec.data import build_eval_data_loader, create_dataset 
from imagerec.models import build_model_from_args
from sbalign.utils.sb_utils import get_diffusivity_schedule
from sbalign.utils.setup import wandb_setup, parse_imagerec_train_args, update_args_from_config
from sbalign.utils.definitions import DEVICE
from types import SimpleNamespace
from pathlib import Path

def main(cmd_args=None):
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4)

    # Load args from command line and replace values with those from config
    print(flush=True)
    args = parse_imagerec_train_args(cmd_args=cmd_args)

    args = update_args_from_config(args=args)
    # parsing specific arguments to imagerec config:

    args = parse(args, is_train=True)
    args = dict_to_nonedict(args)

    # Wandb setup
    wandb_setup(args)
    args.wandb_dir = os.path.dirname(wandb.run.dir)

    dataset_eval = create_dataset(SimpleNamespace(**args.datasets["eval"]), distortion=args.distortion)
    #eval_loader = build_eval_data_loader(dataset_eval, args)

    model = build_model_from_args(args.network_G) 
    weights = torch.load(args.pretrain_model_G)
    
    # There are saved model files with different structured, some of them save metadata also, but here we want purelz the weights:
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:    
        model.load_state_dict(weights) # , weights_only=True 

    if args.log_dir is not None:
        log_dir = os.path.join(args.log_dir, args.run_name)
        os.makedirs(log_dir, exist_ok=True)
        config_file = os.path.join(log_dir, "config_train.yml")

        yaml_dump = yaml.dump(args.__dict__)   
        with open(config_file, "w") as f:
            f.write(yaml_dump)

        print(f"Saved model config to {config_file}", flush=True)
        print(flush=True) 

    print(f"On evaluation start: g_max={args.max_diffusivity}, K={args.K}, H={args.H}, norm={args.norm}, model={args.pretrain_model_G}")
    g = get_diffusivity_schedule(args.diffusivity_schedule, args.max_diffusivity, H=args.H, K=args.K, norm=args.norm)
    initial_images, cleaned_images_half_time, cleaned_images, metrics = inference_epoch_imagerec(model=model, 
                                                                                                g=g,
                                                                                                orig_dataset=dataset_eval,
                                                                                                args=args,
                                                                                                inference_steps=args.inference_steps,
                                                                                                wandb=wandb)

    base_path_to_save_images = os.path.join(log_dir, "eval_images_inference", f"model_{Path(args.pretrain_model_G).stem}")
    if log_dir is not None:
        os.makedirs(base_path_to_save_images, exist_ok=True) 

    log_dict = {}
    concatenated_tensor = cleaned_images[0].clone()
    shapes_to_save = concatenated_tensor.shape

    for i, cleaned_image in enumerate(cleaned_images):
        image_path = os.path.join(base_path_to_save_images, f"image_{i+1}")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        # Convert tensor to PIL Image and save
        torch_save_image(cleaned_image, image_path + ".png")

        psnr = metrics["psnr"][i]
        psnr_y = metrics["psnr_y"][i]
        ssim = metrics["ssim"][i]
        lpips = metrics["lpips"][i]

   

        image_progression = [
            wandb.Image(initial_images[i][0] * 255, caption="Initial (Corrupted)"),
            wandb.Image(cleaned_images_half_time[i][0] * 255, caption="Half-time Denoising"), 
            wandb.Image(cleaned_images[i][0] * 255, caption=f"Final Clean (PSNR: {psnr}), PSNR_Y: {psnr_y}, SSIM: {ssim}, LPIPS: {lpips}" 
            if (psnr is not None and psnr_y is not None and ssim is not None and lpips is not None) else "Final Clean")
        ]
                
        # Log each image progression as a separate wandb entry
        wandb.log({
            f"image_{i+1}_progression": image_progression
        })    

        # Save tensor for the future evaluation
        if i > 0:
            if cleaned_image.shape != shapes_to_save:
                cleaned_image = cleaned_image.permute(0,1,3,2)            
            concatenated_tensor = torch.cat((concatenated_tensor, cleaned_image), dim=0)

    torch.save(concatenated_tensor, os.path.join(base_path_to_save_images, "images.pt"))           
                
    # Log the average metrics for all images in this evaluation
    avg_psnr = np.mean(metrics["psnr"])
    avg_psnr_y = np.mean(metrics["psnr_y"])
    avg_ssim = np.mean(metrics["ssim"])
    avg_lpips = np.mean(metrics["lpips"])
    fid = np.mean(metrics["fid"])

    if args.wandb_mode == "online":
        log_dict["avg_psnr"] = avg_psnr
        log_dict["avg_psnr_y"] = avg_psnr_y
        log_dict["avg_ssim"] = avg_ssim
        log_dict["avg_lpips"] = avg_lpips
        log_dict["fid"] = fid
        wandb.log(log_dict)

    print(f"Validation Inference Average PSNR: {avg_psnr}", flush=True)
    print(f"Validation Inference Average PSNR_Y: {avg_psnr_y}", flush=True)
    print(f"Validation Inference Average SSIM: {avg_ssim}", flush=True)    
    print(f"Validation Inference Average LPIPS: {avg_lpips}", flush=True)    
    print(f"Validation Inference FID: {fid}", flush=True)    


if __name__ == "__main__":
    main()