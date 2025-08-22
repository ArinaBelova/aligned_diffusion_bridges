import yaml
import torch
import wandb
import numpy as np
import copy
import math
from torchvision.utils import save_image as torch_save_image


#######
#  Hack for the server to avoid the horrible setup.py script
import os
os.chdir("/data/cluster/users/belova/projects/sbalign/aligned_diffusion_bridges") 
import sys
sys.path.append(os.getcwd())

# for key, value in os.environ.items():
#     print(f"{key}={value}")
#######

from imagerec.data import build_data_loader, create_dataset # what does this return: only train loader or both train and valid loader? -> depends on the provided datasets
from imagerec.models import build_model_from_args

from sbalign.training.epoch_fns import train_epoch_imagerec, test_epoch_imagerec, inference_epoch_imagerec 
from sbalign.training.losses import loss_fn_from_args
from sbalign.training.updaters import get_optimizer, get_scheduler, get_ema
from sbalign.utils.sb_utils import get_diffusivity_schedule
from sbalign.utils.helper import count_parameters
from sbalign.utils.setup import wandb_setup, parse_imagerec_train_args, update_args_from_config
from sbalign.utils.definitions import DEVICE

from types import SimpleNamespace


def train(args, train_loader, val_loader, model, optimizer, scheduler, ema_weights=None, log_dir=None):
    best_val_loss = math.inf
    best_val_inference_value = math.inf if args.inference_goal == 'min' else 0
    best_epoch = 0
    best_val_inference_epoch = 0
    print("scheduler total steps ", scheduler.total_steps)

    print(f"On training start: g_max={args.max_diffusivity}, K={args.K}, H={args.H}, norm={args.norm}")
    g = get_diffusivity_schedule(args.diffusivity_schedule, args.max_diffusivity, H=args.H, K=args.K, norm=args.norm)
    loss_fn = loss_fn_from_args(args)

    logs = {'val_loss': math.inf, "val_inference_rmsd": math.inf}

    for epoch in range(args.n_epochs):
        if epoch > 10:
            args.inference_steps = args.inference_steps
        else:
            args.inference_steps = 10
        print(f"Epoch #{epoch + 1}")
        log_dict = {}

        train_losses = train_epoch_imagerec(
                model=model, loader=train_loader, 
                optimizer=optimizer, scheduler=scheduler, 
                loss_fn=loss_fn,
                grad_clip_value=args.grad_clip_value, 
                ema_weights=ema_weights, dif = g,
                wandb=wandb
            )
        
        # Print training metrics
        print_msg = f"Epoch {epoch+1}: "
        for item, value in train_losses.items():
            if item == "loss":
                print_msg += f"Training Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item}: {np.round(value, 4)} "
        logs.update({'train_' + k: v for k, v in train_losses.items()})
        print(print_msg, flush=True)

        # Load ema parameters into model
        if ema_weights is not None:
            ema_weights.store(model.parameters())
            ema_weights.copy_to(model.parameters())

        # Compute losses on validation set
        val_losses = test_epoch_imagerec(model=model, loader=val_loader, loss_fn=loss_fn, dif=g)

        # Print validation metrics
        print_msg = f"Epoch {epoch+1}: "
        for item, value in val_losses.items():
            if item == "loss":
                print_msg += f"Validation Loss: {np.round(value, 4)} "
            else:
                print_msg += f"{item}: {np.round(value, 4)} "
        logs.update({'val_' + k: v for k, v in val_losses.items()})
        print(print_msg, flush=True)  


        if ema_weights is not None:
            ema_state_dict = copy.deepcopy(model.state_dict() if DEVICE == 'cuda' else model.state_dict())
            ema_weights.restore(model.parameters())

        model_dict = model.state_dict()
        
        # Inference on validation set and PSNR statistics for validation set
        print(f"Started inference on validation set epoch {epoch + 1}", flush=True)
        if args.inference_every > 0 and (epoch + 1) % args.inference_every == 0:
            #model.eval()
            initial_images, cleaned_images_half_time, cleaned_images, metrics = inference_epoch_imagerec(model=model, 
                                                                                                        g=g,
                                                                                                        orig_dataset=val_loader.dataset,
                                                                                                        args=args,
                                                                                                        inference_steps=args.inference_steps)
            
            # Log images to wandb - grouped by image progression and epoch
            for i in range(args.display_on_inference):               
                psnr = metrics["psnr"][i]
                psnr_y = metrics["psnr_y"][i]
                ssim = metrics["ssim"][i]
                lpips = metrics["lpips"][i]

                image_progression = [
                    wandb.Image(initial_images[i], caption="Initial (Corrupted)"),
                    wandb.Image(cleaned_images_half_time[i], caption="Half-time Denoising"), 
                    wandb.Image(cleaned_images[i], caption=f"Final Clean (PSNR: {psnr}), PSNR_Y: {psnr_y}, SSIM: {ssim}, LPIPS: {lpips}" 
                    if (psnr is not None and psnr_y is not None and ssim is not None and lpips is not None) else "Final Clean")
                ]
                
                # Log each image progression as a separate wandb entry
                wandb.log({
                    f"epoch_{epoch+1}_image_{i+1}_progression": image_progression
                })

            if log_dir is not None:
                os.makedirs(os.path.join(log_dir, "val_images_inference"), exist_ok=True)  

            concatenated_tensor = cleaned_images[0].clone()
            shapes_to_save = concatenated_tensor.shape

            for i, cleaned_image in enumerate(cleaned_images):
                base_path = os.path.join(log_dir, "val_images_inference", f"epoch_{epoch+1}", f"image_{i+1}")
                os.makedirs(os.path.dirname(base_path), exist_ok=True)
                # Convert tensor to PIL Image and save
                torch_save_image(cleaned_image, base_path + ".png")

                # Save tensor for the future evaluation
                if i > 0:
                    if cleaned_image.shape != shapes_to_save:
                        cleaned_image = cleaned_image.permute(0,1,3,2)            
                    concatenated_tensor = torch.cat((concatenated_tensor, cleaned_image), dim=0)

            torch.save(concatenated_tensor, os.path.join(log_dir, "val_images_inference", f"epoch_{epoch+1}", "images.pt"))           
                
            # Log the average metrics for all images in this evaluation
            avg_psnr = np.mean(metrics["psnr"])
            avg_psnr_y = np.mean(metrics["psnr_y"])
            avg_ssim = np.mean(metrics["ssim"])
            avg_lpips = np.mean(metrics["lpips"])

            if args.wandb_mode == "online":
                log_dict["avg_psnr"] = avg_psnr
                log_dict["avg_psnr_y"] = avg_psnr_y
                log_dict["avg_ssim"] = avg_ssim
                log_dict["avg_lpips"] = avg_lpips

                wandb.log(log_dict)

            logs.update({'val_inference_avg_psnr': avg_psnr})
            logs.update({'val_inference_avg_psnr_y': avg_psnr_y})
            logs.update({'val_inference_avg_ssim': avg_ssim})
            logs.update({'val_inference_avg_lpips': avg_ssim})

            print(f"Epoch {epoch+1}: Validation Inference Average PSNR: {avg_psnr}", flush=True)
            print(f"Epoch {epoch+1}: Validation Inference Average PSNR_Y: {avg_psnr_y}", flush=True)
            print(f"Epoch {epoch+1}: Validation Inference Average SSIM: {avg_ssim}", flush=True)
            print(f"Epoch {epoch+1}: Validation Inference Average LPIPS: {avg_lpips}", flush=True)

            #model.train()

            # save the best model based on the inference metric if we improved after this epoch:
            if args.inference_metric in logs.keys() and \
                    (args.inference_goal == 'min' and logs[args.inference_metric] < best_val_inference_value or
                    args.inference_goal == 'max' and logs[args.inference_metric] > best_val_inference_value):
                best_val_inference_value = logs[args.inference_metric]
                best_val_inference_epoch = epoch + 1

                if log_dir is not None:
                    model_file = os.path.join(log_dir, f'best_inference_epoch_{epoch + 1}_model.pt')
                    print(f"After best inference, saving model to {model_file}", flush=True)
                    torch.save(model_dict, model_file)

                    if ema_weights is not None:
                        ema_file = os.path.join(log_dir, f'best_ema_inference_epoch_{epoch + 1}_model.pt')
                        print(f"After best inference, saving ema to {ema_file}", flush=True)
                        torch.save(ema_state_dict, ema_file)    

        # Write logs to wandb
        if args.wandb_mode == "online":
            # Logging metrics and losses
            log_dict.update({'train_' + k: v for k, v in train_losses.items()})
            log_dict.update({'val_' + k: v for k, v in val_losses.items()})
            # if args.inference_every > 0 and (epoch + 1) % args.inference_every == 0:
            #     log_dict.update({'val_inference_' + k: v for k, v in inference_metrics.items()})     
            log_dict['current_lr'] = optimizer.param_groups[0]['lr']
            log_dict["step"] = epoch + 1
            wandb.log(log_dict)

        # TODO: this does not make sense why do we save after every epoch a new "best" model?
        # if log_dir is not None:
        #     model_file = os.path.join(log_dir, "best_model.pt")
        #     print(f"After best validation, saving model to {model_file}", flush=True)
        #     torch.save(model_dict, os.path.join(log_dir, 'best_model.pt'))

        #     if ema_weights is not None:
        #         ema_file = os.path.join(log_dir, "best_ema_model.pt")
        #         print(f"After best validation, saving ema to {ema_file}", flush=True)
        #         torch.save(ema_state_dict, os.path.join(log_dir, 'best_ema_model.pt'))
        #     print(flush=True)

        #if scheduler is not None:
            # if args.early_stop_metric in logs:
            #     scheduler.step(logs[args.early_stop_metric])
            # else:
            #     scheduler.step(logs["val_loss"])

        if log_dir is not None:
            print(f"Saving last model to {log_dir}/last_model.pt", flush=True)
            save_dict = {
                'epoch': epoch,
                'model': model_dict,
                'optimizer': optimizer.state_dict(),
            }

            if ema_weights is not None:
                save_dict['ema_weights'] = ema_weights.state_dict()

            torch.save(save_dict, os.path.join(log_dir, 'last_model.pt'))
            print(flush=True)

    #print(f"Best Validation Loss {best_val_loss} on Epoch {best_epoch}", flush=True)
    print(f"Best Inference Metric {best_val_inference_value} on Epoch {best_val_inference_epoch}", flush=True)


def main(cmd_args=None):
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=4)

    # Load args from command line and replace values with those from config
    print(flush=True)
    args = parse_imagerec_train_args(cmd_args=cmd_args)
    print('args from cmd before update',args)
    # TODO: maybe here we want to update the config with options.py and then parse an extended config? 
    
    args = update_args_from_config(args=args)
    print('args after concatenating the config args set: ', args, flush=True)

    # parsing specific arguments to imagerec config:
    from options import parse, dict_to_nonedict

    args = parse(args, is_train=True)
    args = dict_to_nonedict(args)
    print('args after parsing in the imagerec framework ', args)

    # Wandb setup
    wandb_setup(args)
    args.wandb_dir = os.path.dirname(wandb.run.dir)

    print(f"Args: {args}", flush=True)
    print(flush=True)

    print(f"Experiment Name: {args.run_name}", flush=True)
    print(flush=True)

    # Datasets
    #print("Training dataset arguments: ", args.datasets.train)
    dataset_train = create_dataset(SimpleNamespace(**args.datasets["train"]), distortion=args.distortion)#, wandb=wandb)
    dataset_val = create_dataset(SimpleNamespace(**args.datasets["val"]), distortion=args.distortion)#, wandb=wandb)
    train_loader, val_loader = build_data_loader(dataset_train, dataset_val, args) 

    # Model
    model = build_model_from_args(args.network_G) #build_model_from_args(SimpleNamespace(**args["network_G"]))

    n_params = count_parameters(model=model, log_to_wandb=False and args.online)
    print(f"Model with {n_params / (10**6)}M parameters", flush=True)
    model.to(DEVICE)
    print(flush=True)

    # Optimizers
    optimizer = get_optimizer(model=model, optim_name=args.optim_name,
                              lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer=optimizer, args=args, scheduler_name=args.scheduler,
                              scheduler_mode=args.scheduler_mode, factor=0.7,
                              patience=args.scheduler_patience, min_lr=args.lr / 100)
    print('scheduler is ', scheduler)
    ema = get_ema(model=model, decay_rate=args.ema_decay_rate)

    # Recording the full configuration with which we will train
    if args.log_dir is not None:
        log_dir = os.path.join(args.log_dir, args.run_name)
        os.makedirs(log_dir, exist_ok=True)
        config_file = os.path.join(log_dir, "config_train.yml")

        yaml_dump = yaml.dump(args.__dict__)   
        with open(config_file, "w") as f:
            f.write(yaml_dump)

        print(f"Saved model config to {config_file}", flush=True)
        print(flush=True)

    else:
        log_dir = None

    print(f"Training model for {args.n_epochs} epochs...", flush=True)
    train(args=args, train_loader=train_loader, 
          val_loader=val_loader, model=model, optimizer=optimizer,
          scheduler=scheduler, ema_weights=ema, log_dir=log_dir)

    return model


if __name__ == "__main__":
    main()
