# https://github.com/Hammour-steak/GOUB/blob/main/codes/data/__init__.py

"""create dataset and dataloader"""
import logging

import torch
import torch.utils.data
import torch.distributed as dist
import os

# from data_sampler import DistIterSampler

def build_eval_data_loader(dataset_eval, args):
    eval_ds = torch.utils.data.DataLoader(
        dataset_eval, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) 

    return eval_ds

def build_data_loader(dataset_train, dataset_val, args, sampler=None):
    if args.dist:
        world_size = dist.get_world_size()
        num_workers = min(32, os.cpu_count()) # args.num_workers
        assert args.batch_size % world_size == 0
        batch_size = args.batch_size // world_size
        shuffle = False
    else:
        num_workers = min(32, os.cpu_count()) #args.num_workers * len(args.gpu_ids) # args.gpu_ids
        batch_size = args.datasets["train"]["batch_size"]
        shuffle = True

    # one dataset for train: Train_Dataset; mode LQGT   
    train_ds = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers, #args.datasets["train"]["n_workers"],
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
    )

    # another dataset for validation: Val_Dataset; mode LQGT   
    val_ds = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True) # (phase=="val") 

    return train_ds, val_ds    

def create_dataset(args, distortion):#, wandb=None):
    mode = args.mode
    if mode == "LQGT":  # SFTMD
        from imagerec.data.LQGT_dataset import LQGTDataset as D
        dataset = D(args, distortion)#, wandb)
    elif mode == "latent_LQGT":
        from imagerec.data.LQGT_latent_dataset import PairedLatentDataset as D
        dataset = D(args)    
    elif mode == "GT":  # Corrector
        from imagerec.data.GT_dataset import GTDataset as D
        dataset = D(args, distortion)
    else:
        raise NotImplementedError("Dataset [{:s}] is not recognized.".format(mode))

    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, args.name
        )
    )
    return dataset