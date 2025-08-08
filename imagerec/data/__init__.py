# https://github.com/Hammour-steak/GOUB/blob/main/codes/data/__init__.py

"""create dataset and dataloader"""
import logging

import torch
import torch.utils.data


def build_data_loader(dataset_train, dataset_val, args): # , dataset_opt, opt=None, sampler=None
    # if args.dist:
    #     world_size = torch.distributed.get_world_size()
    #     num_workers = args.num_workers
    #     assert args.batch_size % world_size == 0
    #     batch_size = args.batch_size // world_size
    #     shuffle = False
    # else:
    #     num_workers = args.num_workers * args.gpu_ids #len(args.gpu_ids)
    #     batch_size = args.batch_size
    #     shuffle = True

    # one dataset for train: Train_Dataset; mode LQGT   
    train_ds = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.datasets["train"]["batch_size"],
        #shuffle=shuffle,
        num_workers=args.datasets["train"]["n_workers"],
        #sampler=sampler, # for now no transfer of the code for; look at data_sampler
        drop_last=True,
        pin_memory=False,
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