from argparse import ArgumentParser
from scipy.stats import wasserstein_distance
import torch
import pandas as pd

from reproducibility.reproducibility import *


def str2bool(s):
    # s is already bool
    if isinstance(s, bool):
        return s
    # s is string repr. of bool
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    # s is something else
    else:
        return s


def list_of_ints(arg):
    return list(map(int, arg.split(",")))

def list_of_floats(arg):
    return list(map(float, arg.split(",")))


def args():
    ap = ArgumentParser()
    ap.add_argument("--dataset", type=str, default='moon',choices=['moon','diagonal_matching','diagonal_matching_inverse']) 
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--num_aug", type=list_of_ints, default=[0, 1, 2, 3, 4, 5]) 
    ap.add_argument("--hurst", type=list_of_floats, default=[0.5,0.9,0.8,0.7,0.6,0.4,0.3,0.2,0.1]) 
    ap.add_argument("--n_epoch", type=int, default=1) 
    ap.add_argument("--samples_num", type=int, default=100) 
    
    ap.add_argument("--norm", type=str2bool, default=True, help='whether to normalize the terminal variance of the diffusion process across all values of H')
    return ap

parser = args()
args = parser.parse_args()

dataset = args.dataset
runs = args.runs
num_aug = args.num_aug
hurst = args.hurst
n_epoch = args.n_epoch
samples_num = args.samples_num
norm = args.norm

WSD_mean = torch.zeros(len(num_aug),len(hurst))
WSD_std = torch.zeros(len(num_aug),len(hurst))

for i,K in enumerate(num_aug):
    for j,H in enumerate(hurst):
        if K==0 and H!=0.5:
            continue
        wsd1 = torch.zeros(runs)
        wsd2 = torch.zeros(runs)
        for n in range(runs):
            if dataset == 'moon':
                try:
                    AlignExperiment.run(f"--dataset=moon --h_dim=64  --n_layers=2  --n_epochs={n_epoch}  --reg_weight=1.  --timestep_emb_dim=32  --in_dim=2 --out_dim=2  --diffusivity_schedule=fbb  --max_diffusivity=.7  --H={H}  --K={K} --norm={norm} --use_drift_in_doobs=True  --activation=silu").save("moon_silu")
                except ValueError:
                    continue
            elif dataset == 'diagonal_matching':
                try:
                    AlignExperiment.run(f"--dataset=diagonal_matching  --h_dim=32  --n_layers=3  --n_epochs={n_epoch}   --reg_weight=1.  --timestep_emb_dim=32  --diffusivity_schedule=fbb  --max_diffusivity=1. --H={H} --K={K} --norm={norm} --use_drift_in_doobs=True  --activation=selu").save("t_dataset")
                except ValueError:
                    continue
            elif dataset == 'diagonal_matching_inverse':
                try:
                    AlignExperiment.run(f"--dataset=diagonal_matching_inverse  --h_dim=32  --n_layers=3  --n_epochs={n_epoch}  --reg_weight=1.  --timestep_emb_dim=32  --diffusivity_schedule=fbb  --max_diffusivity=1. --H={H} --K={K} --norm={norm} --use_drift_in_doobs=True  --activation=selu").save("t_dataset_inverse")
                except ValueError:
                    continue

            sampler = AlignExperiment.load("moon_silu")
            samples = sampler.sample(samples_num=samples_num, trials_num=7)
            if len(samples.shape) == 4:
                samples = samples[:,:,:,0]

            marginals = sampler.get_marginals(samples_num=samples_num)
            wsd1[n] = wasserstein_distance(samples[-1,:,0],marginals['final'][:,0])
            wsd2[n] = wasserstein_distance(samples[-1,:,1],marginals['final'][:,1])

        mean_wsd1 = torch.mean(wsd1)
        mean_wsd2 = torch.mean(wsd2)
        WSD_mean[i,j] = (mean_wsd1 + mean_wsd2)/2

        std_wsd1 = torch.std(wsd1)
        std_wsd2 = torch.std(wsd2)
        WSD_std[i,j] = (std_wsd1 + std_wsd2)/2

        df_mean = pd.DataFrame(WSD_mean.numpy(), columns=hurst)

        df_std = pd.DataFrame(WSD_std.numpy(), columns=hurst)

        # Save the DataFrame to a CSV file
        csv_mean = f"mean_{dataset}_norm{norm}_K{len(num_aug)}_H{len(hurst)}_runs{runs}_samples{samples_num}_epochs{n_epoch}.csv"
        csv_std = f"std_{dataset}_norm{norm}_K{len(num_aug)}_H{len(hurst)}_runs{runs}_samples{samples_num}_epochs{n_epoch}.csv"

        df_mean.to_csv(csv_mean, index=False)
        df_std.to_csv(csv_std, index=False)





