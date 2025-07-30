import os
import argparse
import json
import yaml
from argparse import FileType

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import os
os.chdir("/home/fe/belova/projects/bridges/aligned_diffusion_bridges") 
import sys
sys.path.append(os.getcwd())

from sbalign.data import ListDataset
from sbalign.utils.ops import to_numpy
from sbalign.utils.sb_utils import get_diffusivity_schedule
from sbalign.utils.definitions import DEVICE

from proteins.conf.conf_engine import ConfEngine
from proteins.conf.models import build_model_from_args as build_conf_model
from proteins.baselines.conf.egnn_model import build_model_from_args as build_baseline_model


def prepare_inference_setup(args):

    with open(f'{args.log_dir}/{args.run_name}/config_train.yml') as f:
        model_args = argparse.Namespace(**yaml.full_load(f))


    # Model
    if args.method == "sbalign":
        model = build_conf_model(model_args)
    elif args.method == "baseline":
        model = build_baseline_model(model_args)

    model_dict = torch.load(f"{args.log_dir}/{args.run_name}/{args.model_name}",
                             map_location='cpu')
    if args.model_name == "last_model.pt":
        model.load_state_dict(model_dict["model"])
    else:                                 
        model.load_state_dict(model_dict)
    model.to(DEVICE)

    print('In prepare inference setup',args.data_dir)
    # Data Loader
    resolution = model_args.resolution
    processed_dir = f"{args.data_dir}/processed/{args.dataset}/resolution={resolution}"
    if model_args.center_conformations:
        processed_dir += f"_centered_conf"

    with open(f"{args.data_dir}/raw/{args.dataset}/splits.json", "r") as f:
        splits = json.load(f)

    pdb_ids = splits['test']
    pdb_ids = [pdb_id for pdb_id in pdb_ids
            if os.path.exists(f"{processed_dir}/{pdb_id}.pt")]

    print(f"Inference on {len(pdb_ids)} proteins", flush=True)
    print(flush=True)

    loader = DataLoader(
        ListDataset(
            processed_dir=processed_dir,
            id_list=pdb_ids
        )
    )

    # Inference Engine
    if args.method == "sbalign":
        # K = model_args.K
        # H = model_args.H
        # H=0.5
        # K=0
        g_fn = get_diffusivity_schedule(schedule=model_args.diffusivity_schedule, 
                                    g_max=model_args.max_diffusivity,
                                    K=model_args.K,
                                    H=model_args.H,
                                    norm=model_args.norm)

        engine = ConfEngine(
            samples_per_protein=args.n_samples,
            inference_steps=args.inference_steps,
            model=model, g_fn=g_fn
        )
    else:
        engine = None

    return loader, model, engine


def run_inference_sbalign(data, engine):
    _, metrics = engine.generate_conformations(data, apply_mean=False)
    return metrics


def run_inference_baseline(data, model):

    def rmsd_fn(y_pred, y_true):
        se = (y_pred - y_true)**2
        mse = se.sum(axis=1).mean()
        return np.sqrt(mse)

    data = data.to(DEVICE)
    pos_pred = model(data)

    rmsd = rmsd_fn(to_numpy(pos_pred), to_numpy(data.pos_T))
    metrics = {'rmsd': [rmsd]}
    return metrics


def print_statistics(args, metrics):
    # RMSD associated statistics
    print(flush=True)
    rmsds = []
    delta_rmsd  = []
    for pdb_dict in metrics:
        rmsds.extend(pdb_dict['rmsd'])
        delta_rmsd.extend((np.array(pdb_dict['init_rmsd']) - np.array(pdb_dict['rmsd'])).tolist())
    
    mean_rmsd = np.round(np.mean(rmsds), 4)
    median_rmsd = np.round(np.median(rmsds), 4)
    std_rmsd = np.round(np.std(rmsds), 4)
    print(f"RMSD ({args.n_samples}): Mean={mean_rmsd} | Std={std_rmsd} | Median={median_rmsd}")

    mean_delta_rmsd = np.round(np.mean(delta_rmsd), 4)
    median_delta_rmsd = np.round(np.median(delta_rmsd), 4)
    std_delta_rmsd = np.round(np.std(delta_rmsd), 4)
    print(f"DELTA RMSD (data - model) ({args.n_samples}): Mean={mean_delta_rmsd} | Std={std_delta_rmsd} | Median={median_delta_rmsd}")
    
    rmsd_stats = "RMSD (% <) | "
    for threshold in [2.0, 5.0, 10.0]:
        rmsd_less_than_threshold = (np.asarray(rmsds) < threshold).mean()
        rmsd_stats += f"{threshold}= {np.round(rmsd_less_than_threshold, 4)} | "
    print(rmsd_stats, flush=True)
    print(flush=True)

    delta_rmsd_stats = "DELTA RMSD (% <) | "
    for threshold in [2.0, 5.0, 10.0]:
        delta_rmsd_less_than_threshold = (np.abs(np.asarray(delta_rmsd)) < np.abs(threshold)).mean()
        delta_rmsd_stats += f"{threshold}= {np.round(delta_rmsd_less_than_threshold, 4)} | "
    print(delta_rmsd_stats, flush=True)

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
    
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--inverse_problem", type=str2bool)

    parser.add_argument("--method", type=str)

    parser.add_argument("--dataset", default="d3pm", type=str)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--inference_steps", default=100, type=int)
    parser.add_argument("--max_diffusivity", default=1.0, type=float)

    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    
    loader, model, engine = prepare_inference_setup(args=args)

    metrics = []
    for data in loader:
        if args.inverse_problem:
            data_pos_0_copy = data.pos_0.clone()
            data.pos_0 = data.pos_T
            data.pos_T = data_pos_0_copy

        if "conf_id" in data:
            if len(data['conf_id']) > 0:
                conf_id = data['conf_id'][0]
                print(f"Running inference for {conf_id}", flush=True) 

        if args.method == "sbalign":
            pdb_metrics = run_inference_sbalign(data, engine)
        elif args.method == "baseline":
            pdb_metrics = run_inference_baseline(data, model)

        metrics.append(pdb_metrics)

    print_statistics(args=args, metrics=metrics)


if __name__ == "__main__":
    main()
