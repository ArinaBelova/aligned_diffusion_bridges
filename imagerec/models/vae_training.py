"""All code is licensed under the LICENSE file in the root directory of this
repository.
"""
import argparse
import os
from collections import defaultdict
from typing import Optional
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
import wandb
import copy
from contextlib import nullcontext


# This is a stable-diffusion-esque VAE definition with 16 latent channels for
# better reconstructions. NOTE: might make downstream diffusion-tasks a tiny bit 
# harder, but there is no free lunch.
DEFAULT_MODEL_DEF = {
    "act_fn": "silu",
    "block_out_channels": [
        128,
        256,
        512,
        512
    ],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D"
    ],
    "in_channels": 3,
    "latent_channels": 16,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 256,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D"
    ]
}


def pseudo_huber_loss(
    x: torch.Tensor,
    target: torch.Tensor,
    c_huber: Optional[float] = None
) -> torch.Tensor:
    """Computes the Pseudo-Huber loss between two tensors.

    The Pseudo-Huber loss is a smooth approximation of the L2 loss, which is
    less sensitive to outliers.

    Args:
        x (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground truth values.
        c_huber (Optional[float], optional): The smoothness constant. If not
            provided, it is a heuristic based on the number of dimensions in the
            input.

    Returns:
        torch.Tensor: The computed Pseudo-Huber loss.
    """
    if c_huber is None:
        num_dim = np.prod(x.shape[1:])
        c_huber = 0.00054 * math.sqrt(num_dim)
    mse = torch.mean((x - target) ** 2)
    l_pseudo_huber = torch.sqrt(mse + c_huber ** 2) - c_huber
    return l_pseudo_huber


class EMA:
    """Exponential Moving Average (EMA) for model parameters.

    This class maintains an exponentially decaying moving average of the model's
    parameters to stabilize training. The moving average helps smooth out the
    updates and reduce the impact of noisy gradients, leading to more stable
    convergence.

    Args:
        model (nn.Module): The model whose parameters will be tracked by EMA.
        decay (float, optional): The decay rate for the moving average. A higher
            value keeps the average closer to the current parameter values.
            Default is 0.9999.

    Attributes:
        ema_model (torch.nn.Module): A copy of the model with fixed parameters
            that store the EMA.
        decay (float): The decay rate for EMA updates.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        """Initializes the EMA object with a model and decay rate.

        Args:
            model (torch.nn.Module): The model whose parameters will be used for
                EMA.
            decay (float): The decay rate for the moving average.
        """
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)  # don't compute gradients
        self.decay = decay

    def update(self, model: nn.Module) -> None:
        """Updates the EMA model parameters using the given model's current
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be incorporated
                into the EMA.
        """
        with torch.no_grad():
            # updata parameters
            sd_ema = dict(self.ema_model.named_parameters())
            sd_model = dict(model.named_parameters())
            for k in sd_ema.keys():
                sd_ema[k].data.mul_(self.decay).add_(
                    sd_model[k].data, alpha= 1 - self.decay
                )
            # update buffers (e.g., BatchNorm running stats)
            ema_buffers = dict(self.ema_model.named_buffers())
            model_buffers = dict(model.named_buffers())
            for name in ema_buffers.keys():
                ema_buffers[name].copy_(model_buffers[name])


class StandardizeRotation:
    """Standardizes the rotation of a tensor such that the height is always
    greater than or equal to the width.

    If the input tensor has height (H) greater than width (W), it rotates the
    tensor by 90 degrees.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates the input tensor if its height is greater than its width.

        Args:
            x (torch.Tensor): The input tensor to be standardized. It is
                expected to have shape (C, H, W), where C is the number of
                channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: The rotated tensor with the height and width
                standardized.
        """
        _, H, W = x.shape
        if H > W:
            x = torch.rot90(x, k=1, dims=[1, 2])  # Rotate 90 degrees if H > W
        return x


class RandRot180:
    """Randomly rotates the input tensor by 180 degrees (two 90-degree
    rotations) with a probability of 50% to help models generalize better to
    different orientations.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a 180-degree rotation to the input tensor with 50%
        probability.

        Args:
            x (torch.Tensor): The input tensor to be transformed. It is expected
                to have shape (C, H, W), where C is the number of channels, H is
                the height, and W is the width.

        Returns:
            torch.Tensor: The possibly rotated tensor.
        """
        if np.random.uniform(0, 1) > 0.5:
            x = torch.rot90(torch.rot90(x, k=1, dims=[1, 2]), k=1, dims=[1, 2])
        return x


def train(
    data_dir: str,
    output_dir: str = "./checkpoints",
    micro_batch_size: int = 4,
    batch_size: int = 16,
    lr: float = 1e-4,
    warmup_steps: int = 500,
    total_steps: int = 25000,
    beta_kl: float = 1e-6,
    ema_decay: float = 0.9999,
    log_interval: int = 1000,
    load_ckpt: Optional[str] = None,
    device: str = "cuda",
    restart_steps: bool = False,
    finetune: bool = False
) -> None:
    """Trains an AutoencoderKL model on images with optional checkpoint loading
    and EMA tracking.

    This function handles training from scratch or fine-tuning, including data
    loading, transformation, model setup, loss computation (reconstruction and
    KL divergence), gradient accumulation, checkpointing, exponential moving
    average (EMA) tracking, and logging to Weights & Biases (wandb).

    Args:
        data_dir (str): Path to the root directory containing image data. Should
            follow a structure compatible with torchvision's `ImageFolder`.
        output_dir (str, optional): Directory to save model checkpoints.
            Defaults to "./checkpoints".
        micro_batch_size (int, optional): Number of samples per micro-batch used
            in gradient accumulation. Defaults to 4.
        batch_size (int, optional): Total batch size (should be divisible by
            `micro_batch_size`). Defaults to 16.
        lr (float, optional): Learning rate for the AdamW optimizer.
            Defaults to 1e-4.
        warmup_steps (int, optional): Number of steps to linearly increase the
            learning rate at the start. Defaults to 500.
        total_steps (int, optional): Total number of training steps. Defaults to
            25000.
        beta_kl (float, optional): Weight of the KL divergence loss term.
            Defaults to 1e-6.
        ema_decay (float, optional): Decay factor for the exponential moving
            average of model weights. Defaults to 0.9999.
        log_interval (int, optional): Step interval at which to log losses and
            visualizations to wandb. Defaults to 1000.
        load_ckpt (Optional[str], optional): Path to a checkpoint file to resume
            training from. If None, training starts from scratch.
            Defaults to None.
        device (str, optional): Device to use for training ("cuda" or "cpu").
            Defaults to "cuda".
        restart_steps (bool, optional): Whether to reset the training step count
            when loading a checkpoint. Defaults to False.
        finetune (bool, optional): If True, only decoder-related layers are
            updated, and encoder is frozen. Defaults to False.
    """
    assert batch_size >= micro_batch_size, "batch smaller than micro batch"
    # feel free to rename project / change init to your preferences
    wandb.init(project="vae-training", config=locals())

    # Standardizes input aspect ratio and applies some simple permutations
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Random Permutations
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        RandRot180(),
        # Enforcing landscape and mapping 321 x 481 to 320 x 480 pixels
        StandardizeRotation(),
        transforms.Resize(size=(320, 480))
    ])

    # Simply loads all images from sub-directories of respective root directory
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=micro_batch_size, shuffle=True, num_workers=4
    )

    # Set up the model; @Arina you can make this model-def. modular if you want
    model_definition = DEFAULT_MODEL_DEF
    model = AutoencoderKL(**model_definition)
    ema = EMA(model, decay=ema_decay)

    # Load checkpoint if available
    if load_ckpt is not None:
        ckpt = torch.load(load_ckpt)
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
        ema.ema_model.load_state_dict(ckpt['ema'])
        ema.ema_model = ema.ema_model.to(device)
        # regular training / pretraining parameters for optimizer
        if not finetune:
            params = model.parameters()
        # decoder finetuning parameters for optimizer
        else:
            if model.post_quant_conv is not None:
                layers = [model.decoder, model.post_quant_conv]
            else:
                layers = [model.decoder]
            params = list(map(lambda x: {'params': list(x.parameters())},
                              layers))
        # still no official lion in PyTorch, so AdamW it is :(
        optimizer = torch.optim.AdamW(params, lr=lr)
        if not finetune:
            optimizer.load_state_dict(ckpt['opt_state'])
        # continuing or restarting training loop, lr-schedule & logging
        step = ckpt['step'] if not restart_steps else 1
        del ckpt  # freeing some unused / redundant memory
    else:
        model = model.to(device)
        ema.ema_model = ema.ema_model.to(device)
        # finetuning of a from-scratch model is not considered an option
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        step = 1

    # cosine decay of lr after some warmup steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # keeping track of checkpoints
    checkpoint_queue = []

    # batch accumulation
    acc_steps = batch_size // micro_batch_size
    micro_step = 0

    # keeping track of losses
    running_losses = defaultdict(list)

    # context of the encoder (e.g. faster no-grad for finetuning)
    encode_context = torch.no_grad if finetune else nullcontext

    # Training loop
    optimizer.zero_grad()
    while step < total_steps:
        for imgs, _ in dataloader:
            micro_step += 1

            # moving to accelerator
            imgs = imgs.to(device)

            # project to posterior distribution in latent space
            with encode_context():
                posterior = model.encode(imgs).latent_dist
                # sample from posterior distribution
                latents = posterior.sample()
            # reconstruct image from latent posterior
            recons = model.decode(latents).sample

            # calculating the loss
            recon_loss = pseudo_huber_loss(recons, imgs)
            if not finetune:
                # apply posterior regularization (KL-loss term)
                kl_loss = posterior.kl().mean()
            else:
                # encoder & respective latent space is frozen during finetuning
                kl_loss = torch.zeros(1).to(device)
            beta_schedule = 1.0  # step / total_steps
            loss = recon_loss + beta_schedule * beta_kl * kl_loss
            loss = loss / acc_steps

            # optimization step + EMA update
            loss.backward()
            if micro_step % acc_steps == 0:
                optimizer.step()
                scheduler.step()
                ema.update(model)
                optimizer.zero_grad()
                # important; increment steps w.r.t. optimization updates
                step += 1

            # keeping track of losses
            running_losses["recon_loss"].append(
                recon_loss.detach().cpu().item()
            )
            running_losses["kl_loss"].append(
                kl_loss.detach().cpu().item()
            )

            if step % log_interval == 0 and micro_step % acc_steps == 0:
                # logging
                with torch.no_grad():  # EMA samples
                    ema.ema_model = ema.ema_model.eval()
                    ema_recons = ema.ema_model(imgs).sample
                    ema.ema_model = ema.ema_model.train()  # unsure if needed
                # move samples to cpu & clamp to [0, 1]
                _imgs, _recons, ema_recons = map(
                    lambda x: x.type(torch.float32).detach().cpu().clamp(0, 1),
                    (imgs, recons, ema_recons)
                )
                # logging all data in a single wandb request
                wandb.log(
                    {
                        **{k: sum(v)/len(v) for k, v in running_losses.items()},
                        "inputs": wandb.Image(
                            make_grid(_imgs, nrow=2)
                        ),
                        "reconstructions": wandb.Image(
                            make_grid(_recons, nrow=2)
                        ),
                        "EMA reconstructions": wandb.Image(
                            make_grid(ema_recons, nrow=2)
                        )
                    },
                    step=step,
                    commit=True
                )
                # zeroing the bins for next logs
                running_losses = defaultdict(list)

                # checkpointing
                ckpt_path = os.path.join(
                    output_dir, f"ckpt_step_{step}.pt"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.ema_model.state_dict(),
                        "opt_state": optimizer.state_dict(),
                        "step": step,
                    },
                    ckpt_path
                )
                checkpoint_queue.append(ckpt_path)

                # we only keep the latest three checkpoints
                while len(checkpoint_queue) > 3:
                    old_ckpt = checkpoint_queue.pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

                # breaking criteria within epoch
                if step >= total_steps:
                    break


if __name__ == "__main__":
    # Minimal command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--total_steps", type=int, default=25000)
    parser.add_argument("--beta_kl", type=float, default=1e-6)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--restart_steps", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=False)
    args = parser.parse_args()

    # I/O sanity
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    train(**vars(args))
