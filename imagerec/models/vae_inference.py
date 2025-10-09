import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from diffusers.models import AutoencoderKL
from typing import List, Tuple, Callable
from collections import defaultdict
from .vae_training import DEFAULT_MODEL_DEF, StandardizeRotation


def to_float(arr: np.ndarray) -> torch.Tensor:
    """Converts a NumPy array (H, W, C) image to a float32 Torch tensor
    (C, H, W) in [0, 1].

    Args:
        arr (np.ndarray): Image as a NumPy array with shape (H, W, C) and
            dtype convertible to float.

    Returns:
        torch.Tensor: Image tensor with shape (C, H, W) and dtype
        torch.float32, values in [0, 1].
    """
    return torch.from_numpy(
        arr.astype(np.float32).transpose(2, 0, 1) / 255.
    )


def load_imgs(
    root: str,
    shape: Tuple[int, ...],
    tensor_fn: Callable = to_float,
    suffixes: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
) -> List[torch.Tensor]:
    """Loads and resizes all images with specified suffixes from a directory
    and its subdirectories.

    Args:
        root (str): Root directory containing images.
        shape (Tuple[int, ...]): Image shape as (width, height).
        tensor_fn (Callable, optional): Function to convert a NumPy array to
            a Torch tensor. Defaults to `to_float`.
        suffixes (Tuple[str, ...], optional): Allowed file extensions for
            image loading. Defaults to ('.jpg', '.jpeg', '.png').

    Returns:
        list[torch.Tensor]: List of loaded and resized image tensors.
    """
    assert len(shape) == 2, 'shape should contain width & height'
    imgs = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if any(f.lower().endswith(sfx) for sfx in suffixes):
                image = Image.open(os.path.join(r, f))
                resized_image = image.resize(shape, Image.LANCZOS)
                imgs.append(tensor_fn(np.array(resized_image)))
    return imgs


@torch.no_grad()
def encode(
    vae: AutoencoderKL,
    x: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Encodes images into latent representations using an AutoencoderKL.

    Args:
        vae (AutoencoderKL): The trained VAE model.
        x (torch.Tensor): Input image tensor of shape (N, C, H, W).
        device (torch.device, optional): Device for computation. Defaults to
            CPU.

    Returns:
        torch.Tensor: Latent tensor of shape
        (N, latent_channels, latent_height, latent_width).
    """
    vae = vae.to(device).eval()
    #print("X SHAPE IS ", x.shape, flush=True)
    latents = vae.encode(x.to(device)).latent_dist.mode()
    return latents


@torch.no_grad()
def decode(
    vae: AutoencoderKL,
    latents: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Decodes latent vectors back into images using an AutoencoderKL.

    Args:
        vae (AutoencoderKL): The trained VAE model.
        latents (torch.Tensor): Latent tensor of shape
            (N, latent_channels, latent_height, latent_width).
        device (torch.device, optional): Device for computation. Defaults to
            CPU.

    Returns:
        torch.Tensor: Reconstructed image tensor of shape (N, C, H, W).
    """
    vae = vae.to(device).eval()
    preds = vae.decode(latents.to(device)).sample
    return preds


def load_latents(pth: str) -> torch.Tensor:
    """Loads latent tensors from a `.npy` file.

    Args:
        pth (str): Path to the `.npy` file containing latent vectors.

    Returns:
        torch.Tensor: Loaded latents as a float32 tensor.
    """
    return torch.from_numpy(np.load(pth).astype(np.float32))


def save_latents(latents: torch.Tensor, pth: str) -> None:
    """Saves latent tensors to a `.npy` file.

    Args:
        latents (torch.Tensor): Latent tensor to save.
        pth (str): Path where the `.npy` file will be saved.
    """
    np.save(pth, latents.detach().cpu().numpy())


@torch.no_grad()
def decode_latent_dataset(
    vae: AutoencoderKL,
    latents: torch.Tensor,
    batch_size: int = 8,
    verbose: bool = False,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Decodes a dataset of latent vectors into images in batches.

    Args:
        vae (AutoencoderKL): The trained VAE model.
        latents (torch.Tensor): Latent dataset tensor of shape
            (N, latent_channels, H, W).
        batch_size (int, optional): Number of samples per decoding batch.
            Defaults to 8.
        verbose (bool, optional): If True, displays a progress bar.
            Defaults to False.
        device (torch.device, optional): Device for computation.
            Defaults to CPU.

    Returns:
        torch.Tensor: Tensor of reconstructed images with shape
        (N, C, H, W).
    """
    reconstructions = []
    with torch.no_grad():
        vae = vae.cuda().eval()
        batch = []
        iterator = tqdm(latents) if verbose else latents
        for i, img in enumerate(iterator):
            batch.append(img)
            if (i + 1) % batch_size == 0 or (i + 1) == len(latents):
                x = torch.stack(batch).cuda()
                rec = decode(vae, x, device=device).cpu()
                reconstructions += list(rec)
                batch = []
    return torch.stack(reconstructions)


@torch.no_grad()
def encode_latent_dataset(
    vae: AutoencoderKL,
    imgs: torch.Tensor,
    batch_size: int = 32,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """Encodes a dataset of images into latents in batches.

    Args:
        vae (AutoencoderKL): The trained VAE model.
        imgs (torch.Tensor): Image dataset tensor of shape (N, C, H, W).
        batch_size (int, optional): Number of samples per encoding batch.
            Defaults to 32.
        device (torch.device, optional): Device for computation.
            Defaults to CPU.

    Returns:
        torch.Tensor: Tensor of latents with shape
        (N, latent_channels, H, W).
    """
    with torch.no_grad():
        vae = vae.cuda().eval()
        db_latents = []
        batch = []
        for i, img in enumerate(tqdm(imgs)):
            batch.append(img)
            if (i + 1) % batch_size == 0 or (i + 1) == len(imgs):
                x = torch.stack(batch).cuda()
                latents = encode(vae, x, device=device).cpu()
                db_latents += list(latents)
                batch = []
    return torch.stack(db_latents)


@torch.no_grad()
def inference(
    data_dir: str,
    output_dir: str,
    micro_batch_size: int,
    load_ckpt: str,
    device: str
) -> None:
    """Performs inference using a trained AutoencoderKL model and saves outputs.

    This function loads a dataset, applies preprocessing, and performs inference
    using both the trained model and its Exponential Moving Average (EMA)
    version. For each image, it extracts latent representations and
    reconstructions, then saves them (along with the original inputs) as NumPy
    arrays for downstream use.

    Args:
        data_dir (str): Path to the root directory containing image data.
            Must follow a structure compatible with torchvision's `ImageFolder`.
        output_dir (str): Directory where the output `.npy` files will be saved.
        micro_batch_size (int): Number of samples per inference batch.
        load_ckpt (str): Path to the model checkpoint file containing both model
            and EMA weights.
        device (str): Device to run inference on ("cuda" or "cpu").

    Returns:
        None: All results are saved to disk; the function does not return
        anything.
    """
    # Standardizes input aspect ratio
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Enforcing landscape and mapping 321 x 481 to 320 x 480 pixels
        StandardizeRotation(),
        transforms.Resize(size=(320, 480))
    ])

    # Simply loads all images from sub-directories of respective root directory
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=micro_batch_size, shuffle=False, num_workers=4
    )
    # Set up the model; @Arina you can make this model-def. modular if you want
    model_definition = DEFAULT_MODEL_DEF
    model = AutoencoderKL(**model_definition)
    ema_model = AutoencoderKL(**model_definition)

    # load weights
    ckpt = torch.load(load_ckpt)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    ema_model.load_state_dict(ckpt['ema'])
    ema_model = ema_model.to(device).eval()
    vaes, vae_names = (model, ema_model), ('model', 'ema_model')

    # collect latent representations, reconstructions & reference inputs
    latents = defaultdict(list)
    reconstructions = defaultdict(list)
    inputs = defaultdict(list)

    for imgs, _ in tqdm(dataloader):
        imgs = imgs.to(device)

        for vae, name in zip(vaes, vae_names):
            lats = encode(vae, imgs, device)
            recs = decode(vae, lats, device)
            latents[name].append(lats.cpu())
            reconstructions[name].append(recs.cpu())
            inputs[name].append(imgs.cpu())

    # save to numpy arrays
    for name in vae_names:
        f_lat = os.path.join(output_dir, f'latents_{name}.npy')
        f_rec = os.path.join(output_dir, f'reconstructions_{name}.npy')
        f_inp = os.path.join(output_dir, f'inputs_{name}.npy')
        np.save(f_lat, torch.cat(latents[name], dim=0).numpy())
        np.save(f_rec, torch.cat(reconstructions[name], dim=0).numpy())
        np.save(f_inp, torch.cat(inputs[name], dim=0).numpy())


if __name__ == "__main__":
    # Minimal command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # I/O sanity
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    inference(**vars(args))
