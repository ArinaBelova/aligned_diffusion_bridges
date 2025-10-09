from os import cpu_count
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


NUM_MAX_WORKERS = 1


class PairedLatentDataset(Dataset):

    @torch.no_grad()
    def __init__(
        self,
        args: dict
    ) -> None:
        super().__init__()
        self.latents_lq = torch.load(args.path_to_latents_lq)
        self.latents_gt = torch.load(args.path_to_latents_gt)
        assert self.latents_lq.shape == self.latents_gt.shape, (
            f'shape mismatch of latents LQ with shape {self.latents_lq.shape}'
            f'and latents GT with shape {self.latents_gt.shape}'
        )

    @torch.no_grad()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x / 0.745

    @torch.no_grad()
    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.745

    def dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(cpu_count(), NUM_MAX_WORKERS)
        )

    def __get(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latents_lq[idx], self.latents_gt[idx]

    def __len__(self) -> int:
        return len(self.latents_lq)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lat_LQ, lat_GT = self.__get(idx)
        lat_LQ = self.preprocess(lat_LQ)
        lat_GT = self.preprocess(lat_GT)

        return {"LQ": lat_LQ, "GT": lat_GT}