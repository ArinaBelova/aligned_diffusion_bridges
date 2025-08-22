import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
import imagerec.data.util as util

# try:
#     sys.path.append("..")
#     import imagerec.data.util as util
# except ImportError:
#     pass


class LQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, args, distortion):#, wandb=None):
        super().__init__()
        self.LR_paths, self.GT_paths = None, None
        self.LR_env, self.GT_env = None, None  # environment for lmdb
        #self.LR_size, self.GT_size = args.LR_size, args.GT_size
        self.args = args
        self.distortion = distortion
        #self.wandb = wandb
        # read image list from lmdb or image files
        if self.args.data_type == "lmdb":
            self.LR_paths, self.LR_sizes = util.get_image_paths(
                self.args.data_type, self.args.dataroot_LQ
            )
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                self.args.data_type, self.args.dataroot_GT
            )
        elif self.args.data_type == "img":
            self.LR_paths = util.get_image_paths(
                self.args.data_type, self.args.dataroot_LQ
            )  # LR list
            self.GT_paths = util.get_image_paths(
                self.args.data_type, self.args.dataroot_GT
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.args.dataroot_GT,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.LR_env = lmdb.open(
            self.args.dataroot_LQ,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.args.data_type == "lmdb":
            if (self.GT_env is None) or (self.LR_env is None):
                self._init_lmdb()

        GT_path, LR_path = None, None
        scale = self.args.scale if self.args.scale else 1
        GT_size = self.args.GT_size
        LR_size = self.args.LR_size

        # get GT image
        GT_path = self.GT_paths[index]
        if self.args.data_type == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        # TODO: decide how we do this switch between the phases!
        if self.args.phase != "train":
            img_GT = util.modcrop(img_GT, scale)

        # get LR image
        if self.LR_paths:  # LR exist
            LR_path = self.LR_paths[index]
            if self.args.data_type == "lmdb":
                resolution = [int(s) for s in self.LR_sizes[index].split("_")]
            else:
                resolution = None
            img_LR = util.read_img(self.LR_env, LR_path, resolution)
        # geenrate a LR image for inpainting:    
        elif self.distortion == "inpaint":
            # # Log original image to wandb
            # if self.wandb and self.wandb.run is not None:
            #     # Convert BGR to RGB for visualization
            #     img_GT_rgb = cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB)
            #     # Convert from float [0,1] to uint8 [0,255]
            #     img_GT_rgb = (img_GT_rgb * 255).astype(np.uint8)
            #     self.wandb.log({
            #         "original_image": self.wandb.Image(
            #             img_GT_rgb,
            #             caption="Original Image Before Inpainting"
            #         )
            #     })
            img_LR = util.mask_to_fixed(img_GT) #, wandb=self.wandb) # not normalised image yet
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.args.phase == "train":
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(
                    np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR
                )
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LR = util.imresize(img_GT, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        if self.args.phase == "train":
            if self.distortion == "derain":
                # For now we don't do any cropping in inpainting
                H, W, C = img_LR.shape
                assert LR_size == GT_size // scale, "GT size does not match LR size"

                # randomly crop
                rnd_h = random.randint(0, max(0, H - LR_size))
                rnd_w = random.randint(0, max(0, W - LR_size))
                img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[
                    rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
                ]

            # augmentation - flip, rotate
            img_LR, img_GT = util.augment(
                [img_LR, img_GT],
                self.args.use_flip,
                self.args.use_rot,
                self.args.mode,
                self.args.use_swap,
            )

        # For now we don't do any cropping neither in training nor in validation
        # elif LR_size is not None:
        #     H, W, C = img_LR.shape
        #     assert LR_size == GT_size // scale, "GT size does not match LR size"

        #     if LR_size < H and LR_size < W:
        #         # center crop
        #         rnd_h = H // 2 - LR_size//2
        #         rnd_w = W // 2 - LR_size//2
        #         img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
        #         rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
        #         img_GT = img_GT[
        #             rnd_h_GT : rnd_h_GT + GT_size, rnd_w_GT : rnd_w_GT + GT_size, :
        #         ]

        # change color space if necessary
        #if self.args.color:
        if getattr(self.args, 'color', False):
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.args.color, [img_LR])[
                0
            ]  # TODO during val no definition
            img_GT = util.channel_convert(img_GT.shape[2], self.args.color, [img_GT])[
                0
            ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()

        if LR_path is None:
            LR_path = GT_path
        
        # Here images are torch.float32 format
        # print("LR image dtype ", img_LR.dtype)
        # print("max value of LR image: ", torch.max(img_LR),flush=True)
        # print("min value of LR image: ", torch.min(img_LR),flush=True)
        # assert ((torch.max(img_LR) <= 1).all() and (torch.min(img_LR) >= -1).all()).item(), "img_LR image is not normalised in [-1,1]"
        # print("GT image dtype ", img_GT.dtype)
        # print("max value of GT image: ", torch.max(img_GT),flush=True)
        # print("min value of GT image: ", torch.min(img_GT),flush=True)
        # assert ((torch.max(img_GT) <= 1).all() and (torch.min(img_GT) >= -1).all()).item(), "img_GT image is not normalised in [-1,1]"

        #return {"LQ": img_LR, "GT": img_GT, "LQ_path": LR_path, "GT_path": GT_path}
        if self.args.only_get_LQ:
            if (self.distortion == "derain") and (self.args.common_shape != img_LR.shape[1]):
                return torch.permute(img_LR, (0, 2, 1))
            else:
                return img_LR
        elif self.args.only_get_GT:
            if (self.distortion == "derain") and (self.args.common_shape != img_GT.shape[1]) :
                return torch.permute(img_GT, (0, 2, 1))
            else:
                return img_GT
        
        return {"LQ": img_LR, "GT": img_GT}
    
    def __len__(self):
        return len(self.GT_paths)