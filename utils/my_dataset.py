import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class My_dataset(Dataset):
    def __init__(self, ikDataset, size, mapping={}, transforms=None):
        self.data = ikDataset
        self.size = size
        self.mapping = mapping
        self.transforms = transforms

    def __len__(self):
        return len(self.data["images"])

    @classmethod
    def preprocess(cls, pil_img, size_h_w, is_mask):
        newW = size_h_w
        newH = size_h_w
        assert newW > 0 and newH > 0, 'size is too small'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        return img_ndarray.transpose((2, 0, 1))

    @classmethod
    def load(cls, filename):
        return Image.open(filename).convert("RGB")

    @classmethod
    def mask_to_class(cls, mask: np.ndarray, mapping):
        mask_ = np.zeros((mask.shape[1], mask.shape[2]))
        for k in mapping:
            k_array = np.array(k)
            # to have the same dim as the mask
            k_array = np.expand_dims(k_array, axis=(1, 2))
            # Extract each class indexes
            idx = (mask == k_array)
            # check there is 3 channels
            validx = (idx.sum(0) == 3)
            mask_[validx] = mapping[k]
        return mask_


    def __getitem__(self, idx):

        name = self.data["images"][idx]
        assert "semantic_seg_masks_file" in name, "semantic_seg_masks_file not found !"
        # load the image from disk with the load function
        img = self.load(name["filename"])
        mask = self.load(name["semantic_seg_masks_file"])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'


        # apply transformation only to train set
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            img = self.transforms(img)
            mask = self.transforms(mask)

        img = self.preprocess(img, self.size, is_mask=False)
        mask = self.preprocess(mask, self.size, is_mask=True)

        # mapping the class colors
        mask = self.mask_to_class(mask, self.mapping)
        # return a dict of the image and its mask
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }