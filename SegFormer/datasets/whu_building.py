import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pycocotools.coco import COCO


class WHUBuilding(Dataset):
    """
    WHU Building Rooftop Dataset
    - Single foreground class: building (id = 1 in COCO)
    - Labels: 0 = background, 1 = building
    """

    CLASSES = ['background', 'building']

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.root = Path(root)
        self.split = split

        if split == 'train':
            img_dir = self.root / 'train'
            ann_file = self.root / 'annotation' / 'train.json'
        else:
            img_dir = self.root / 'val'
            ann_file = self.root / 'annotation' / 'validation.json'

        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        self.coco = COCO(str(ann_file))
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        if len(self.ids) == 0:
            raise RuntimeError(f"No images found for split={split} in {img_dir}")

        print(f"WHU {split}: {len(self.ids)} images, "
              f"{len(self.coco.anns)} annotations, "
              f"categories: {self.coco.cats}")

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, img_info) -> Tensor:
        file_name = img_info['file_name']   # e.g. "xxx.tif"
        img_path = self.img_dir / file_name
        if not img_path.exists():
            # fallback if paths in COCO differ
            img_path = next(self.img_dir.glob(file_name.split('/')[-1]))
        image = io.read_image(str(img_path))   # [C,H,W], uint8
        return image

    def _build_mask(self, img_id: int, height: int, width: int) -> Tensor:
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            if ann['category_id'] != 1:
                continue
            m = self.coco.annToMask(ann)
            mask = np.maximum(mask, m)

        return torch.from_numpy(mask.astype(np.uint8))  # [H,W], 0/1

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_id = self.ids[index]
        img_info = self.coco.loadImgs([img_id])[0]

        image = self._load_image(img_info)                           # [C,H,W]
        mask = self._build_mask(img_id, img_info['height'], img_info['width'])  # [H,W]

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        mask = mask.long()   # class ids: 0 or 1
        return image, mask
