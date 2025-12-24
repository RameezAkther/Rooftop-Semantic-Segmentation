import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image  # ← use PIL instead of torchvision.io


class WHUBuilding(Dataset):
    """
    WHU Building Rooftop Dataset
    - Single foreground class: building (id = 1 in COCO)
    - Labels: 0 = background, 1 = building
    """

    CLASSES = ['background', 'building']

    def __init__(self, root: str, split: str = 'train', transform=None, scale_aware: bool = True, small_area: int = 32) -> None:
        super().__init__()
        import random
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.root = Path(root)
        self.split = split
        self.scale_aware = scale_aware
        self.small_area = small_area

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

        # Precompute images containing small roofs (for oversampling)
        self.small_img_ids = []
        if self.scale_aware and self.split == 'train':
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                anns = self.coco.loadAnns(ann_ids)
                has_small = False
                for ann in anns:
                    # bbox: [x,y,width,height]
                    bbox_area = ann['bbox'][2] * ann['bbox'][3]
                    if bbox_area < (self.small_area * self.small_area):
                        has_small = True
                        break
                if has_small:
                    self.small_img_ids.append(img_id)
            print(f"Found {len(self.small_img_ids)}/{len(self.ids)} images with small roofs")

        print(f"WHU {split}: {len(self.ids)} images, "
              f"{len(self.coco.anns)} annotations, "
              f"categories: {self.coco.cats}")

    def __len__(self) -> int:
        # if scale-aware oversampling is enabled for training, length increases to include oversampled small images
        if hasattr(self, 'small_img_ids') and len(self.small_img_ids) > 0 and self.split == 'train':
            return max(len(self.ids), len(self.small_img_ids) * 3)
        return len(self.ids)

    def _load_image(self, img_info) -> Tensor:
        file_name = img_info['file_name']   # e.g. "xxx.tif"
        img_path = self.img_dir / file_name
        if not img_path.exists():
            img_path = next(self.img_dir.glob(file_name.split('/')[-1]))

        # PIL can read .tif; convert to RGB and then to tensor [C,H,W]
        img = Image.open(str(img_path)).convert("RGB")
        img = np.array(img)                  # H,W,3 uint8
        img = torch.from_numpy(img).permute(2, 0, 1)  # to [C,H,W]
        return img

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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        # sample from small roofs with higher probability during training
        if hasattr(self, 'small_img_ids') and len(self.small_img_ids) > 0 and self.split == 'train':
            import random
            if random.random() < 0.6:
                img_id = random.choice(self.small_img_ids)
            else:
                img_id = self.ids[index % len(self.ids)]
        else:
            img_id = self.ids[index % len(self.ids)]

        img_info = self.coco.loadImgs([img_id])[0]

        image = self._load_image(img_info)                           # [C,H,W]
        mask  = self._build_mask(img_id, img_info['height'], img_info['width'])  # [H,W]

        # add channel dim so transforms see it as [1,H,W]
        if self.transform is not None:
            mask = mask.unsqueeze(0)          # [1,H,W]
            image, mask = self.transform(image, mask)
            mask = mask.squeeze(0)            # back to [H,W]

        mask = mask.long()

        # Build edge GT: edge = mask - erode(mask) with 3x3 kernel
        # erosion: pixel remains 1 only if all 3x3 neighbors are 1
        import torch.nn.functional as F
        mask_float = mask.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32)
        conv = F.conv2d(mask_float, kernel, padding=1)
        eroded = (conv == 9).squeeze(0).squeeze(0).to(dtype=torch.uint8)
        edge = (mask - eroded).clamp(0, 1).to(dtype=torch.uint8)

        return image, mask, edge

