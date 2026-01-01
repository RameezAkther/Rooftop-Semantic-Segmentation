#!/usr/bin/env python3
"""
Simple evaluation script to load a checkpoint and evaluate on the validation set.
Saves/prints the metrics (confusion matrix / IoU / boundary F1 printed by engine.evaluate).
"""
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import *               # model factory functions (e.g., make_SegFormerB1)
from datasets import *             # dataset classes (WHUBuilding, CityScapes)
from utils.augmentations import get_val_augmentation
from engine import evaluate
from inference import _maybe_remove_ddp_prefix  # helper used in repository


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./save_weights/best_model.pth')
    parser.add_argument('--data_root', type=str, required=True, help='dataset root (same as training)')
    parser.add_argument('--dataset', type=str, default='whu', choices=['whu', 'cityscapes'])
    parser.add_argument('--model', type=str, default='make_SegFormerB1')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--image_size', nargs=2, type=int, default=[512, 512], help='H W for val transform')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_print_freq', type=int, default=50)
    parser.add_argument('--ignore_label', type=int, default=255)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(Path(args.checkpoint).parent, exist_ok=True)

    # create model from factory
    if args.model not in globals():
        model_fn = getattr(__import__('models', fromlist=[args.model]), args.model)
    else:
        model_fn = globals()[args.model]
    model = model_fn(num_classes=args.num_classes)

    # load checkpoint
    ck = torch.load(args.checkpoint, map_location='cpu')
    state = ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck
    state = _maybe_remove_ddp_prefix(state)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # dataset + dataloader
    transform = get_val_augmentation(tuple(map(int, args.image_size)))
    if args.dataset == 'whu':
        val_set = WHUBuilding(args.data_root, 'val', transform, scale_aware=False)
    else:
        val_set = CityScapes(args.data_root, 'val', transform)

    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False, pin_memory=True)

    # attach required args fields used by evaluate (num_classes, ignore_label, val_print_freq)
    args.num_classes = args.num_classes
    args.ignore_label = args.ignore_label

    confmat = evaluate(args, model, val_loader, device, args.val_print_freq)
    print("Evaluation complete. Confusion matrix summary:\n")
    print(confmat)


if __name__ == '__main__':
    main()