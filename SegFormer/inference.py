import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from models import *  # register model factory functions
from utils.augmentations import get_val_augmentation


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def _load_image_as_tensor(path: Path):
    img = Image.open(str(path)).convert('RGB')
    W, H = img.size
    arr = np.array(img)  # H,W,3 uint8
    img = torch.from_numpy(arr).permute(2, 0, 1)  # [C,H,W]
    return img, (H, W)


def _maybe_remove_ddp_prefix(state_dict):
    # Some checkpoints have keys beginning with 'module.' from DDP training. Remove that if present.
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v
    return new_state


def infer_on_directory(checkpoint_path: str, input_dir: str, output_dir: str,
                       image_size=(512, 512), device='cpu', model_name='make_SegFormerB1', num_classes=2,
                       save_overlay=False):
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    # create model
    # model factories are registered as functions in models module (e.g. make_SegFormerB1)
    if model_name not in globals():
        # fallback: try to import function from models module namespace
        model_fn = getattr(__import__('models', fromlist=[model_name]), model_name)
    else:
        model_fn = globals()[model_name]

    model = model_fn(num_classes=int(num_classes))

    # load checkpoint
    ck = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ck, dict) and 'model_state' in ck:
        state = ck['model_state']
    else:
        state = ck

    state = _maybe_remove_ddp_prefix(state)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    transform = get_val_augmentation(tuple(map(int, image_size)))

    for p in tqdm(sorted(Path(input_dir).glob('*'))):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            img_tensor, (orig_h, orig_w) = _load_image_as_tensor(p)
        except Exception:
            continue

        # prepare dummy mask for transform (transform expects (img, mask))
        mask_dummy = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.uint8)
        img_proc, _ = transform(img_tensor, mask_dummy)  # [C,Hp,Wp]

        x = img_proc.unsqueeze(0).to(device)

        with torch.no_grad():
            mask_logits, _ = model(x)
            preds = torch.argmax(mask_logits, dim=1).squeeze(0).cpu().to(torch.uint8)  # [Hp,Wp]

        # convert to PIL and resize to original size using nearest neigh
        pred_pil = Image.fromarray(preds.numpy())
        pred_pil = pred_pil.resize((orig_w, orig_h), resample=Image.NEAREST)

        # save predicted mask
        out_name = p.stem + '_pred.png'
        pred_pil.save(Path(output_dir) / out_name)

        if save_overlay:
            orig_pil = Image.open(str(p)).convert('RGB')
            # create a simple overlay: red for class 1 (if exists)
            import numpy as np
            color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            pred_np = np.array(pred_pil)
            # assign coloring for up to 3 classes (extendable)
            cmap = {
                0: (0, 0, 0),
                1: (255, 0, 0),
                2: (0, 255, 0),
                3: (0, 0, 255)
            }
            for k, col in cmap.items():
                color_mask[pred_np == k] = col
            overlay = (0.6 * np.array(orig_pil).astype(float) + 0.4 * color_mask.astype(float)).astype(np.uint8)
            Image.fromarray(overlay).save(Path(output_dir) / (p.stem + '_overlay.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SegFormer inference script')
    parser.add_argument('--checkpoint', type=str, default='./save_weights/best_model.pth', help='path to checkpoint')
    parser.add_argument('--input_dir', type=str, required=True, help='input images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory for predictions')
    parser.add_argument('--image_size', nargs=2, type=int, default=[512, 512], help='H W for val transform')
    parser.add_argument('--device', type=str, default='cpu', help='device (cpu or cuda:0)')
    parser.add_argument('--model', type=str, default='make_SegFormerB1', help='model factory name')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--save_overlay', action='store_true', help='save overlay images as well')

    args = parser.parse_args()
    infer_on_directory(args.checkpoint, args.input_dir, args.output_dir, args.image_size, args.device,
                       model_name=args.model, num_classes=args.num_classes, save_overlay=args.save_overlay)
