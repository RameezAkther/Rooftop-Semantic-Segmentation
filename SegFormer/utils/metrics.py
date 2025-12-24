import torch
from torch import Tensor
from typing import Tuple


def compute_boundary(mask: Tensor, kernel_size: int = 3) -> Tensor:
    """Compute thin boundary map of binary mask(s).
    mask: [B,H,W] or [H,W] (0/1 or bool)
    returns: [B,H,W] float tensor with 0/1
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    # ensure float tensor [B,1,H,W]
    m = mask.unsqueeze(1).float()
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device, dtype=m.dtype)
    conv = torch.nn.functional.conv2d(m, kernel, padding=kernel_size // 2)
    eroded = (conv == (kernel_size * kernel_size)).float()
    boundary = (m - eroded).clamp(0, 1).squeeze(1)
    return boundary


def boundary_f1(pred_logits: Tensor, gt_mask: Tensor, threshold: float = 0.5) -> float:
    """Compute boundary F1 between predicted logits (multi-class or binary) and gt_mask (0/1)
    pred_logits: [B,C,H,W] or [B,1,H,W]
    gt_mask: [B,H,W]
    """
    if pred_logits.dim() == 4 and pred_logits.shape[1] > 1:
        pred = pred_logits.argmax(dim=1)
    else:
        # binary logits
        if pred_logits.dim() == 4:
            pred = (torch.sigmoid(pred_logits.squeeze(1)) > threshold).long()
        else:
            pred = (torch.sigmoid(pred_logits) > threshold).long()

    gt = gt_mask.long()
    pred_b = compute_boundary(pred)
    gt_b = compute_boundary(gt)

    tp = (pred_b * gt_b).sum().float()
    pred_sum = pred_b.sum().float()
    gt_sum = gt_b.sum().float()

    prec = tp / (pred_sum + 1e-6)
    rec = tp / (gt_sum + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return f1.item()


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1).flatten()
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)