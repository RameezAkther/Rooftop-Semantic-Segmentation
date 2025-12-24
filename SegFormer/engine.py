import torch
import math
from torch.nn import functional as F
from tqdm import tqdm
from utils.metrics import Metrics, boundary_f1
from torch.cuda.amp import autocast
import utils.distributed_utils as utils
from utils.losses import Dice



def train_one_epoch(args, model, optimizer, loss_fn, dataloader, sampler, scheduler,
                    epoch, device, print_freq, scaler=None):
    model.train()

    if args.DDP:
        sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    def _binary_dice(preds, targets, eps=1e-6):
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()
        inter = (preds_flat * targets_flat).sum(1)
        union = preds_flat.sum(1) + targets_flat.sum(1)
        score = (2 * inter + eps) / (union + eps)
        return (1 - score).mean()

    for iter, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):

        # support datasets that return (img, label) or (img, label, edge)
        if len(batch) == 2:
            img, lbl = batch
            edge = None
        elif len(batch) == 3:
            img, lbl, edge = batch
        else:
            raise ValueError("Unsupported batch format from dataloader")

        img = img.to(device)
        lbl = lbl.to(device)
        if edge is not None:
            edge = edge.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(enabled=args.amp):
                outputs = model(img)
        else:
            outputs = model(img)

        # model can return single logits tensor or (mask_logits, edge_logits)
        if isinstance(outputs, tuple):
            mask_logits, edge_logits = outputs
        else:
            mask_logits = outputs
            edge_logits = None

        # Mask loss: combine configured loss (e.g. OHEM/CE/Focal) with Dice
        # dice: handle binary (2-class) specially, otherwise use multiclass Dice
        if mask_logits.shape[1] == 2:
            probs_fg = torch.softmax(mask_logits, dim=1)[:, 1, :, :]
            dice_loss_mask = _binary_dice(probs_fg, (lbl == 1).float())
        else:
            dice_fn = Dice()
            probs = torch.softmax(mask_logits, dim=1)
            dice_loss_mask = dice_fn(probs, lbl)

        ce_loss = loss_fn(mask_logits, lbl)
        mask_loss = 0.5 * ce_loss + 0.5 * dice_loss_mask

        total_loss = mask_loss

        # Edge loss if provided by dataset and model
        if (edge_logits is not None) and (edge is not None):
            edge_logits_s = edge_logits.squeeze(1)
            edge_bce = torch.nn.functional.binary_cross_entropy_with_logits(edge_logits_s, edge.float())
            edge_probs = torch.sigmoid(edge_logits_s)
            edge_dice = _binary_dice(edge_probs, edge)
            edge_loss = edge_bce + edge_dice
            lambda_edge = getattr(args, 'lambda_edge', 0.3)
            total_loss = mask_loss + lambda_edge * edge_loss

        loss = total_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()
        torch.cuda.synchronize()

        loss_value = loss.item()
        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(loss=loss_value, lr=lr)

    torch.cuda.empty_cache()

    return metric_logger.meters["loss"].global_avg, lr



@torch.no_grad()
def evaluate(args, model, dataloader, device, print_freq):
    model.eval()

    confmat = utils.ConfusionMatrix(args.num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    boundary_scores = []

    for batch in metric_logger.log_every(dataloader, print_freq, header):
        # support (images, labels) or (images, labels, edge)
        if len(batch) == 2:
            images, labels = batch
        elif len(batch) == 3:
            images, labels, _ = batch
        else:
            raise ValueError('Unsupported batch format from dataloader')

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            mask_logits = outputs[0]
        else:
            mask_logits = outputs

        confmat.update(labels.flatten(), mask_logits.argmax(1).flatten())

        # compute boundary F1 per batch (using predicted mask)
        try:
            f1 = boundary_f1(mask_logits.detach().cpu(), labels.detach().cpu())
            boundary_scores.append(f1)
        except Exception:
            # if shapes/types unexpected, skip boundary metric
            pass

    confmat.reduce_from_all_processes()

    if len(boundary_scores) > 0:
        avg_bf1 = sum(boundary_scores) / len(boundary_scores)
        print(f"Boundary F1 (mean over batches): {avg_bf1:.4f}")

    return confmat



@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for batch in tqdm(dataloader):
        # support (images, labels) or (images, labels, edge)
        if len(batch) == 2:
            images, labels = batch
        elif len(batch) == 3:
            images, labels, _ = batch
        else:
            raise ValueError('Unsupported batch format from dataloader')

        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou