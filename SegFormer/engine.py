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

    # Automatic NaN recovery: keep a recent good checkpoint and try to restore when NaNs occur
    last_good_state = None
    nan_retries = 0
    max_nan_retries = getattr(args, 'max_nan_retries', 3)
    nan_lr_reduce = getattr(args, 'nan_lr_reduce', 0.2)
    nan_save_every = 50  # how often (in iterations) to refresh last_good_state


    def _binary_dice(preds, targets, eps=1e-6):
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()
        inter = (preds_flat * targets_flat).sum(1)
        union = preds_flat.sum(1) + targets_flat.sum(1)
        score = (2 * inter + eps) / (union + eps)
        return (1 - score).mean()

    for iter, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):

        # Save a recent good state periodically so we can recover if a NaN occurs
        if (iter % nan_save_every) == 0 or last_good_state is None:
            try:
                last_good_state = {
                    'model': {k: v.cpu() for k, v in model.state_dict().items()},
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'scheduler': scheduler.state_dict() if scheduler is not None else None
                }
            except Exception as e:
                print(f"[WARN] Failed to save last_good_state: {e}")

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

        # Quick sanity debug: print foreground ratio for first few batches
        if iter < 5:
            with torch.no_grad():
                fg_ratio = (lbl == 1).float().mean().item()
                if fg_ratio == 0:
                    print(f"[Debug] Batch {iter}: no foreground pixels (fg_ratio=0). Check dataset/transforms.")
                else:
                    print(f"[Debug] Batch {iter}: foreground ratio = {fg_ratio:.6f}")

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

        # Quick debug: log prediction distribution for early iterations
        if iter < 10:
            with torch.no_grad():
                if mask_logits.shape[1] > 1:
                    preds = mask_logits.argmax(dim=1)
                    pred_fg_ratio = (preds == 1).float().mean().item()
                    probs_fg_mean = torch.softmax(mask_logits, dim=1)[:, 1, :, :].mean().item()
                    print(f"[Debug] Iter {iter}: pred_fg_ratio={pred_fg_ratio:.6f}, probs_fg_mean={probs_fg_mean:.6f}")
                else:
                    # binary logits case
                    probs_fg_mean = torch.sigmoid(mask_logits).mean().item()
                    print(f"[Debug] Iter {iter}: probs_fg_mean(binary)={probs_fg_mean:.6f}")

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

            # Debug: print loss breakdown for early iters
            if iter < 10:
                print(f"[Debug] Iter {iter}: mask_loss={mask_loss:.6f}, edge_loss={edge_loss:.6f}, lambda_edge={lambda_edge}")

        loss = total_loss

        # Debug: print total loss and lr for early iters
        if iter < 5:
            lr_debug = optimizer.param_groups[0]["lr"]
            print(f"[Debug] Iter {iter}: total_loss={loss.item():.6f}, lr={lr_debug:.8f}")

        # Detect NaN/Inf in loss before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] NaN or Inf loss detected at iter={iter}")

            # Save quick debug dump for inspection
            try:
                torch.save({
                    'epoch': epoch, 'iter': iter,
                    'img': img.detach().cpu(),
                    'lbl': lbl.detach().cpu(),
                    'edge': edge.detach().cpu() if edge is not None else None,
                    'mask_logits': mask_logits.detach().cpu() if 'mask_logits' in locals() and mask_logits is not None else None,
                    'edge_logits': edge_logits.detach().cpu() if 'edge_logits' in locals() and edge_logits is not None else None,
                }, f'nan_debug_epoch{epoch}_iter{iter}.pt')
                print(f"[ERROR] Saved NaN debug dump: nan_debug_epoch{epoch}_iter{iter}.pt")
            except Exception as e:
                print(f"[ERROR] Could not save debug dump: {e}")

            # Attempt automatic recovery: restore last good state and reduce lr
            if last_good_state is not None and nan_retries < max_nan_retries:
                print(f"[RECOVERY] Restoring last good state and reducing LR by factor {nan_lr_reduce} (attempt {nan_retries+1}/{max_nan_retries})")
                try:
                    model.load_state_dict(last_good_state['model'])
                except Exception as e:
                    print(f"[RECOVERY] model load failed: {e}")
                try:
                    optimizer.load_state_dict(last_good_state['optimizer'])
                    for g in optimizer.param_groups:
                        g['lr'] = g.get('lr', 0.0) * nan_lr_reduce
                except Exception as e:
                    print(f"[RECOVERY] optimizer restore failed: {e}")
                if scaler is not None and last_good_state.get('scaler') is not None:
                    try:
                        scaler.load_state_dict(last_good_state['scaler'])
                    except Exception as e:
                        print(f"[RECOVERY] scaler restore failed: {e}")

                # zero grads and skip this batch
                optimizer.zero_grad()
                nan_retries += 1
                continue
            else:
                print("[ERROR] No last good state available or max retries exceeded; aborting.")
                raise RuntimeError("NaN loss encountered and recovery failed")

        # backward with optional AMP
        if scaler is not None:
            scaler.scale(loss).backward()
            # gradient clipping (AMP-compatible)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            # check for NaN/Inf in gradients
            grad_nan = False
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    grad_nan = True
                    break
            if grad_nan:
                print(f"[ERROR] NaN/Inf found in gradients at iter={iter}; skipping optimizer step")
                optimizer.zero_grad()
            else:
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

    for batch_i, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
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

        # Debug: print prediction distribution and gt ratio for first few val batches
        if batch_i < 10:
            with torch.no_grad():
                preds = mask_logits.argmax(1)
                pred_fg_ratio = (preds == 1).float().mean().item()
                gt_fg_ratio = (labels == 1).float().mean().item()
                try:
                    probs_fg_mean = torch.softmax(mask_logits, dim=1)[:, 1, :, :].mean().item()
                except Exception:
                    probs_fg_mean = torch.sigmoid(mask_logits).mean().item()
                print(f"[Val Debug] batch {batch_i}: pred_fg_ratio={pred_fg_ratio:.6f}, gt_fg_ratio={gt_fg_ratio:.6f}, probs_fg_mean={probs_fg_mean:.6f}")

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