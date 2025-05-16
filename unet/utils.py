"""
 Implementation based on:
 - https://youtu.be/IHq1t7NxS8k?si=d9dofGF9n96192R8
 - https://youtu.be/HS3Q_90hnDg?si=6BFVv_jLfQLhuA5i
 - https://d2l.ai/chapter_convolutional-modern/batch-norm.html
 - https://www.youtube.com/watch?v=oLvmLJkmXuc
"""

import torch
from dataset import load
from torch.utils.data import DataLoader

def save_checkpoint(checkpoint, filename):
    print("=> Saving model checkpoint...")
    torch.save(checkpoint, filename)
    print("=> Model checkpoint saved with success ✓")

def load_model(checkpoint, model):
    print("=> Loading model...")
    model.load_state_dict(checkpoint["state_dict"])
    print("=> Model loaded with success ✓")

def get_loaders(
    data_dir,
    datasets,
    original,
    mask,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4
):
    train_ds = load(
        data_dir=data_dir,
        datasets=datasets,
        original=original,
        mask=mask,
        transform=train_transform,
        train = True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    val_ds = load(
        data_dir=data_dir,
        datasets=datasets,
        original=original,
        mask=mask,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    return train_loader, val_loader

def get_metrics(loader, model, device: str = "cpu"):
    # Good explanation: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    model.eval()

    num_correct = num_incorrect = num_pixels = 0
    total_dice_coeff = total_IoU = 0
    total_TP = total_TN = total_FP = total_FN = 0
    total_acc = total_precision = total_recall = total_f1 = 0
    num_batches = len(loader)

    # no_grad disables gradient calculation
    # Since I'm only infering there is no need for it.
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()

            # Overlap metrics:
            # - Dice Coefficient
            # - IoU (Intersection over Union)
            num_correct += (preds == y).sum()
            num_incorrect += (preds != y).sum()
            num_pixels += torch.numel(preds)
            total_dice_coeff += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            total_IoU += (preds * y).sum() / ((preds == 1.0).sum() + (y == 1.0).sum() + 1e-8)

            # Pixel-level metrics:
            # - Accuracy
            # - Precision & Recall
            # - F1 
            TP = ((preds == 1.0) & (y == 1.0)).sum()
            total_TP += TP
            TN = ((preds == 0.0) & (y == 0.0)).sum()
            total_TN += TN
            FP = ((preds == 1.0) & (y == 0.0)).sum()
            total_FP += FP
            FN = ((preds == 0.0) & (y == 1.0)).sum()
            total_FN += FN
            
            iter_acc = (TP + TN) / (TP + TN + FP + FN)
            total_acc += iter_acc

            iter_precision = TP / (TP + FP + 1e-8)
            total_precision += iter_precision

            iter_recall = TP / (TP + FN + 1e-8)
            total_recall += iter_recall

            iter_f1 = TP / (TP + (0.5 * (FP + FN)) + 1e-8)
            total_f1 += iter_f1

            # Boundary-Based Metrics (I would like to work on this one aswell):
            # - Hausdorff Distance
    
    # Global metrics (based on total counts)
    global_acc = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
    global_precision = total_TP / (total_TP + total_FP + 1e-8)
    global_recall = total_TP / (total_TP + total_FN + 1e-8)
    global_f1 = total_TP / (total_TP + (0.5 * (total_FP + total_FN)) + 1e-8)
    pixel_acc = num_correct / num_pixels if num_pixels > 0 else 0
    
    # Mean metrics (average across batches)
    mean_dice_coeff = total_dice_coeff / num_batches
    mean_IoU = total_IoU / num_batches
    mean_acc = total_acc / num_batches
    mean_precision = total_precision / num_batches
    mean_recall = total_recall / num_batches
    mean_f1 = total_f1 / num_batches
    
    metrics = {
        # Global metrics
        "global_acc": global_acc.item(),
        "global_precision": global_precision.item(),
        "global_recall": global_recall.item(),
        "global_f1": global_f1.item(),
        "pixel_acc": pixel_acc.item(),
        
        # Mean metrics
        "mean_dice_coeff": mean_dice_coeff.item(),
        "mean_IoU": mean_IoU.item(),
        "mean_acc": mean_acc.item(),
        "mean_precision": mean_precision.item(),
        "mean_recall": mean_recall.item(),
        "mean_f1": mean_f1.item(),
        
        # Confusion matrix components
        "TP": total_TP.item(),
        "TN": total_TN.item(),
        "FP": total_FP.item(),
        "FN": total_FN.item()
    }
    
    print("\n" + "="*50)
    print("SEGMENTATION METRICS:")
    print("="*50)
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean Dice Coefficient: {mean_dice_coeff:.4f}")
    print(f"Mean IoU: {mean_IoU:.4f}")
    print("\nGLOBAL METRICS (calculated on total TP, TN, FP, FN):")
    print(f"Accuracy: {global_acc:.4f}")
    print(f"Precision: {global_precision:.4f}")
    print(f"Recall: {global_recall:.4f}")
    print(f"F1 Score: {global_f1:.4f}")
    print("\nMEAN METRICS (averaged across batches):")
    print(f"Accuracy: {mean_acc:.4f}")
    print(f"Precision: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"F1 Score: {mean_f1:.4f}")
    print("="*50 + "\n")
    
    # Can not forget to turn train back on! 
    model.train()
    return metrics