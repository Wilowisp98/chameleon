"""
 Implementation based on:
 - https://youtu.be/IHq1t7NxS8k?si=d9dofGF9n96192R8
 - https://youtu.be/HS3Q_90hnDg?si=6BFVv_jLfQLhuA5i
 - https://d2l.ai/chapter_convolutional-modern/batch-norm.html
 - https://www.youtube.com/watch?v=oLvmLJkmXuc
"""

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from model import UNet
from configs import * 
from utils import (
    load_model,
    save_checkpoint,
    get_loaders,
    get_metrics
)

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    total_loss = 0

    for _, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        total_loss += loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    
    return total_loss/len(loader)

def main():
    train_transformations = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transformations = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print('=> Loading data...')
    train_loader, val_loader = get_loaders(
        DATA_DIR,
        DATASETS,
        ORIGINAL_FOLDER,
        SKIN_MASKS_FOLDER,
        BATCH_SIZE,
        train_transformations,
        val_transformations,
        NUM_WORKERS
    )
    print('=> Data loaded with success ✓')

    metrics = []
    model_number = 0
    if LOAD_MODEL:
        saved_models = os.listdir(MODEL_FOLDER)
        saved_models.sort()
        model_number += max(int(model.split('_')[2].split('.')[0].replace('E', '')) for model in saved_models)
        model_name = f'{MODEL_NAME}_E{model_number}.pth.tar'
        model_path = os.path.join(MODEL_FOLDER, model_name)
        checkpoint = torch.load(model_path)
        load_model(checkpoint, model)
        if "metrics" in checkpoint:
            metrics = checkpoint["metrics"]
            print(f"Loaded {len(metrics)} previous metrics records with success ✓")
    
    for epoch in range(model_number + 1, NUM_EPOCHS + model_number + 1):
        print(f"=> Epoch {epoch}/{NUM_EPOCHS + model_number}")
        average_loss = train_fn(train_loader, model, optimizer, LOSS_FN)
        print(f"Average loss: {average_loss:.4f}")
        
        epoch_metrics = get_metrics(val_loader, model, device=DEVICE)
        epoch_metrics['average_loss'] = average_loss
        metrics.append(epoch_metrics)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics
        }
        save_checkpoint(checkpoint, f"{MODEL_PATH}/{MODEL_NAME}_E{epoch}.pth.tar")
    
    # Need to develop the metrics plotting here.

if __name__ == "__main__":
    main()