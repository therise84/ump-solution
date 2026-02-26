import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np

class RoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, img_name), 0)
        mask = (mask == 255).astype(np.float32)

        img = img / 255.0
        img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask

def train():
    dataset = RoadDataset("reference/sats", "reference/maps")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = smp.losses.DiceLoss(mode="binary")

    model.train()
    for epoch in range(5):
        for imgs, masks in loader:
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item()}")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    train()