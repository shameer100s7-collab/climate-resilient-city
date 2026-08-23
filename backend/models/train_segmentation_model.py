"""
Trains a REAL U-Net segmentation model for water detection — but only runs once
you provide real labeled data. This is intentionally not runnable out-of-the-box
with fake data; get real labeled flood-water images first:

  1. Public datasets (download manually, these require account/license acceptance
     so cannot be auto-fetched from this environment):
       - Kaggle "Flood Area Segmentation" dataset
       - FloodNet (UAV flood imagery with segmentation masks)
  2. Your own CCTV frames, labeled with a free tool like CVAT (https://cvat.ai)
     or Labelme — even 100-200 labeled frames is enough to fine-tune meaningfully.

Expected folder structure once you have data:
  backend/data/segmentation/
      images/*.jpg
      masks/*.png     (same filename as image, white=water, black=not-water)

Install extra dependency first:
  pip install segmentation-models-pytorch torch torchvision

Run:
  python train_segmentation_model.py
"""
import os
import glob

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "segmentation", "images")
MASKS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "segmentation", "masks")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "flood_segmentation.pt")


def main():
    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*")))
    masks = sorted(glob.glob(os.path.join(MASKS_DIR, "*")))

    if not images or not masks:
        print("ERROR: No labeled images/masks found.")
        print(f"Expected images in: {IMAGES_DIR}")
        print(f"Expected masks in:  {MASKS_DIR}")
        print("Label real CCTV frames or download a real public flood-segmentation "
              "dataset before running this script. See the docstring above.")
        return

    print(f"Found {len(images)} images and {len(masks)} masks. Starting training...")

    import torch
    from torch.utils.data import Dataset, DataLoader
    import segmentation_models_pytorch as smp
    import cv2
    import numpy as np

    class WaterDataset(Dataset):
        def __init__(self, image_paths, mask_paths, size=256):
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.size = size

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img = cv2.imread(self.image_paths[idx])
            img = cv2.resize(img, (self.size, self.size))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))

            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.size, self.size))
            mask = (mask > 127).astype(np.float32)[None, ...]

            return torch.tensor(img), torch.tensor(mask)

    dataset = WaterDataset(images, masks)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                      in_channels=3, classes=1, activation=None).to(device)
    loss_fn = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, masks_ in loader:
            imgs, masks_ = imgs.to(device), masks_.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, masks_)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
