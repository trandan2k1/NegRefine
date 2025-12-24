import clip
import torch
from PIL import Image
import os
import csv
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

IMAGE_DIR = "/kaggle/input/imagenet-mini-1000-torch-mobilenet"
OUTPUT_CSV = "image_features.csv"

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    for root, _, files in os.walk(IMAGE_DIR):
        for img in tqdm(files):
            if not img.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(root, img)
            try:
                image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = model.encode_image(image)
                    feat = feat / feat.norm(dim=-1, keepdim=True)

                writer.writerow(
                    [img] + feat.cpu().numpy().flatten().tolist()
                )
            except:
                continue
