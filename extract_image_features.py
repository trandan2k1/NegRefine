import os
import csv
import torch
import clip
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "/kaggle/input/imagenet-mini-1000-torch-mobilenet"
OUTPUT_CSV = "image_features.csv"
BATCH_SIZE = 1 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

image_paths = []
for root, _, files in os.walk(IMAGE_DIR):
    for f in files:
        image_paths.append(os.path.join(root, f))
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    for path in tqdm(image_paths):
        try:
            image = Image.open(path).convert("RGB")
        except:
            continue

        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        img_name = os.path.basename(path)

        writer.writerow(
            [img_name] + image_features.cpu().numpy().flatten().tolist()
        )
