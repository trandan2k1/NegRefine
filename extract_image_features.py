import os
import csv
import torch
import clip
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "/kaggle/input/imagenetmini-1000"

OUTPUT_CSV = "image_features.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

paths = []
for root, dirs, files in os.walk(IMAGE_DIR):
    for f in files:
        paths.append(os.path.join(root, f))

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    for path in tqdm(paths):
        try:
            image = Image.open(path).convert("RGB")
        except:
            continue

        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model.encode_image(image_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        img_name = os.path.basename(path)

        writer.writerow([img_name] + feat.cpu().numpy().flatten().tolist())

print(f"🎉 Done! Saved to {OUTPUT_CSV}")
