import clip
import torch
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

classnames = [
    "dog", "cat", "car", "airplane", "truck"
]

templates = ["a photo of a {}"]

with open("text_features.csv", "w", newline="") as f:
    writer = csv.writer(f)

    for cname in classnames:
        texts = [t.format(cname) for t in templates]
        tokens = clip.tokenize(texts).to(device)

        with torch.no_grad():
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.mean(dim=0)

        writer.writerow([cname] + feat.cpu().numpy().tolist())
