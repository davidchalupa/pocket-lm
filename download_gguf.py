# this script downloads and install GGUF for the SLM model used

import os
import requests
from tqdm import tqdm

repo_id = "TheBloke/rocket-3B-GGUF"
filename = "rocket-3b.Q4_K_M.gguf"
dest_dir = "models"
os.makedirs(dest_dir, exist_ok=True)

# HF token (optional) — export HF_HUB_TOKEN or HUGGINGFACE_HUB_TOKEN if the model is gated
token = os.environ.get("HF_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
headers = {"Authorization": f"Bearer {token}"} if token else {}

url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
tmp_path = os.path.join(dest_dir, filename + ".part")
out_path = os.path.join(dest_dir, filename)

with requests.get(url, stream=True, headers=headers, timeout=30) as r:
    if r.status_code == 403:
        raise SystemExit("HTTP 403: Access denied — gated model. Set HF_HUB_TOKEN and accept the model on Hugging Face.")
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                         unit_divisor=1024, desc=filename) as pbar:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if not chunk:
                continue
            f.write(chunk)
            pbar.update(len(chunk))

# atomic rename
os.replace(tmp_path, out_path)
print("Model downloaded to:", out_path)
