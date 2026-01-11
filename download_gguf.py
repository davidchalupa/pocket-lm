# this script downloads and install GGUF for the SLM model used

import os
import requests
from tqdm import tqdm

import os
import requests
from tqdm import tqdm

repo_id = "TheBloke/rocket-3B-GGUF"
filename = "rocket-3b.Q4_K_M.gguf"
dest_dir = "models"
os.makedirs(dest_dir, exist_ok=True)

url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
tmp_path = os.path.join(dest_dir, filename + ".part")
out_path = os.path.join(dest_dir, filename)

with requests.get(url, stream=True, timeout=30) as r:
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                         unit_divisor=1024, desc=filename) as pbar:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if not chunk:
                continue
            f.write(chunk)
            pbar.update(len(chunk))

os.replace(tmp_path, out_path)
print("Model downloaded to:", out_path)
