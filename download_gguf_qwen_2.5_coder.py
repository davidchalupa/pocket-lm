import os
import requests
from tqdm import tqdm

# Updated for Qwen 2.5 Coder 7B (GGUF)
# Using bartowski's repo as it's a highly reliable source for GGUF quants
repo_id = "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
filename = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
dest_dir = "models"

os.makedirs(dest_dir, exist_ok=True)

# Hugging Face resolve URL
url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
tmp_path = os.path.join(dest_dir, filename + ".part")
out_path = os.path.join(dest_dir, filename)

print(f"Starting download: {filename}...")
print(f"This is approximately 4.7GB. Grab a coffee!")

try:
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        with open(tmp_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    os.replace(tmp_path, out_path)
    print("\n✅ Success! Model saved to:", out_path)

except Exception as e:
    print(f"\n❌ Download failed: {e}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
