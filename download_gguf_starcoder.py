import os
import requests
from tqdm import tqdm

# StarCoder2-7B - Ungated Public Mirror
# Using QuantFactory to bypass the 401 Authorization issues
repo_id = "QuantFactory/starcoder2-7b-GGUF"
filename = "starcoder2-7b.Q4_K_M.gguf"
dest_dir = "models"

os.makedirs(dest_dir, exist_ok=True)

# Hugging Face resolve URL
url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
out_path = os.path.join(dest_dir, filename)

print(f"🚀 Downloading StarCoder2-7B (Ungated Mirror)...")
print(f"Size: ~4.5GB | Provider: QuantFactory")

try:
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        with open(out_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"\n✅ Success! StarCoder2 is saved at: {out_path}")

except Exception as e:
    print(f"\n❌ Download failed: {e}")
    if "401" in str(e):
        print("💡 Still getting 401? Hugging Face might be requiring a login for StarCoder2 today.")
        print("Try: Creating a free 'Read' token at huggingface.co/settings/tokens")
