import os
import requests
from pathlib import Path
from PIL import Image

CACHE_DIR = Path.home() / ".cache" / "belle_bot" / "images"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_image(image_id):
    # Check in specific local paths
    for split in ["train2017", "val2017"]:
        img_path = f"/Users/belle/Downloads/{split}/{image_id}"
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    return img.convert("RGB")
            except Exception:
                pass
    
    # Check in cache
    # If image_id is a URL, use filename, else use image_id as filename
    filename = os.path.basename(image_id)
    cache_path = CACHE_DIR / filename
    if cache_path.exists():
        try:
            with Image.open(cache_path) as img:
                return img.convert("RGB")
        except Exception:
            pass
    
    # Try to download if it looks like a URL (starts with http)
    if image_id.startswith("http"):
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(image_id, stream=True, headers=headers)
            response.raise_for_status()
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with Image.open(cache_path) as img:
                return img.convert("RGB")
        except Exception as e:
            print(f"Failed to download image {image_id}: {e}")
            pass
            
    return None
