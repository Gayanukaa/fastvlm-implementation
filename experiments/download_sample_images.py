#!/usr/bin/env python3
"""
Download or Generate Sample Images for FastVLM Ablations
Aligned with the resolutions used in FastVLM (CVPR 2025): 256, 512, 768, 1024 px.
These images are only used for latency / token-count experiments (not accuracy).
"""

import os
import io
import requests
from PIL import Image, ImageDraw
from typing import List

TARGET_RESOLUTIONS = [256, 512, 768, 1024]
IMAGE_SOURCES = [
    "https://picsum.photos/{res}/{res}?random={i}",
    "https://loremflickr.com/{res}/{res}/nature,city?lock={i}",
]

def download_image(url: str, filename: str, target_dir: str = "../images") -> bool:
    """Download an image from URL and save it to target directory."""
    try:
        os.makedirs(target_dir, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        filepath = os.path.join(target_dir, filename)
        img.save(filepath, "JPEG", quality=95)
        print(f"✅ Saved {filename} ({img.width}x{img.height})")
        return True
    except Exception as e:
        print(f"❌ Failed {filename}: {e}")
        return False


def download_fastvlm_images():
    """Download minimal set of images at required ablation resolutions."""
    print("📦 Downloading sample images for FastVLM ablation experiments...")
    target_dir = "../images"
    os.makedirs(target_dir, exist_ok=True)

    success = 0
    for res in TARGET_RESOLUTIONS:
        for i, src in enumerate(IMAGE_SOURCES, start=1):
            filename = f"sample_{res}px_{i}.jpg"
            url = src.format(res=res, i=i)
            print(f"🔗 Fetching {filename}")
            if download_image(url, filename, target_dir):
                success += 1

    print(f"\n🎯 Summary: {success} images saved to {target_dir}")
    verify_images(target_dir)


def create_fallback_synthetics():
    """Create synthetic placeholders at 256–1024px if downloads fail."""
    print("\n🧪 Creating fallback synthetic images...")
    from random import randint

    target_dir = "../images"
    os.makedirs(target_dir, exist_ok=True)

    for res in TARGET_RESOLUTIONS:
        filename = f"synthetic_{res}px.jpg"
        img = Image.new("RGB", (res, res), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        for _ in range(8):
            x1, y1 = randint(0, res // 2), randint(0, res // 2)
            x2, y2 = randint(res // 2, res), randint(res // 2, res)
            color = (randint(50, 200), randint(50, 200), randint(50, 200))
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0))
        draw.text((res // 4, res // 2), f"{res}px", fill=(0, 0, 0))
        img.save(os.path.join(target_dir, filename), "JPEG", quality=95)
        print(f"✅ Created {filename}")

    verify_images(target_dir)


def verify_images(target_dir: str):
    """List all images and their sizes."""
    from PIL import Image
    all_imgs = [
        f for f in os.listdir(target_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"\n📋 Total images in {target_dir}: {len(all_imgs)}")
    for imgf in sorted(all_imgs):
        path = os.path.join(target_dir, imgf)
        try:
            with Image.open(path) as im:
                print(f"  📷 {imgf}: {im.width}x{im.height}")
        except:
            print(f"  ⚠️ Could not read {imgf}")


def main():
    print("🚀 FastVLM Image Downloader (CVPR 2025 resolutions)")
    print("=" * 55)

    choice = input(
        "Choose option:\n"
        "1. Download images (256–1024 px)\n"
        "2. Create fallback synthetic images\n"
        "3. Both\n"
        "Enter choice (1/2/3): "
    ).strip()

    if choice in ["1", "3"]:
        download_fastvlm_images()
    if choice in ["2", "3"]:
        create_fallback_synthetics()

    print("\n🎯 Image preparation complete.")
    print("💡 These resolutions match the ablations in Figures 3–4 and Tables 4–5.")
    print("   You can now run experiments such as exp_resolution_scaling.py")

if __name__ == "__main__":
    main()
