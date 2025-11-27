# #!/bin/bash

# echo "===== Preparing folder ====="
# mkdir -p comparison_models
# cd comparison_models

# # ---------- ConvNeXt-L ----------
# CONV_MODEL="convnext_large.pth"

# echo "----- Checking ConvNeXt-L -----"
# if [ -f "$CONV_MODEL" ]; then
#     echo "✅ $CONV_MODEL already exists — skipping download."
# else
#     echo "⬇ Downloading ConvNeXt-L pretrained weights..."
#     python3 - << 'EOF'
# import torch
# from torchvision.models import convnext_large, ConvNeXt_Large_Weights

# print("Downloading ConvNeXt-L (ImageNet1K)…")
# model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
# torch.save(model.state_dict(), "convnext_large.pth")
# print("\n✅ Saved convnext_large.pth")
# EOF
# fi


# # ---------- FastViTHD ----------
# FASTVITHD="fastvithd.pth"
# FASTVITHD_URL="https://ml-fastvlm.s3.amazonaws.com/fastvithd_weights/fastvithd.pth"

# echo "----- Checking FastViTHD -----"
# if [ -f "$FASTVITHD" ]; then
#     echo "✅ $FASTVITHD already exists — skipping download."
# else
#     echo "⬇ Downloading FastViTHD checkpoint..."
#     wget -O "$FASTVITHD" "$FASTVITHD_URL"
#     echo "✅ Saved fastvithd.pth"
# fi


# # ---------- ViT-L/14 (OpenCLIP) ----------
# VIT_MODEL="vit_l_14_openclip.pt"

# echo "----- Checking ViT-L/14 (OpenCLIP) -----"
# if [ -f "$VIT_MODEL" ]; then
#     echo "✅ $VIT_MODEL already exists — skipping download."
# else
#     echo "⬇ Downloading ViT-L/14 (OpenCLIP) checkpoint..."
#     python3 - << 'EOF'
# import torch
# import open_clip

# print("Loading ViT-L/14 (OpenCLIP, pretrained='openai')…")
# model, _, _ = open_clip.create_model_and_transforms(
#     "ViT-L-14",
#     pretrained="openai"
# )

# torch.save(model.state_dict(), "vit_l_14_openclip.pt")
# print("\n✅ Saved vit_l_14_openclip.pt")
# EOF
# fi

# echo "===== 🎉 All model downloads completed ====="
# echo "Files stored in: comparison_models/"




#!/bin/bash
set -e

echo "===== Preparing folder ====="
mkdir -p encoder_models
cd encoder_models

########################################
# 1) ConvNeXt-L  --> convnext_large.pt
########################################
CONV_MODEL="convnext_large.pt"

echo "----- Checking ConvNeXt-L -----"
if [ -f "$CONV_MODEL" ]; then
    echo "✅ $CONV_MODEL already exists — skipping download."
else
    echo "⬇ Downloading ConvNeXt-L pretrained weights..."
    python3 - << 'EOF'
import torch
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

print("Downloading ConvNeXt-L (ImageNet1K)…")
model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

save_path = "convnext_large.pt"
torch.save(model.state_dict(), save_path)
print(f"\n✅ Saved {save_path}")
EOF
fi


# ---------- FastViTHD ----------
FASTVITHD="fastvithd.pt"

echo "----- Checking FastViTHD -----"
if [ -f "$FASTVITHD" ]; then
    echo "✅ $FASTVITHD already exists — skipping download."
else
    echo "⬇ Downloading FastViTHD (FastViT-HD) from Hugging Face..."
    python3 - << 'EOF'
import torch
from transformers import AutoModel

print("Loading FastViT-HD encoder from 'kevin510/fast-vit-hd'…")
model = AutoModel.from_pretrained(
    "kevin510/fast-vit-hd",
    trust_remote_code=True
)

save_path = "fastvithd.pt"
torch.save(model.state_dict(), save_path)
print(f"\n✅ Saved {save_path}")
EOF
fi



########################################
# 3) ViT-L/14 (OpenCLIP) --> vit_l_14_openclip.pt
########################################
VIT_MODEL="vit_l_14_openclip.pt"

echo "----- Checking ViT-L/14 (OpenCLIP) -----"
if [ -f "$VIT_MODEL" ]; then
    echo "✅ $VIT_MODEL already exists — skipping download."
else
    echo "⬇ Downloading ViT-L/14 (OpenCLIP) checkpoint..."
    python3 - << 'EOF'
import torch
import open_clip

print("Loading ViT-L/14 (OpenCLIP, pretrained='openai')…")
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)

save_path = "vit_l_14_openclip.pt"
torch.save(model.state_dict(), save_path)
print(f"\n✅ Saved {save_path}")
EOF
fi

echo "===== 🎉 All model downloads completed ====="
echo "Files stored in: comparison_models/"
