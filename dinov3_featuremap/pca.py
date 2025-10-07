import os
import torch
import pickle
import urllib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor, AutoModel
from sklearn.decomposition import PCA

# 基本配置
MODEL_NAME = "dinov3_vitb16"
LOCAL_MODEL_DIR = "/mnt/afs/wusize/models/facebook/dinov3-vitb16-pretrain-lvd1689m"
save_root = "/mnt/afs/wusize/projects/dinov3/results"
model_path = os.path.join(save_root, "fg_classifier.pkl")

PATCH_SIZE = 16
IMAGE_SIZE = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型与分类器
processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
model = AutoModel.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
model.to(device).eval()

with open(model_path, "rb") as f:
    clf = pickle.load(f)

# 加载测试图像
image_path = "/mnt/afs/wusize/projects/dinov3/images/1.jpg"

def load_image(image_path_or_url):
    """支持本地路径或URL"""
    if image_path_or_url.startswith("http"):
        with urllib.request.urlopen(image_path_or_url) as f:
            img = Image.open(f).convert("RGB")
    else:
        img = Image.open(image_path_or_url).convert("RGB")
    return img

image = load_image(image_path)

# resize 保证尺寸为 768×768
def resize_transform(image: Image, size=IMAGE_SIZE):
    return TF.to_tensor(TF.resize(image, (size, size)))

image_resized = resize_transform(image)
image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

# 提取 patch 特征
with torch.no_grad():
    outputs = model(
        pixel_values=image_resized_norm.unsqueeze(0).to(device),
        output_hidden_states=True
    )
    last_hidden = outputs.hidden_states[-1]
    feats = last_hidden[:, 1:, :].squeeze(0).cpu().numpy()  # (num_patches, dim)

print(f"Feature shape: {feats.shape}")

# 计算 patch 网格并裁剪
expected_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
h_patches = w_patches = IMAGE_SIZE // PATCH_SIZE
print(f"Expected patches: {h_patches}×{w_patches} = {expected_patches}")

if feats.shape[0] != expected_patches:
    print(f"[Warning] Patch count mismatch: got {feats.shape[0]}, expected {expected_patches}")
    # 裁剪到可整形大小
    feats = feats[:expected_patches, :]
    print(f"[Fixed] Trimmed feature shape: {feats.shape}")

x = feats  # 为保持与官方示例一致

#前景预测 + 中值滤波
fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

# 可视化前景检测结果
plt.figure(figsize=(9, 3), dpi=300)
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(image_resized.permute(1, 2, 0))
plt.title('input image')

plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(fg_score)
plt.title('foreground score')

plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(fg_score_mf)
plt.title('+ median filter')

output_save="/mnt/afs/wusize/projects/dinov3/output"
os.makedirs(output_save, exist_ok=True)
plt.tight_layout()
save_path = os.path.join(output_save, "pca_foreground_result.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"\n Foreground result saved to: {save_path}")

# PCA 彩色可视化（Rainbow Foreground Visualization）
print("\n[Stage] Computing PCA rainbow visualization...")

# 提取前景 patch（阈值 0.5）
foreground_selection = fg_score_mf.view(-1) > 0.5
fg_patches = x[foreground_selection.numpy()]

print(f"Foreground patches selected: {fg_patches.shape[0]} / {x.shape[0]}")

# 拟合 PCA
pca = PCA(n_components=3, whiten=True)
pca.fit(fg_patches)
print("PCA fitted successfully.")

# 对所有特征做 PCA 投影
projected_image = torch.from_numpy(pca.transform(x)).view(h_patches, w_patches, 3)

# 色彩增强（Sigmoid + 乘2）
projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)

# 用前景 mask 掩盖背景
projected_image *= (fg_score_mf.unsqueeze(0) > 0.5)

# 显示与保存结果
plt.figure(figsize=(5, 5), dpi=300)
plt.imshow(projected_image.permute(1, 2, 0))
plt.axis('off')
plt.title("Rainbow PCA Visualization (Foreground Only)")
plt.tight_layout()

pca_save_path = os.path.join(output_save, "pca_rainbow_foreground.png")
plt.savefig(pca_save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f" PCA rainbow visualization saved to: {pca_save_path}")
