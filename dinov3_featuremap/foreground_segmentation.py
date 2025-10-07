import os
import torch
import io
import pickle
import tarfile
import urllib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


IMAGES_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_images.tar.gz"
LABELS_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/foreground_segmentation_labels.tar.gz"

PATCH_SIZE = 16
IMAGE_SIZE = 768

# examples of available DINOv3 models:
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_NAME = MODEL_DINOV3_VITB

LOCAL_MODEL_DIR = "/mnt/afs/wusize/models/facebook/dinov3-vitb16-pretrain-lvd1689m"


# ==== 加载模型 ====
processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
model = AutoModel.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
model.cuda().eval()


SAVE_DIR = "/mnt/afs/wusize/projects/dinov3/output/foreground_results"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_DIR_PR = "/mnt/afs/wusize/projects/dinov3/output/pr_curves"
os.makedirs(SAVE_DIR_PR, exist_ok=True)

def load_images_from_remote_tar(tar_uri: str) -> list[Image.Image]:
    images = []
    with urllib.request.urlopen(tar_uri) as f:
        tar = tarfile.open(fileobj=io.BytesIO(f.read()))
        for member in tar.getmembers():
            image_data = tar.extractfile(member)
            image = Image.open(image_data)
            images.append(image)
    return images
    
images = load_images_from_remote_tar(IMAGES_URI)
labels = load_images_from_remote_tar(LABELS_URI)
n_images = len(images)
assert n_images == len(labels), f"{len(images)=}, {len(labels)=}"

print(f"Loaded {n_images} images and labels")

data_index = 0

print(f"Showing image / mask at index {data_index}:")

image = images[data_index]
mask = labels[data_index]
foreground = Image.composite(image, mask, mask)
mask_bg_np = np.copy(np.array(mask))
mask_bg_np[:, :, 3] = 255 - mask_bg_np[:, :, 3]
mask_bg = Image.fromarray(mask_bg_np)
background = Image.composite(image, mask_bg, mask_bg)

data_to_show = [image, mask, foreground, background]
data_labels = ["Image", "Mask", "Foreground", "Background"]

plt.figure(figsize=(16, 4), dpi=300)
for i in range(len(data_to_show)):
    plt.subplot(1, len(data_to_show), i + 1)
    plt.imshow(data_to_show[i])
    plt.axis('off')
    plt.title(data_labels[i], fontsize=12)
plt.show()

# quantization filter for the given patch size
patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

# image resize transform to dimensions divisible by patch size
def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

mask_0 = labels[0].split()[-1]
mask_0_resized = resize_transform(mask_0)
with torch.no_grad():
    mask_0_quantized = patch_quant_filter(mask_0_resized).squeeze().detach().cpu()

plt.figure(figsize=(4, 2), dpi=300)
plt.subplot(1, 2, 1)
plt.imshow(mask_0)
plt.axis('off')
plt.title(f"Original Mask, Size {mask_0.size}", fontsize=5)
plt.subplot(1, 2, 2)
plt.imshow(mask_0_quantized)
plt.axis('off')
plt.title(f"Quantized Mask, Size {tuple(mask_0_quantized.shape)}", fontsize=5)
plt.show()

#combined_path = os.path.join(SAVE_DIR, f"combined_{data_index}.png")
#plt.savefig(combined_path, bbox_inches='tight')

xs = []
ys = []
image_index = []

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

# 使用 Hugging Face 接口提取 patch-level 特征（替换get_intermediate_layers 调用）
xs = []
ys = []
image_index = []

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

with torch.inference_mode():
    for i in tqdm(range(n_images), desc="Processing images"):
        # ground-truth / mask quantization (unchanged)
        mask_i = labels[i].split()[-1]   # alpha channel
        mask_i_resized = resize_transform(mask_i)
        mask_i_quantized = patch_quant_filter(mask_i_resized).squeeze().view(-1).detach().cpu()
        ys.append(mask_i_quantized)

        #image preprocessing 
        image_i = images[i].convert("RGB")
        # resize to patch-aligned tensor (C,H,W) in [0,1]
        image_i_resized = resize_transform(image_i)   # (3, H, W)
        # normalize (ImageNet) 
        image_i_resized = TF.normalize(image_i_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # create batch and move to device
        pixel_values = image_i_resized.unsqueeze(0).to(model.device)  # (1,3,H,W)

        #forward through Hugging Face model, ask for hidden states
        outputs = model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple: (embeddings, layer1, ..., layerN)
        # take last layer hidden states
        last_hidden = hidden_states[-1]  # shape: (1, seq_len, hidden_dim)

        # drop cls token and convert to (num_patches, dim)
        # many ViT implementations put cls token at index 0
        if last_hidden.shape[1] > 1:
            patch_tokens = last_hidden[:, 1:, :]   # (1, P, C)
        else:
            patch_tokens = last_hidden        # (1, P, C)
        patch_tokens = patch_tokens.squeeze(0)    # (P, C)

        # patch alignment fix
        n_feat = patch_tokens.shape[0]
        n_mask = mask_i_quantized.shape[0]
        if n_feat != n_mask:
            min_n = min(n_feat, n_mask)
            print(f"[WARN] image {i}: feature patches={n_feat}, mask patches={n_mask} → trunc to {min_n}")
            patch_tokens = patch_tokens[:min_n, :]
            mask_i_quantized = mask_i_quantized[:min_n]
        # end fix

        # append to xs in same layout as original
        xs.append(patch_tokens.detach().cpu())    # (P, C)
        ys[-1] = mask_i_quantized                 # 更新 ys[-1] 为截断后的版本
        # ---- image index for each patch (CPU tensor) ----
        image_index.append(i * torch.ones(mask_i_quantized.shape))

# concatenate into final tensors 
xs = torch.cat(xs)            # (total_kept_patches, C)
ys = torch.cat(ys)            # (total_patches,)
image_index = torch.cat(image_index)

#  filter only confident patches (unchanged) 
idx = (ys < 0.01) | (ys > 0.99)
xs = xs[idx]
ys = ys[idx]
image_index = image_index[idx]


print("Design matrix of size : ", xs.shape)
print("Label matrix of size : ", ys.shape)


cs = np.logspace(-7, 0, 8)
scores = np.zeros((n_images, len(cs)))

for i in range(n_images):
    print('validation using image_{:02d}.jpg'.format(i+1))
    train_selection = image_index != float(i)
    fold_x = xs[train_selection].numpy()
    fold_y = (ys[train_selection] > 0).long().numpy()
    val_x = xs[~train_selection].numpy()
    val_y = (ys[~train_selection] > 0).long().numpy()

    plt.figure()
    for j, c in enumerate(cs):
        print("training logisitic regression with C={:.2e}".format(c))
        clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(fold_x, fold_y)
        output = clf.predict_proba(val_x)
        precision, recall, thresholds = precision_recall_curve(val_y, output[:, 1])
        s = average_precision_score(val_y, output[:, 1])
        scores[i, j] = s
        plt.plot(recall, precision, label='C={:.1e} AP={:.1f}'.format(c, s*100))

    plt.grid()
    plt.xlabel('recall')
    plt.title('image_{:02d}.jpg'.format(i+1))
    plt.ylabel('precision')
    plt.axis([0, 1, 0, 1])
    plt.legend()
    plt.show()

    save_path = os.path.join(SAVE_DIR_PR, f"pr_image_{i+1:02d}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.figure(figsize=(3, 2), dpi=300)
plt.rcParams.update({
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.labelsize": 5,
})
plt.plot(scores.mean(axis=0))
plt.xticks(np.arange(len(cs)), ["{:.0e}".format(c) for c in cs])
plt.xlabel('data fit C')
plt.ylabel('average AP')
plt.grid()

# 保存结果图 
save_dir = "/mnt/afs/wusize/projects/dinov3/output"  # 结果目录
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "average_mAP_vs_C.png")

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()   # 关闭图像以释放内存

print(f" Saved average mAP curve at: {save_path}")

clf = LogisticRegression(random_state=0, C=0.01, max_iter=100000, verbose=2).fit(xs.numpy(), (ys > 0).long().numpy())

import numpy as np

print("Training finished.")
print("Coefficient shape:", clf.coef_.shape)
print("Intercept:", clf.intercept_)
print("Mean of coefficients:", np.mean(clf.coef_))

test_image_fpath = "https://dl.fbaipublicfiles.com/dinov3/notebooks/foreground_segmentation/test_image.jpg"

def load_image_from_url(url: str) -> Image.Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")

test_image = load_image_from_url(test_image_fpath)

#PATCH_SIZE 必须与模型相匹配
PATCH_SIZE = 16  

resize_transform = torch.nn.Sequential(
    torch.nn.Identity()
)

# 统一到 DINOv3 预训练大小（768x1024）
test_image_resized = TF.resize(test_image, (768, 1024))
test_image_tensor = TF.to_tensor(test_image_resized)
test_image_normalized = TF.normalize(test_image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

# 2. 提取 patch 特征 (Hugging Face 方式)
with torch.no_grad():
    outputs = model(
        pixel_values=test_image_normalized.unsqueeze(0).to(device),
        output_hidden_states=True
    )
    # 取最后一层特征
    last_hidden = outputs.hidden_states[-1]  # shape: (1, num_patches+1, dim)
    feats = last_hidden[:, 1:, :]             # 去掉 CLS token
    feats = feats.squeeze(0).cpu()            # (num_patches, dim)
    print("Feature shape:", feats.shape)

# 3. 调整形状匹配 patch grid
h_patches, w_patches = [
    int(test_image_resized.size[1] / PATCH_SIZE),
    int(test_image_resized.size[0] / PATCH_SIZE)
]
print(f"Expected patches: {h_patches}×{w_patches} = {h_patches*w_patches}")

expected_tokens = h_patches * w_patches
if feats.shape[0] != expected_tokens:
    print(f"[Warning] Patch count mismatch: got {feats.shape[0]}, expected {expected_tokens}")
    feats = feats[:expected_tokens, :]

x = feats.numpy()

# 4. 前景预测 + 中值滤波
fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

# 5. 保存结果图像到服务器
plt.figure(figsize=(9, 3), dpi=300)
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(test_image_resized)
plt.title('Input image')

plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(fg_score)
plt.title('Foreground score')

plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(fg_score_mf)
plt.title('+ Median filter')

save_dir = "/mnt/afs/wusize/projects/dinov3/output/test_interference"  # 结果目录
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "test_interference.png")

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()   # 关闭图像以释放内存

save_root = '/mnt/afs/wusize/projects/dinov3/results'  
os.makedirs(save_root, exist_ok=True)

model_path = os.path.join(save_root, "fg_classifier.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print(f"Classifier saved to {model_path}")

