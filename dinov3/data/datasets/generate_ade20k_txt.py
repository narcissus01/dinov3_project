import os

root = "/mnt/afs/wusize/projects/dinov3/dinov3/data/ADE20K"
images_dir = os.path.join(root, "images")

splits = {
    "train": os.path.join(images_dir, "training"),
    "val": os.path.join(images_dir, "validation"),
}

for split, split_dir in splits.items():
    txt_path = os.path.join(root, f"ADE20K_object150_{split}.txt")
    image_files = sorted([
        os.path.join(os.path.basename(split_dir), f)  # e.g. "training/ADE_train_00000001.jpg"
        for f in os.listdir(split_dir)
        if f.endswith(".jpg")
    ])
    with open(txt_path, "w") as f:
        f.write("\n".join(image_files))
    print(f" Generated: {txt_path} ({len(image_files)} images)")
