import os
import random
import shutil

def reduce_dataset(src_img, src_lbl, dest_img, dest_lbl, num_samples):
    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_lbl, exist_ok=True)

    images = os.listdir(src_img)
    
    # ensure we don't exceed available images
    num_samples = min(num_samples, len(images))

    selected = random.sample(images, num_samples)

    for img in selected:
        shutil.copy(os.path.join(src_img, img), os.path.join(dest_img, img))

        label = img.replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(os.path.join(src_lbl, label)):
            shutil.copy(os.path.join(src_lbl, label), os.path.join(dest_lbl, label))


BASE = "D:/BTP/smart-waste-ai/data/YOLO-Waste-Detection-1"

reduce_dataset(BASE + "/train/images", BASE + "/train/labels",
               BASE + "/small/train/images", BASE + "/small/train/labels", 5000)

reduce_dataset(BASE + "/valid/images", BASE + "/valid/labels",
               BASE + "/small/valid/images", BASE + "/small/valid/labels", 1000)

reduce_dataset(BASE + "/test/images", BASE + "/test/labels",
               BASE + "/small/test/images", BASE + "/small/test/labels", 500)

print("✅ Dataset reduced successfully!")