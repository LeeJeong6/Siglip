from datasets import load_dataset, load_from_disk
import os
import json
from PIL import Image
from tqdm import tqdm

def download(split):

    ds = load_dataset("clane9/imagenet-100")
    if split=="train":
        ds["train"].save_to_disk("/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/ImageNet-100/train")
        dataset = load_from_disk("/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/ImageNet-100/train")
        label_names = dataset.features["label"].names
        export_dir = "/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Datset/ImageNet-100-export/train"
    else : 
        ds["validation"].save_to_disk("/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/ImageNet-100/val")       
        dataset = load_from_disk("/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/ImageNet-100/val")
        label_names = dataset.features["label"].names
        export_dir = "/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/ImageNet-100-export/val"

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Saving images"):
        image: Image.Image = item["image"]
        label_idx = item["label"]
        label_name = label_names[label_idx]

        label_dir = os.path.join(export_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

        image_filename = f"{idx:06d}.jpg"
        image_path = os.path.join(label_dir, image_filename)
        
        image.save(image_path)
    label_map_path = os.path.join(export_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({i: name for i, name in enumerate(label_names)}, f, indent=2)



if __name__ == "__main__":
    download("val")        
    print(os.listdir("/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Datset/ImageNet-100-export/val"))