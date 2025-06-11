import zipfile
import os

zip_path = "/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/annotations_trainval2017.zip"
extract_dir = "/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/MSCOCO/train2017"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"압축이 '{extract_dir}' 폴더에 풀렸습니다.")