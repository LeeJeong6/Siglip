from textencoder import *
from visionencoder import *
from main import *
import torch
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
import json
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
check_point_path = "./checkpoint/model.pt"
model = Encoder().to(device)
model.load_state_dict(torch.load(check_point_path))
model.eval()

vision_encoder = model.vision_encoder
text_encoder = model.text_encoder

batch_size=32
val_dataset = ImageNet_DataLoader("./Dataset/ImageNet-100-export/val",test=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)

all_prompts = [generate_prompt(label) for label in val_dataset.label_names]  
model = TE().to(device)
model.eval()

text_emb_all = []
for prompt in all_prompts:
    with torch.no_grad():
        emb = text_encoder(prompt)             
        emb = emb.to(device)           
        text_emb_all.append(emb)

text_emb_all = torch.stack(text_emb_all, dim=0)  
text_emb_all = F.normalize(text_emb_all, dim=1)  
text_emb_all = text_emb_all.squeeze(1)

correct=0
total=0
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    for image,label_idx in tqdm(val_loader):
        image = image.to(device)
        v_x = vision_encoder(image)
        label_idx = label_idx.to(device)
        matrix = 100* v_x @ text_emb_all.T #B,768 @ 768,100
        acc1,acc5 = accuracy(matrix,label_idx,topk=(1,5))
        top1+=acc1
        top5+=acc5
        n+=image.size(0)
    top1 = (top1/n)*100
    top5 = (top5/n)*100
print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")

