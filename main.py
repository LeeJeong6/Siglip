from textencoder import *
from visionencoder import *

import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
from torch.nn import functional as F
import json
from collections import defaultdict
from random import choice
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_prompt(label):
    return f"a photo of a {label}"

def get_label_mappings(image_folder_path):
    """ 100class label , {label:index}, {index:label} """
    label_names = sorted(os.listdir(image_folder_path)) 
    label2idx = {label: idx for idx, label in enumerate(label_names)} 
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label_names, label2idx, idx2label

class ImageNet_DataLoader(Dataset):
    """ImageNet DATA를 불러와줌. default: train"""
    def __init__(self,image_folder_path,test=False):
        super().__init__()
        self.test = test
        self.image_paths = []
        self.labels = []
        self.label_names, self.label2idx, self.idx2label = get_label_mappings(image_folder_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        for label in self.label_names:
            label_dir = os.path.join(image_folder_path,label)
            for fname in os.listdir(label_dir) : 
                self.image_paths.append(os.path.join(label_dir,fname)) #image_path
                self.labels.append(self.label2idx[label])              #label index
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label_idx = self.labels[idx]
        label_name = self.idx2label[label_idx]
        prompt = generate_prompt(label_name)
        if self.test:
            return image,label_idx       
        
        return image,prompt

class Flickr8k(Dataset):
    def __init__(self, split):
        self.root = "/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/Flicker8k_Dataset"
        self.token_file = os.path.join(self.root, 'text/token.txt')
        if split=="train":
            self.imglist_file = os.path.join(self.root, 'text/trainimage.txt')
        else:
            self.imglist_file = os.path.join(self.root, 'text/testimage.txt')

        with open(self.imglist_file, 'r') as f:
            self.imglist = [line.strip() for line in f.readlines()]
          
        self.img_caption_dict = {}
        for imgname in self.imglist:
            self.img_caption_dict[imgname] = []
        with open(self.token_file, 'r') as f:
            for line in f.readlines():
                img_name, caption = line.strip().split('\t')
                img_name = img_name.split('#')[0]
                if img_name in self.img_caption_dict:
                    self.img_caption_dict[img_name].append(caption)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    def __getitem__(self, idx):
        img_path = self.imglist[idx]
        image = Image.open(os.path.join(self.root, "image", img_path)).convert("RGB")
        image = self.transform(image)
        caption = self.img_caption_dict[img_path][0]
        return image, caption

    def __len__(self):
        return len(self.img_caption_dict)
    
class COCOLoader(Dataset):

    def __init__(self,image_folder):
        super().__init__()
        with open('/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/MSCOCO/train2017/annotations/captions_train2017.json') as f:
            coco = json.load(f)
        self.id_to_filename = {img['id']:img['file_name'] for img in coco['images']}    
        self.caption_dict = defaultdict(list)
        self.image_folder = image_folder
        self.transform = self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
       
        for ann in coco['annotations']:
            self.caption_dict[ann['image_id']].append(ann['caption'])
        self.img_ids = list(self.caption_dict.keys())        

    def __len__(self):
        return  len(self.img_ids)
    
    def __getitem__(self,idx):
        image_id = self.img_ids[idx]
        file_name = self.id_to_filename[image_id]
        caption = choice(self.caption_dict[image_id])
        image = Image.open(os.path.join(self.image_folder,file_name)).convert("RGB")
        image = self.transform(image)
        
        return image, caption
    
def mask(B):
    mask = torch.ones((B,B),dtype=bool)
    mask.fill_diagonal_(0)
    return mask

def normalize(v_x:torch.tensor,t_x:torch.tensor):
    """L2 Normalization"""
    v_x = F.normalize(v_x, dim=-1)
    t_x = F.normalize(t_x, dim=-1)
    return v_x,t_x


def Sigmoid_loss(image_vector:torch.tensor,text_vector:torch.tensor)-> int:
    """SIGLIP의 핵심. Sigmoid Loss구현"""
    """기존에 내가 짠 코드임. 내적 값을 출력해보면 모든 값이 다 똑같이 나오는 문제점이 있었음.
    siglip은 -log(sigmoid(x)) 라는 loss function으로 내적값이 pos-pair는 최대한 크게 neg-pair는 최대한 작게 나오게끔 하는 게 목표였는데
    이 코드처럼 해버리면 내적값이 0으로 가게함. 즉 0,1로 하는 binary classification에서는 기능을 못하게 하는거였음
    loss를 몇번 업데이트 하면 내적값이 다 똑같이 나와버리는 문제가 있었음
    이 문제는 binary class로 해버리니까 모델이 y를 대부분 0으로 보내버리는 데 학습이 됐기 때문임
    모든 내적값이 다 작게끔 학습이 되어버리니까 모든 벡터들이 다 서로 비슷하게끔 학습이 된 듯하다.
    """
    B = image_vector.shape[0]
    image_vector,text_vector = normalize(image_vector,text_vector)
    dot_product = image_vector @ text_vector.T #(B,768) @ (768,B) -> (B,B)
    sigmoid = torch.sigmoid(dot_product)
    m = mask(B)
    negative = sigmoid[m].reshape(B,-1)
    positive = torch.diag(sigmoid).unsqueeze(1)
    pair = torch.cat((positive,negative),dim=1)
    y = torch.zeros_like(pair)
    y[:,0]=1


    criterion = nn.BCELoss()
    loss = criterion(pair,y)
    return loss

    
def Sigmoid_loss_GPT(image_vector:torch.tensor,text_vector:torch.tensor,scale=1.0)-> int:
    """Log-sigmoid 기반 contrastive loss (SIGLIP 스타일)"""

    image_vector = F.normalize(image_vector, dim=-1)
    text_vector = F.normalize(text_vector, dim=-1)

    logits = scale * (image_vector @ text_vector.T)  # (B, B)
    labels = torch.full_like(logits, -1.0)
    labels.fill_diagonal_(1.0)

    loss_matrix = F.logsigmoid(labels * logits)  
    loss = -loss_matrix.mean()

    return loss
    

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = VE().to(device)
        self.text_encoder = TE().to(device) 

    def forward(self,image,prompt):
        v_x = self.vision_encoder(image)
        t_x = self.text_encoder(prompt)
        return v_x,t_x    
        
def flicker8k_train():
    epochs=1
    batch_size=32
    lr=0.01
    save_dir = "./checkpoint/flicker8k"
    loss_history=[]
    model = Encoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

    train_dataset = Flickr8k("train")    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    loss_history=[]
    for step in trange(epochs):
        total_loss=0.0
        for image,prompt in tqdm(train_loader):
            image = image.to(device)
            v_x,t_x = model(image,prompt)

            loss = Sigmoid_loss_GPT(v_x,t_x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(loss)

        avg_loss = total_loss/len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch: {step+1}, Train loss: {avg_loss:.4f}")


    torch.save(model.state_dict(),os.path.join(save_dir,"model.pt"))
    
    plt.plot(range(1, epochs+1), loss_history, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)


    loss_plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()  
    return print("Flicker8k학습 완료 ")

def MSCOCO_train():
    epochs=10
    batch_size=512
    lr=0.01
    save_dir = "./checkpoint/mscoco"
    loss_history=[]
    model = Encoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    train_dataset = COCOLoader("/mnt/hdd_6tb/bill0914/VLM_Research/SigLip/Dataset/MSCOCO/train2017/image")
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)        
    loss_history=[]
    for step in trange(epochs):
        total_loss=0.0
        for image,prompt in tqdm(train_loader):
            image = image.to(device)
            v_x,t_x = model(image,prompt)
            loss = Sigmoid_loss_GPT(v_x,t_x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(loss)

        avg_loss = total_loss/len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch: {step+1}, Train loss: {avg_loss:.4f}")


    torch.save(model.state_dict(),os.path.join(save_dir,"model.pt"))
    
    plt.plot(range(1, epochs+1), loss_history, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)


    loss_plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()  
    return print("MSCOCO학습 완료 ")
if __name__ == "__main__": 
    MSCOCO_train()

    














