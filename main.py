from textencoder import *
from visionencoder import *

import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
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
    """ImageNet DATA를 불러와줌"""
    def __init__(self,image_folder_path):
        super().__init__()
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

        return image,prompt

def mask(B):
    mask = torch.ones((B,B),dtype=bool)
    mask.fill_diagonal_(0)
    return mask

def Sigmoid_loss(image_vector:torch.tensor,text_vector:torch.tensor)-> int:
    """SIGLIP의 핵심. Sigmoid Loss구현"""
    B = image_vector.shape[0]
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

    

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = VE().to(device)
        self.text_encoder = TE().to(device) 

    def forward(self,image,prompt):
        v_x = self.vision_encoder(image)
        t_x = self.text_encoder(prompt)
        return v_x,t_x    
        

if __name__ == "__main__":

    epochs=100
    batch_size=16
    lr=0.001
    save_dir = "./checkpoint"
    loss_history=[]
    model = Encoder().to(device)
    dataset = ImageNet_DataLoader("./ImageNet-100-export/train")
    train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for step in trange(epochs):
        total_loss=0.0
        for image,prompt in tqdm(train_loader):
            image = image.to(device)
            v_x,t_x = model(image,prompt)
            loss = Sigmoid_loss(v_x,t_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss/len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch: {step+1}, loss: {round(avg_loss,4)}")

    torch.save(model.state_dict(),os.path.join(save_dir,"model.pt"))
    

    plt.plot(range(1, epochs+1), loss_history, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)


    loss_plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()  
