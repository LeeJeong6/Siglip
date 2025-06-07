import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel,SiglipVisionConfig,SiglipTokenizer
"""siglip의 vision encoder를 파이썬 환경에서 구현했습니다 """


"""
Siglip의 vision model은 아래와 같습니다. 이걸 보고 직접 구현한 코드는 아래와 같습니다
SiglipVisionModel(
  (vision_model): SiglipVisionTransformer(
    (embeddings): SiglipVisionEmbeddings(
      (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), padding=valid)
      (position_embedding): Embedding(196, 768)
    )
    (encoder): SiglipEncoder(
                            (layers): ModuleList(
                                                (0-11): 12 x SiglipEncoderLayer(
                                                                                (layer_norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                                                                                (self_attn): SiglipAttention(
                                                                                                              (k_proj): Linear(in_features=768, out_features=768, bias=True)
                                                                                                              (v_proj): Linear(in_features=768, out_features=768, bias=True)
                                                                                                              (q_proj): Linear(in_features=768, out_features=768, bias=True)
                                                                                                              (out_proj): Linear(in_features=768, out_features=768, bias=True)
                                                                                                            )
                                                                                (layer_norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                                                                                (mlp): SiglipMLP(
                                                                                  (activation_fn): PytorchGELUTanh()
                                                                                  (fc1): Linear(in_features=768, out_features=3072, bias=True)
                                                                                  (fc2): Linear(in_features=3072, out_features=768, bias=True)
                                                                                                )
                                                                                  )
                                                  )
                             )
                            (post_layernorm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
                            ) 
)"""
def preprocess_image(image,image_size=224):
    """(3,H,W) -> (1,3,224,224)"""
    preprocess = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229,0.224,0.225])
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


class SigLipVisionEmbedding(nn.Module):
    def __init__(self,embed_dim,patch_size,image_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.patch_conv_layer = nn.Conv2d(in_channels=3,
                                        out_channels=embed_dim, #768
                                        kernel_size=patch_size, #16
                                        stride = patch_size)    #16
                                
        self.num_patches= (image_size//patch_size)**2

        self.positional_embedding_table = nn.Embedding(self.num_patches,embed_dim)

    def forward(self,x):
        # x: (B, 3, H, W)
        x = self.patch_conv_layer(x) # (B, E, H/P, W/P)
        
        B,E,H,W = x.shape
        device = x.device
        
        x = x.flatten(2).transpose(1,2)  # (B, num_patches, E)
        pos_emb = self.positional_embedding_table(torch.arange(0,self.num_patches,device=device)).unsqueeze(0) # (1, num_patches,E)

        out = x + pos_emb #broadcasting 
        return out # (B,num_patches,E)

class SiglipAttention(nn.Module):
    """attention matrix"""
    def __init__(self,head_size):
        super().__init__()
        self.q = nn.Linear(768,head_size,bias=True)
        self.k = nn.Linear(768,head_size,bias=True)
        self.v = nn.Linear(768,head_size,bias=True)
        
    def forward(self,x):
        q = self.q(x)    
        k = self.k(x)
        v = self.v(x)
        wei = q @ k.transpose(-2,-1) *(q.shape[-1]**-0.5)
        wei = F.softmax(wei,dim=-1)
        out = wei@v
        
        return out


class SiglipEncoderLayer(nn.Module):
    """multihead attention + layernorm + MLP"""
    def __init__(self,head_size,num_head=12):
        super().__init__()
        self.num_head=num_head
        self.head_size=head_size
        self.heads = nn.ModuleList([SiglipAttention(head_size) for _ in range(num_head)]) 
        self.lan1 = nn.LayerNorm(768,eps=1e-06, elementwise_affine=True)
        self.lan2 = nn.LayerNorm(768,eps=1e-06, elementwise_affine=True)
        self.proj = nn.Linear(768,768,bias=True)
        self.mlp = SiglipMLP()

    def forward(self,x):
        x = self.lan1(x)
        x = torch.cat([h(x) for h in self.heads],dim=-1)
        x = self.proj(x)
        out = self.lan2(x)
        out = self.mlp(out)
        return out       
    
class SiglipMLP(nn.Module):
    """MLP"""
    def __init__(self, embed_dim=768, hidden_dim=3072):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                      torch.nn.GELU(),  
                      nn.Linear(hidden_dim, embed_dim))
        

    def forward(self, x):
        return self.layers(x)
    

class SiglipEncoder(nn.Module):
    """Main Class"""
    def __init__(self, num_layers=12, head_size=64, num_heads=12):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(head_size=head_size, num_head=num_heads) for _ in range(num_layers)])
        
        self.post_layernorm = nn.LayerNorm(768, eps=1e-06, elementwise_affine=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        return x
    
class SiglipPoolingHead(nn.Module):
    """(B,196,768) -> (B,1,768)으로 정보를 압축하는 과정
       (1, 1, 768)의 학습가능한 파라미터 self.query는 multiheadattention을 통해 
       x의 정보를 압축하는 과정을 거치게 됨. """
    def __init__(self,num_heads,embed_dim=768):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.query = nn.Parameter(torch.randn(1,1,embed_dim))
        self.layernorm = nn.LayerNorm(embed_dim,eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim,3072),
            nn.GELU(),
            nn.Linear(3072,embed_dim)
        )

    def forward(self,x):
        B = x.size(0)
        x = self.layernorm(x)
        query = self.query.expand(B,-1,-1)
        out,_ = self.attention(query,x,x)
        out = self.mlp(out)
            
        return out.squeeze(1)
    
def original_siglip_load(image):
    """Hugging Face에서 불러온 siglip의 vision encoder가 같은 이미지를 받았을 때, 어떤 shape으로 출력하는지를 나타냅니다."""

    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    vision_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224",config = SiglipVisionConfig(vision_use_head=False))
    inputs = processor(images=image, return_tensors="pt")
    outputs = vision_model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    return last_hidden_state.shape


class VE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_model = SigLipVisionEmbedding(embed_dim=768, patch_size=16, image_size=224)
        self.encoder_model = SiglipEncoder(num_layers=12, head_size=64, num_heads=12)
        self.pooling_head = SiglipPoolingHead(num_heads=12,embed_dim=768)
        
    def forward(self,image_tensor):
        embedded = self.embedding_model(image_tensor)
        encoded = self.encoder_model(embedded)
        output = self.pooling_head(encoded)

        return output    
    
if __name__ == "__main__":
    image = Image.open("./cat.jpg")
    
    image_size = 224
    embed_dim = 768
    patch_size = 16
    num_heads=12

    #1.데이터 전처리
    image_tensor = preprocess_image(image, image_size)

    #2.token embedding + positional embedding
    embedding_model = SigLipVisionEmbedding(embed_dim, patch_size, image_size)
    embedded = embedding_model(image_tensor)

    #3.Encoding 
    encoder_model = SiglipEncoder(num_layers=12, head_size=64, num_heads=num_heads)
    encoded = encoder_model(embedded)

    #4.condense 
    pooling_head = SiglipPoolingHead(num_heads,embed_dim)
    output = pooling_head(encoded)
    print(output.shape)


    #5.combine
    model = VE()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

 