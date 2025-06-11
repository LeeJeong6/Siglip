import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from transformers import SiglipTokenizer,SiglipTextModel

device="cuda" if torch.cuda.is_available() else "cpu"
"""siglip의 text encoder를 파이썬 환경에서 구현했습니다 """


"""
Siglip의 vision model은 아래와 같습니다. 이걸 보고 직접 구현한 코드는 아래와 같습니다
SiglipTextModel(
  (text_model): SiglipTextTransformer(
    (embeddings): SiglipTextEmbeddings(
      (token_embedding): Embedding(32000, 768)
      (position_embedding): Embedding(64, 768)
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
    (final_layer_norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
    (head): Linear(in_features=768, out_features=768, bias=True)
  )
)"""



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
    
class TextEncoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.tokenizer = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")
        
        self.embedding_table = nn.Embedding(32000, 768)
        self.pos_emb_table = nn.Embedding(64, 768)

    def forward(self,text:str):    
        inputs = self.tokenizer(text, return_tensors="pt",padding=True,truncation=True)['input_ids'].to(device) 
        token_emb = self.embedding_table(inputs)
        
        ids = torch.arange(inputs.shape[1],device=device).unsqueeze(0)
        pos_emb = self.pos_emb_table(ids)
        return token_emb + pos_emb
 
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
    
def original_siglip_load(text):
    """Hugging Face에서 불러온 siglip의 vision encoder가 같은 이미지를 받았을 때, 어떤 shape으로 출력하는지를 나타냅니다."""

    tokenizer = SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224")
    model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.shape

class TE(nn.Module):
    def __init__(self):
        super().__init__()
        self.textencoder = TextEncoder()
        self.encoder_model = SiglipEncoder(num_layers=12, head_size=64, num_heads=12)
        self.pooling_head = SiglipPoolingHead(num_heads=12,embed_dim=768)
        

    def forward(self,text):
        text_embedding = self.textencoder(text)
        encoded = self.encoder_model(text_embedding)
        output = self.pooling_head(encoded)
        return output    
    
if __name__ == "__main__":


    text = "Hello World"
    
    head_size=64
    num_heads=12
    embed_dim = 768

    #1.token embedding + positional embedding
    textencoder = TextEncoder()
    text_embedding = textencoder(text)

    #2.Encoding
    encoder_model = SiglipEncoder(num_layers=12, head_size=head_size, num_heads=num_heads)
    encoded = encoder_model(text_embedding)

    #3.condense
    pooling_head = SiglipPoolingHead(num_heads,embed_dim)
    output = pooling_head(encoded)
    print(output.shape)



    #4.combine
    model = TE()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    