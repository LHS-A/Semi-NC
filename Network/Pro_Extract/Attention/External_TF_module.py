import torch
import torch.nn as nn

class MEAttention(nn.Module):
    def __init__(self, dim, configs_head=8):
        super(MEAttention, self).__init__()
        self.num_heads = configs_head
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
    
        query_key_dim = dim * self.coef // self.num_heads  # 128 * 4 / 32 = 16
        self.linear_0 = nn.Linear(query_key_dim, self.k)
        self.key_liner = nn.Linear(dim, dim * self.coef)
        
        merge_dim = dim * self.coef * 2  # 128 * 4 * 2 = 1024
        self.linear = nn.Linear(merge_dim, dim * self.coef)  # 1024 -> 512
        
        self.linear_1 = nn.Linear(self.k, query_key_dim)
        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, src1, src2):
        B1, N1, C1 = src1.shape
        B2, N2, C2 = src2.shape
        
        query = self.query_liner(src1)  # [B1, N1, dim*coef] = [B1, N1, 512]
        key = self.key_liner(src2)      # [B2, N2, dim*coef] = [B2, N2, 512]
        
        if B1 != B2 or N1 != N2:
            if B1 != B2:
                key = key.expand(B1, -1, -1)
            if N1 != N2:
                key = key[:, :N1, :]
        
        merge = torch.cat([query, key], dim=2)  # [B1, N1, 1024]
        attn = self.linear(merge)              # [B1, N1, 512]
        
        attn = attn.view(B1, N1, self.num_heads, -1).permute(0, 2, 1, 3)  # [B1, num_heads, N1, query_key_dim]
        attn = self.linear_0(attn)              # [B1, num_heads, N1, k]
        
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        
        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B1, N1, -1)  # [B1, N1, 512]
        x = self.proj(x)  # [B1, N1, dim] = [B1, N1, 128]
        
        return x