"""Own 50M model: pad attention head 52 -> 56 only inside parameter-free SDPA."""
import math
import torch.nn.functional as F
from model import Neral20M,CausalSelfAttention
from architectures import FIFTY

class PaddedAttention(CausalSelfAttention):
    def forward(self,x):
        from model import apply_rope
        b,t,c=x.shape;q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        k=k.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        v=v.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        q,k=apply_rope(q,k,self.theta)
        padding=(-self.head_dim)%8
        q,k,v=(F.pad(a,(0,padding)) for a in (q,k,v))
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.,is_causal=True,scale=1/math.sqrt(self.head_dim))
        y=y[...,:self.head_dim].transpose(1,2).contiguous().view(b,t,c)
        return self.out_proj(y)

class Neral50M(Neral20M):
    def __init__(self,cfg=None):
        super().__init__(cfg or FIFTY)
        for block in self.blocks:
            attention=PaddedAttention(self.cfg);attention.load_state_dict(block.attn.state_dict());block.attn=attention
