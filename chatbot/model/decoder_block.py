import torch
import torch.nn as nn
from torch.nn import functional as F
from multihead_attention import MultiHeadAttention
from feed_forward import FeedForward


class DecoderBlock(nn.Module):
    # Transformer block: communication followed by computation
    
    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimension, n_head: the number of head we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x