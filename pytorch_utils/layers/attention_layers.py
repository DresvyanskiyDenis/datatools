from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class ScaledDotProductAttention_MultiHead(nn.Module):

    def __init__(self):
        super(ScaledDotProductAttention_MultiHead, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            raise ValueError("Mask is not supported yet")

        # key, query, value shapes: [batch_size, num_heads, seq_len, dim]
        emb_dim = key.shape[-1]

        # Calculate attention weights
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(emb_dim)

        # masking
        if mask is not None:
            raise ValueError("Mask is not supported yet")

        # Softmax
        attention_weights = self.softmax(attention_weights)

        # modify value
        value = torch.matmul(attention_weights, value)

        return value, attention_weights


class PositionWiseFeedForward(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout:float=0.1):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm= nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # feed-forward network
        x = self.layer_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.layer_2(x)

        return x


class Add_and_Norm(nn.Module):

    def __init__(self, input_dim, dropout:Optional[float]=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)


    def forward(self, x1, x2):
        # add and then norm
        x = x1 + x2
        x = self.layer_norm(x)
        # apply dropout of needed
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return x



class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, num_heads, dropout:Optional[float]=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        self.head_dim = input_dim // num_heads
        self.dropout = dropout

        # initialize weights
        self.query_w = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.keys_w = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.values_w = nn.Linear(input_dim, self.num_heads * self.head_dim, bias=False)
        self.ff_layer_after_concat = nn.Linear(self.num_heads * self.head_dim, input_dim, bias=False)

        self.attention = ScaledDotProductAttention_MultiHead()

        if self.dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        # query, keys, values shapes: [batch_size, seq_len, input_dim]
        batch_size, len_query, len_keys, len_values = queries.size(0), queries.size(1), keys.size(1), values.size(1)

        # linear transformation before attention
        queries = self.query_w(queries).view(batch_size, len_query, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_len, dim]
        keys = self.keys_w(keys).view(batch_size, len_keys, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_len, dim]
        values = self.values_w(values).view(batch_size, len_values, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_len, dim]

        # attention itself
        values, attention_weights = self.attention(queries, keys, values, mask=mask) # values shape:[batch_size, num_heads, seq_len, dim]

        # concatenation
        out = values.transpose(1, 2).contiguous().view(batch_size, len_values, self.num_heads * self.head_dim) # [batch_size, seq_len, num_heads * dim = input_dim]
        # go through last linear layer
        out = self.ff_layer_after_concat(out)

        return out





class EncoderLayer(nn.Module):

    def __init__(self, input_dim, num_heads, dropout:Optional[float]=0.1, positional_encoding:bool=True):
        super(EncoderLayer, self).__init__()
        self.positional_encoding = positional_encoding
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.dropout = dropout

        # initialize layers
        self.self_attention = MultiHeadAttention(input_dim, num_heads, dropout=dropout)
        self.feed_forward = PositionWiseFeedForward(input_dim, input_dim, dropout=dropout)
        self.norm_after_attention = nn.LayerNorm(input_dim)
        self.norm_after_ff = nn.LayerNorm(input_dim)

        # calculate positional encoding
        if self.positional_encoding:
            self.positional_encoding = PositionalEncoding1D(input_dim)



    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # positional encoding
        if self.positional_encoding:
            x = self.positional_encoding(x)

        # multi-head attention
        residual = x
        x = self.self_attention(x, x, x)
        x = self.norm_after_attention(x + residual)

        # feed forward
        residual = x
        x = self.feed_forward(x)
        x = self.norm_after_ff(x + residual)

        return x

class PositionalEncoding1D(nn.Module):
    # taken from: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)




class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe.permute(1,0,2)[:,:x.size(1)]
        return self.dropout(x)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # test of multi-head attention
    torch.manual_seed(12)
    x = torch.randn(2, 10, 512).to(device)
    pos_encoding = PositionalEncoding(512, max_len=10).to(device)
    x = pos_encoding(x)
    print(pos_encoding.pe)
    print('----------')
    print(x)

