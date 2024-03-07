import torch
import torch.nn as nn
from torch import Tensor, BoolTensor

from typing import Optional, Tuple, Iterable
from model.diffusion import SinusoidalPosEmb
from torch.nn.functional import pad


import math

def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions


class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-Head Self-Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head: The number of heads.
        d: The number of features.
        dropout: Dropout rate.
        zero_triu: Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(
            self, d: int, n_head: int, dropout: float
    ):
        super().__init__()
        assert d % n_head == 0
        self.c = d // n_head
        self.h = n_head

        self.linear_q = nn.Linear(d, d)
        self.linear_k = nn.Linear(d, d)
        self.linear_v = nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)

        self.p_attn = None
        self.dropout = nn.Dropout(p=dropout)

        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(d, d, bias=False)

        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.u = nn.Parameter(torch.Tensor(self.h, self.c))
        self.v = nn.Parameter(torch.Tensor(self.h, self.c))
        # [H, C]
        torch.nn.init.xavier_uniform_(self.u)
        torch.nn.init.xavier_uniform_(self.v)

    def forward_qkv(self, query, key, value) -> Tuple[Tensor, ...]:
        """Transform query, key and value.
        Args:
            query (Tensor): [B, S, D].
            key (Tensor): [B, T, D].
            value (Tensor): [B, T, D].
        Returns:
            q (Tensor): [B, H, S, C].
            k (Tensor): [B, H, T, C].
            v (Tensor): [B, H, T, C].
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.c)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.c)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.c)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(self, v, scores, mask, causal=False) -> Tensor:
        """Compute attention context vector.
        Args:
            v (Tensor): [B, H, T, C].
            scores (Tensor): [B, H, S, T].
            mask (BoolTensor): [B, T], True values are masked from scores.
        Returns:
            result (Tensor): [B, S, D]. Attention result weighted by the score.
        """
        n_batch, H, S, T = scores.shape
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2).to(bool),
                float("-inf"),  # [B, H, S, T]
            )
        if causal:
            k_grid = torch.arange(0, S, dtype=torch.int32, device=scores.device)
            v_grid = torch.arange(0, T, dtype=torch.int32, device=scores.device)
            kk, vv = torch.meshgrid(k_grid, v_grid, indexing="ij")
            causal_mask = vv > kk
            scores = scores.masked_fill(
                causal_mask.view(1, 1, S, T), float("-inf")
            )

        p_attn = self.p_attn = torch.softmax(scores, dim=-1)  # [B, H, S, T]
        p_attn = self.dropout(p_attn)  # [B, H, S, T]

        x = torch.matmul(p_attn, v)  # [B, H, S, C]
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.c)
        )  # [B, S, D]

        return self.linear_out(x)  # [B, S, D]

    def rel_shift(self, x):
        """Converting (..., i, i - j) matrix into (..., i, j) matrix.
        Args:
            x (Tensor): [B, H, S, 2S-1].
        Returns:
            x (Tensor): [B, H, S, S].
        Example: Take S = 2 for example, larger values work similarly.
        x = [
            [(0, -1), (0, 0), (0, 1)],
            [(1, 0),  (1, 1), (1, 2)]
        ]
        x_padded = [
            [(x, x), (0, -1), (0, 0), (0, 1)],
            [(x, x), (1, 0),  (1, 1), (1, 2)]]
        ]
        x_padded = [
            [(x, x), (0, -1)],
            [(0, 0), (0, 1)],
            [(x, x), (1, 0)],
            [(1, 1), (1, 2)]
        ]
        x = [
            [(0, 0), (0, 1)],
            [(1, 0), (1, 1)]
        ]
        """
        B, H, S, _ = x.shape
        zero_pad = torch.zeros((B, H, S, 1), device=x.device, dtype=x.dtype)
        # [B, H, S, 1]
        x_padded = torch.cat([zero_pad, x], dim=-1)
        # [B, H, S, 2S]
        x_padded = x_padded.view(B, H, 2 * S, S)
        # [B, H, 2S, S]
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, :S]
        # only keep the positions from 0 to S
        # [B, H, 2S-1, S] <view> [B, H, S, 2S - 1] <truncate in dim -1> [B, H, S, S]
        return x

    def forward(
            self, query, key, value, pos_emb, mask=None, causal=False):
        """Compute self-attention with relative positional embedding.
        Args:
            query (Tensor): [B, S, D].
            key (Tensor): [B, S, D].
            value (Tensor): [B, S, D].
            pos_emb (Tensor): [1/B, 2S-1, D]. Positional embedding.
            mask (BoolTensor): [B, S], True for masked.
            causal (bool): True for applying causal mask.
        Returns:
            output (Tensor): [B, S, D].
        """
        # Splitting Q, K, V:
        q, k, v = self.forward_qkv(query, key, value)
        # [B, H, S, C], [B, H, S, C], [B, H, S, C]

        # Adding per head & channel biases to the query vectors:
        q_u = q + self.u.unsqueeze(1)
        q_v = q + self.v.unsqueeze(1)
        # [B, H, S, C]

        # Splitting relative positional coding:
        n_batch_pos = pos_emb.size(0)  # [1/B, 2S-1, D]
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.c)
        # [1/B, 2S-1, H, C]
        p = p.transpose(1, 2)  # [1/B, H, 2S-1, C].

        # Compute query, key similarity:
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
        # [B, H, S, C] x [B, H, C, S] -> [B, H, S, S]

        matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
        # [B, H, S, C] x [1/B, H, C, 2S-1] -> [B, H, S, 2S-1]
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.c)
        # [B, H, S, S]

        return self.forward_attention(v, scores, mask, causal)  # [B, S, D]


class ConditionalBiasScale(nn.Module):
    def __init__(self, channels: int, cond_channels: int):
        super().__init__()
        self.scale_transform = nn.Linear(
            cond_channels, channels, bias=True
        )
        self.bias_transform = nn.Linear(
            cond_channels, channels, bias=True
        )
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.constant_(self.scale_transform.weight, 0.0)
        torch.nn.init.constant_(self.scale_transform.bias, 1.0)
        torch.nn.init.constant_(self.bias_transform.weight, 0.0)
        torch.nn.init.constant_(self.bias_transform.bias, 0.0)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Applying conditional bias and scale.
        Args:
            x (Tensor): [..., channels].
            cond (Tensor): [..., cond_channels].
        Returns:
            y (Tensor): [..., channels].
        """
        a = self.scale_transform.forward(cond)
        b = self.bias_transform.forward(cond)
        return x * a + b


class FeedForwardModule(torch.nn.Module):
    """Positionwise feed forward layer used in conformer"""

    def __init__(
            self, d_in: int, d_hidden: int,
            dropout: float, bias: bool = True, d_cond: int = 0
    ):
        """
        Args:
            d_in (int): Input feature dimension.
            d_hidden (int): Hidden unit dimension.
            dropout (float): dropout value for first Linear Layer.
            bias (bool): If linear layers should have bias.
            d_cond (int, optional): The channels of conditional tensor.
        """
        super(FeedForwardModule, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(d_in)

        if d_cond > 0:
            self.cond_layer = ConditionalBiasScale(d_in, d_cond)

        self.w_1 = torch.nn.Linear(d_in, d_hidden, bias=bias)
        self.w_2 = torch.nn.Linear(d_hidden, d_in, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): [..., D].
        Returns:
            y (Tensor): [..., D].
            cond (Tensor): [..., D_cond]
        """
        x = self.layer_norm(x)

        if cond is not None:
            x = self.cond_layer.forward(x, cond)

        x = self.w_1(x)
        x = silu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return self.dropout(x)


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding cache.

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Default maximum input length.
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.cached_code = None
        self.l = 0
        self.gen_code(torch.tensor(0.0).expand(1, max_len))

    def gen_code(self, x: Tensor):
        """Generate positional encoding with a reference tensor x.
        Args:
            x (Tensor): [B, L, ...], we extract the device, length, and dtype from it.
        Effects:
            self.cached_code (Tensor): [1, >=(2L-1), D].
        """
        l = x.size(1)
        if self.l >= l:
            if self.cached_code.dtype != x.dtype or self.cached_code.device != x.device:
                self.cached_code = self.cached_code.to(dtype=x.dtype, device=x.device)
            return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        code_pos = torch.zeros(l, self.d_model)  # [L, D]
        code_neg = torch.zeros(l, self.d_model)  # [L, D]
        pos = torch.arange(0, l, dtype=torch.float32).unsqueeze(1)  # [L, 1]
        decay = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )  # [D // 2]
        code_pos[:, 0::2] = torch.sin(pos * decay)
        code_pos[:, 1::2] = torch.cos(pos * decay)
        code_neg[:, 0::2] = torch.sin(-1 * pos * decay)
        code_neg[:, 1::2] = torch.cos(-1 * pos * decay)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        code_pos = torch.flip(code_pos, [0]).unsqueeze(0)  # [1, L, D]
        code_neg = code_neg[1:].unsqueeze(0)  # [1, L - 1, D]
        code = torch.cat([code_pos, code_neg], dim=1)  # [1, 2L - 1, D]
        self.cached_code = code.to(device=x.device, dtype=x.dtype)
        self.l = l

    def forward(self, x: Tensor) -> Tensor:
        """Get positional encoding of appropriate shape given a reference Tensor.
        Args:
            x (Tensor): [B, L, ...].
        Returns:
            y (Tensor): [1, 2L-1, D].
        """
        self.gen_code(x)
        l = x.size(1)
        pos_emb = self.cached_code[
                  :, self.l - l: self.l + l - 1,
                  ]
        return pos_emb


class ConformerBlock(torch.nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(
            self, d: int, d_hidden: int,
            attention_heads: int, dropout: float,
            depthwise_conv_kernel_size: int = 7,
            causal: bool = False, d_cond: int = 0
    ):
        """
        Args:
            d (int): Block input output channel number.
            d_hidden (int): FFN layer dimension.
            attention_heads (int): Number of attention heads.
            dropout (float): dropout value.
            depthwise_conv_kernel_size (int): Size of kernel in depthwise conv.
            d_cond (int, optional): The channels of conditional tensor.
        """
        super(ConformerBlock, self).__init__()
        self.causal = causal
        self.ffn1 = FeedForwardModule(
            d, d_hidden, dropout, bias=True, d_cond=d_cond
        )

        self.self_attn_layer_norm = torch.nn.LayerNorm(d)

        if d_cond > 0:
            self.cond_layer = ConditionalBiasScale(d, d_cond)

        self.self_attn = RelPositionMultiHeadedAttention(
            d, attention_heads, dropout=dropout
        )
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = ConvolutionModule(
            d_in=d, d_hidden=d,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout, d_cond=d_cond
        )

        self.ffn2 = FeedForwardModule(
            d, d_hidden, dropout, bias=True, d_cond=d_cond
        )

        self.final_layer_norm = torch.nn.LayerNorm(d)

    def forward(
            self, x: Tensor, mask: BoolTensor, pos_emb: Tensor,
            cond: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x (Tensor): [B, T, D_in].
            mask (BoolTensor): [B, T], True for masked.
            pos_emb (Tensor): [1 or B, 2T-1, D].
            cond (Tensor, optional): [B, ?, D_cond].
        Returns:
            y (Tensor): [B, T, D_in].
        """
        y = x

        x = self.ffn1(x) * 0.5 + y
        y = x
        # [B, T, D_in]

        x = self.self_attn_layer_norm(x)

        if cond is not None:
            x = self.cond_layer.forward(x, cond)

        x = self.self_attn.forward(
            query=x, key=x, value=x,
            pos_emb=pos_emb,
            mask=mask, causal=self.causal
        )
        x = self.self_attn_dropout(x) + y
        y = x
        # [B, T, D_in]

        x = self.conv_module.forward(x, mask) + y
        y = x
        # [B, T, D_in]

        x = self.ffn2(x) * 0.5 + y

        x = self.final_layer_norm(x)

        x.masked_fill(mask.unsqueeze(-1), 0.0)

        return x


class ConvolutionModule(torch.nn.Module):
    """Convolution Block inside a Conformer Block."""

    def __init__(
            self, d_in: int, d_hidden: int,
            depthwise_kernel_size: int,
            dropout: float, bias: bool = False,
            causal: bool = False, d_cond: int = 0
    ):
        """
        Args:
            d_in (int): Embedding dimension.
            d_hidden (int): Number of channels in depthwise conv layers.
            depthwise_kernel_size (int): Depthwise conv layer kernel size.
            dropout (float): dropout value.
            bias (bool): If bias should be added to conv layers.
            conditional (bool): Whether to use conditional LayerNorm.
        """
        super(ConvolutionModule, self).__init__()
        assert (depthwise_kernel_size - 1) % 2 == 0, "kernel_size should be odd"
        self.causal = causal
        self.causal_padding = (depthwise_kernel_size - 1, 0)
        self.layer_norm = torch.nn.LayerNorm(d_in)

        # Optional conditional LayerNorm:
        self.d_cond = d_cond
        if d_cond > 0:
            self.cond_layer = ConditionalBiasScale(d_in, d_cond)

        self.pointwise_conv1 = torch.nn.Conv1d(
            d_in, 2 * d_hidden,
            kernel_size=1,
            stride=1, padding=0,
            bias=bias
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            d_hidden, d_hidden,
            kernel_size=depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2 if not causal else 0,
            groups=d_hidden, bias=bias
        )
        self.pointwise_conv2 = torch.nn.Conv1d(
            d_hidden, d_in,
            kernel_size=1,
            stride=1, padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: BoolTensor, cond: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): [B, T, D_in].
            mask (BoolTensor): [B, T], True for masked.
            cond (Tensor): [B, T, D_cond].
        Returns:
            y (Tensor): [B, T, D_in].
        """
        x = self.layer_norm(x)

        if cond is not None:
            x = self.cond_layer.forward(x, cond)

        x = x.transpose(-1, -2)  # [B, D_in, T]

        x = self.pointwise_conv1(x)  # [B, 2C, T]
        x = self.glu(x)  # [B, C, T]

        # Take care of masking the input tensor:
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(1), 0.0)

        # 1D Depthwise Conv
        if self.causal:  # Causal padding
            x = pad(x, self.causal_padding)
        x = self.depthwise_conv(x)
        # FIXME: BatchNorm should not be used in variable length training.
        x = silu(x)  # [B, C, T]

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(1), 0.0)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(-1, -2)  # [B, T, D_in]


class Conformer(torch.nn.Module):
    def __init__(
            self,
            d: int,
            d_hidden: int,
            n_heads: int,
            n_layers: int,
            dropout: float,
            depthwise_conv_kernel_size: int,
            causal: bool = False,
            d_cond: int = 0
    ):
        super().__init__()
        self.pos_encoding = RelPositionalEncoding(1024, d)
        self.causal = causal

        self.blocks = torch.nn.ModuleList(
            [
                ConformerBlock(
                    d=d,
                    d_hidden=d_hidden,
                    attention_heads=n_heads,
                    dropout=dropout,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    causal=causal,
                    d_cond=d_cond
                )
                for _ in range(n_layers)
            ]
        )  # type: Iterable[ConformerBlock]

    def forward(
            self, x: Tensor, mask: BoolTensor, cond: Tensor = None
    ) -> Tensor:
        """Conformer forwarding.
        Args:
            x (Tensor): [B, T, D].
            mask (BoolTensor): [B, T], with True for masked.
            cond (Tensor, optional): [B, T, D_cond].
        Returns:
            y (Tensor): [B, T, D]
        """
        pos_emb = self.pos_encoding(x)  # [1, 2T-1, D]

        for block in self.blocks:
            x = block.forward(x, mask, pos_emb, cond)

        return x


class CNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, cond_dim, kernel_size, stride):
        super(CNNBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, stride),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim,),
            nn.Dropout(p=dropout)
        )

    def forward(self, inp):
        out = self.layers(inp)
        return out


class CNNClassifier(nn.Module):
    def __init__(self, in_dim, d_decoder, decoder_dropout, cond_dim):
        super(CNNClassifier, self).__init__()
        self.cnn = nn.Sequential(
            CNNBlock(in_dim, d_decoder, decoder_dropout, cond_dim, 8, 4),
            CNNBlock(d_decoder, d_decoder, decoder_dropout, cond_dim, 8, 4),
            CNNBlock(d_decoder, d_decoder, decoder_dropout, cond_dim, 4, 2),
            CNNBlock(d_decoder, d_decoder, decoder_dropout, cond_dim, 4, 2),
        )  # receptive field is 180, frame shift is 64
        self.cond_layer = nn.Sequential(
            nn.Linear(cond_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim)
        )

    def forward(self, inp, mask, cond):
        inp = inp.transpose(-1, -2)
        cond = cond.transpose(-1, -2)
        inp.masked_fill_(mask.unsqueeze(1), 0.0)
        cond = self.cond_layer(cond.transpose(-1, -2)).transpose(-1, -2)
        cond.masked_fill_(mask.unsqueeze(1), 0.0)
        inp = inp + cond
        return self.cnn(inp)


class CNNClassifierWithTime(nn.Module):
    def __init__(self, in_dim, d_decoder, decoder_dropout, cond_dim, time_emb_dim=512):
        super(CNNClassifierWithTime, self).__init__()
        self.cnn = nn.Sequential(
            CNNBlock(in_dim, d_decoder, decoder_dropout, cond_dim, 8, 4),
            CNNBlock(d_decoder, d_decoder, decoder_dropout, cond_dim, 8, 4),
            CNNBlock(d_decoder, d_decoder, decoder_dropout, cond_dim, 4, 2),
            CNNBlock(d_decoder, d_decoder, decoder_dropout, cond_dim, 4, 2),
        )  # receptive field is 180, frame shift is 64
        self.cond_layer = nn.Sequential(
            nn.Linear(cond_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim)
        )
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.time_layer = nn.Sequential(
            nn.Linear(time_emb_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim)
        )

    def forward(self, inp, mask, cond, t):
        time_emb = self.time_emb(t)  # [B, T]
        time_emb = self.time_layer(time_emb.unsqueeze(1)).transpose(-1, -2)
        inp = inp.transpose(-1, -2)
        cond = cond.transpose(-1, -2)
        inp.masked_fill_(mask.unsqueeze(1), 0.0)
        cond = self.cond_layer(cond.transpose(-1, -2)).transpose(-1, -2)
        cond.masked_fill_(mask.unsqueeze(1), 0.0)
        inp = inp + cond + time_emb
        return self.cnn(inp)


class SpecClassifier(nn.Module):
    def __init__(self, in_dim, d_decoder, h_decoder,
                 l_decoder, decoder_dropout,
                 k_decoder, n_class, cond_dim, model_type='conformer'):
        super(SpecClassifier, self).__init__()
        self.model_type = model_type
        self.prenet = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=d_decoder)
        )
        if model_type == 'conformer':
            self.conformer = Conformer(d=d_decoder, d_hidden=d_decoder, n_heads=h_decoder,
                                       n_layers=l_decoder, dropout=decoder_dropout,
                                       depthwise_conv_kernel_size=k_decoder, d_cond=cond_dim)
        elif model_type == 'CNN':
            self.conformer = CNNClassifier(in_dim=d_decoder, d_decoder=d_decoder,
                                           decoder_dropout=decoder_dropout, cond_dim=cond_dim)
        elif model_type == 'CNN-with-time':
            self.conformer = CNNClassifierWithTime(in_dim=d_decoder, d_decoder=d_decoder,
                                                   decoder_dropout=decoder_dropout, cond_dim=cond_dim, time_emb_dim=256)
        self.classifier = nn.Linear(d_decoder, n_class)

    def forward(self, noisy_mel, condition, mask, **kwargs):
        """
        Args:
            noisy_mel: [B, T, D]
            condition: [B, T, D]
            mask: [B, T] with True for un-masked (real-values)

        Returns:
            classification logits (un-softmaxed)
        """
        # print(noisy_mel.shape)
        noisy_mel = noisy_mel.masked_fill(~mask.unsqueeze(-1), 0.0)

        # print(self.prenet, noisy_mel.shape)
        hiddens = self.prenet(noisy_mel)

        if self.model_type == 'CNN-with-time':
            hiddens = self.conformer.forward(hiddens, ~mask, condition, kwargs['t'])
        else:
            hiddens = self.conformer.forward(hiddens, ~mask, condition)  # [B, T, D]

        if self.model_type == 'conformer':
            averaged_hiddens = torch.mean(hiddens, dim=1)  # [B, D]
            logits = self.classifier(averaged_hiddens)
            return logits
        elif self.model_type == 'CNN' or self.model_type == 'CNN-with-time':
            hiddens = hiddens.transpose(-1, -2)
            return self.classifier(hiddens)  # [B, T', C]

    @property
    def nparams(self):
        return sum([p.numel() for p in self.parameters()])

