import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear
from torch.nn.functional import softmax, relu


class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(-2)],
                                        requires_grad=False)
        return self.dropout(x)


class Attention(torch.nn.Module):
    """
    Attention
    (B, L, E) (B, L, E) (B, L, E) -> (B, L, D)
    """

    def __init__(self, E, D):
        super().__init__()
        self.Wq = Linear(E, D)
        self.Wk = Linear(E, D)
        self.Wv = Linear(E, D)
        self.D = D

    def forward(self, Q, K, V, mask):
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        Z = softmax(
            (Q @ K.transpose(-2, -1) / math.sqrt(self.D)).masked_fill(mask == 0, -1e9), dim=-1) @ V
        return Z


class MultiHeadAttention(torch.nn.Module):
    """
    Multi Head Attention:

    concatenate multiple attention and reduce results by linear transform

    (B, L, E) (B, L, E) (B, L, E) -> (B, L, E / H) * H -> (B, L, E)
    """

    def __init__(self, E, H, dropout):
        super().__init__()
        self.attentions = torch.nn.ModuleList(
            [Attention(E, E // H) for _ in range(H)])
        self.Wo = Linear(E, E)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, Q, K, V, mask):
        # (B, L, E) -> (B, L, D)
        # (B, L, E) -> (B, L, H * D) * Wo(H * D, D) -> (B, L, D)
        Z = torch.cat([att(Q, K, V, mask) for att in self.attentions], dim=-1)
        return self.dropout(self.Wo(Z))


class FF(torch.nn.Module):
    """
    (B, L, E) -> (B, L, E)
    (B, L, E) -> RELU( (B, L, E) * w1(E, D) ) * w2(D, E) -> (B, L, E)
    """

    def __init__(self, E, D, dropout):
        super().__init__()
        self.W1 = Linear(E, D)
        self.W2 = Linear(D, E)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.W2(self.dropout(relu(self.W1(x)))))


class AddNorm(torch.nn.Module):
    """
    Add two tensor and perform Layer Normalize
    (B, L, E) (B, L, E) -> (B, L, E)
    """

    def __init__(self, E):
        super().__init__()
        self.norm = torch.nn.LayerNorm(E)

    def forward(self, x, sub):
        return self.norm(x + sub)


class DecoderLayer(torch.nn.Module):
    """
    (B, L, E) -> (B, L, E)
    """

    def __init__(self, E, H, Dff, dropout):
        super().__init__()
        self.att = MultiHeadAttention(E=E, H=H, dropout=dropout)
        self.addnorm1 = AddNorm(E=E)
        self.addnorm2 = AddNorm(E=E)
        self.ff = FF(E=E, D=Dff, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.addnorm1(x, self.att(x, x, x, mask))
        x = self.addnorm2(x, self.ff(x))
        return x


class Decoder(torch.nn.Module):
    """
    stacked decoder layer
    (B, L, E) -> (B, L, E)
    """

    def __init__(self, E, Dff, H, Dlayer, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(E=E, Dff=Dff, H=H, dropout=dropout) for _ in range(Dlayer)])

    def forward(self, x, mask):
        for e in self.layers:
            x = e(x, mask)
        return x


class SubsequentMask(torch.nn.Module):
    def __init__(self, pad_id):
        super().__init__()
        self.pad_id = pad_id

    def forward(self, src):
        L = src.shape[1]
        src_mask = (src != self.pad_id).unsqueeze(-2)
        subseq_mask = (torch.from_numpy(
            np.triu(np.ones((1, L, L), dtype=np.uint8), k=1)) == 0).to(src.device)
        return src_mask & subseq_mask


class TransformerLanguageModel(torch.nn.Module):
    """
    Transformer Language Model
    """

    def __init__(
        self,
        d_emb: int,
        dropout: float,
        num_linear_layers: int,
        num_rnn_layers: int,
        pad_token_id: int,
        vocab_size: int
    ):
        super().__init__()

        self.subseqmask = SubsequentMask(pad_id=pad_token_id)
        self.pad_id = pad_token_id
        self.pad_id = pad_token_id

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_emb,
            padding_idx=pad_token_id
        )

        self.pe = PositionalEncoding(d_emb, dropout)

        self.decoder = Decoder(
            E=d_emb,
            Dlayer=num_rnn_layers,
            Dff=num_linear_layers,
            H=8,
            dropout=dropout
        )

    def forward(self, src):
        # (B, L) -> (B, L, E) -> (B, L, E) -> (B, L, V)
        mask = self.subseqmask(src=src)
        src = self.pe(self.embedding(src))
        return self.decoder(src, mask) @ self.embedding.weight.transpose(0, 1)

    def predict(self, x):
        return softmax(self(x), dim=-1)
