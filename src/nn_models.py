"""Neural networks on top of our own word2vec embeddings (Area A)."""
import torch
import torch.nn as nn

PAD, UNK = 0, 1
MAX_TOKENS = 64


class AvgFFN(nn.Module):
    """Feed-forward network on the average of the word2vec vectors of a message."""

    def __init__(self, emb_matrix: torch.Tensor, hidden=256, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb_matrix, freeze=True, padding_idx=PAD)
        self.net = nn.Sequential(
            nn.Linear(emb_matrix.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2))

    def forward(self, ids):
        mask = (ids != PAD).unsqueeze(-1).float()
        avg = (self.emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.net(avg)


class TextCNN(nn.Module):
    """1D CNN (Kim, 2014): word2vec embeddings -> convolutions (2/3/4-grams) -> max-pool -> dense."""

    def __init__(self, emb_matrix: torch.Tensor, n_filters=128, kernels=(2, 3, 4), dropout=0.5,
                 freeze=False):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb_matrix, freeze=freeze, padding_idx=PAD)
        d = emb_matrix.shape[1]
        self.convs = nn.ModuleList(nn.Conv1d(d, n_filters, k, padding=k // 2) for k in kernels)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernels), 2)

    def forward(self, ids):
        x = self.emb(ids).transpose(1, 2)                                   # B x D x T
        mask = (ids == PAD).unsqueeze(1)
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))[:, :, : ids.shape[1]]
            h = h.masked_fill(mask, -1e4)
            feats.append(h.max(dim=2).values)                               # max-over-time pooling
        return self.fc(self.drop(torch.cat(feats, 1)))


def encode_tokens(tokens, vocab, max_len=MAX_TOKENS):
    ids = [vocab.get(t, UNK) for t in tokens[:max_len]]
    return ids or [UNK]


def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max(L, 4)), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out
