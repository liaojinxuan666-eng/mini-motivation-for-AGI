import torch
import torch.nn as nn

from .mamba_block import MambaBlock
from .attention_block import AttentionBlock


class WorldModelConf(nn.Module):
    """
    WorldModel + 置信度头。

    trunk 同阶段 3，末尾分叉：
        coord_head → 预测下一 token（204 类）
        conf_head  → 预测"coord 猜对了吗"（二分类 logit）
    """

    def __init__(
        self,
        vocab_size=204,
        d_model=128,
        n_head=4,
        window=64,
        layer_kinds=None,
        max_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        if layer_kinds is None:
            layer_kinds = ["ssm", "ssm", "ssm", "attn", "ssm", "ssm", "ssm"]
        self.layer_kinds = layer_kinds
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        layers = []
        for k in layer_kinds:
            if k == "attn":
                layers.append(AttentionBlock(d_model, n_head, window, dropout))
            else:
                layers.append(MambaBlock(d_model, dropout=dropout))
        self.layers = nn.ModuleList(layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.coord_head = nn.Linear(d_model, vocab_size, bias=False)
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def trunk(self, x):
        B, L = x.shape
        assert L <= self.max_len
        pos = torch.arange(L, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h)
        return self.ln_f(h)

    def forward(self, x, return_conf=False):
        h = self.trunk(x)
        coord_logits = self.coord_head(h)
        if not return_conf:
            return coord_logits
        conf_logit = self.conf_head(h).squeeze(-1)
        return coord_logits, conf_logit