"""Token ↔ embedding-space codec for continuous diffusion."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedEmbeddingModule(nn.Module):
    """
    encode(): token ids -> L2-normalised, √D-scaled, bottlenecked vectors.
    decode(): bottlenecked vectors -> log-probabilities over the vocabulary.

    The L2 normalisation with √D scaling keeps every clean embedding on a
    fixed-radius sphere, which is what makes a single Gaussian noise schedule
    meaningful across all tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pretrained_embeds: Optional[torch.Tensor] = None,
        use_pretrained: bool = True,
        freeze_pretrained: bool = False,
        bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim if bottleneck_dim else embed_dim
        self.scale = math.sqrt(embed_dim)  # √D for normalization scaling

        # Determine embedding mode
        self.is_pretrained = pretrained_embeds is not None and use_pretrained

        # Initialize embedding layer
        if self.is_pretrained:
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embeds.detach().clone(),
                freeze=freeze_pretrained
            )
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)

        # Encoder: embed → bottleneck
        if bottleneck_dim and bottleneck_dim != embed_dim:
            self.encoder_proj = nn.Sequential(
                nn.Linear(embed_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim)
            )
        else:
            self.encoder_proj = nn.Identity()

        # Decoder: bottleneck → embed → vocab
        if bottleneck_dim and bottleneck_dim != embed_dim:
            self.decoder_proj = nn.Sequential(
                nn.Linear(bottleneck_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        else:
            self.decoder_proj = nn.Identity()

        # Unembedding layer initialized with pretrained embeddings
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        if self.is_pretrained:
            with torch.no_grad():
                self.output_head.weight.copy_(pretrained_embeds.detach().clone())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode token IDs to embeddings with normalization and scaling."""
        embeds = self.embed(x)
        # Apply L2 normalization with √D scaling (critical for stability)
        embeds = F.normalize(embeds, p=2, dim=-1) * self.scale
        # Project to bottleneck dimension
        embeds = self.encoder_proj(embeds)
        return embeds

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to log probabilities over vocabulary."""
        # Expand from bottleneck back to embedding dimension
        x = self.decoder_proj(x)
        # Optional: renormalize before unembedding (CDCD framework recommendation)
        # x = F.normalize(x, p=2, dim=-1)
        # Unembed to vocabulary logits
        logits = self.output_head(x)
        # Return log softmax for numerical stability with NLL loss
        return F.log_softmax(logits, dim=-1)

    def soft_decode(self, x: torch.Tensor):
        logits = self.decode(x)
        soft_decode = logits @ self.embed.weight
        out = self.encoder_proj(soft_decode)

        return logits, out
