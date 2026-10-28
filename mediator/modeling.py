"""MEDIATOR model — inference-only implementation.

MEDIATOR learns a discrete neural interlingua: a codebook of 100k codes that
any clinical concept string can be mapped to via its text-encoder embedding.
This module rebuilds the released checkpoint (`pytorch_model.bin`) with the
exact same state-dict layout used at training time:

    adapter.net.{0,2,4}.*        — MLP: 4096 -> 1024 -> 512 (GELU, LayerNorm)
    codebook.embedding.weight     — [100000, 512] codebook vectors

Only those parameters are required at inference time. The checkpoint also
contains training-only heads (contrast projections, temperatures, the
reconstruction MLP and EMA buffers); they are loaded with ``strict=False``
and silently ignored.

Usage:
    from mediator import MediatorVQ
    model = MediatorVQ.from_pretrained("MahmoodLab/MEDIATOR")   # or a local dir
    code_ids = model.quantize(embeddings)      # embeddings: [B, 4096]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from huggingface_hub import snapshot_download
except ImportError:  # only needed for Hub loading
    snapshot_download = None

# State-dict entries required to reconstruct the inference path.
_REQUIRED_KEYS = (
    "adapter.net.0.weight",
    "adapter.net.0.bias",
    "adapter.net.2.weight",
    "adapter.net.2.bias",
    "adapter.net.4.weight",
    "adapter.net.4.bias",
    "codebook.embedding.weight",
)

_DEFAULT_CONFIG = {
    "architecture": "MediatorVQ",
    "encoder_dim": 4096,
    "vq_dim": 512,
    "num_codes": 100000,
    "distance": "cos",
    "adapter_hidden_dim": 1024,
}


class EmbeddingAdapter(nn.Module):
    """Learnable MLP mapping frozen text-encoder embeddings to the VQ space."""

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024,
                 output_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQCodebook(nn.Module):
    """Vector-quantization codebook with cosine (or L2) assignment.

    Inference-only: assignment is a hard argmax over code similarities,
    exactly matching the evaluation path used in the paper.
    """

    def __init__(self, num_codes: int, embed_dim: int = 512, distance: str = "cos"):
        super().__init__()
        assert distance in {"cos", "l2"}
        self.num_codes = int(num_codes)
        self.embed_dim = int(embed_dim)
        self.distance = distance
        self.embedding = nn.Embedding(self.num_codes, self.embed_dim)

    def _compute_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Similarities between [B, D] vectors and the codebook — [B, num_codes]."""
        if self.distance == "l2":
            z2 = (z ** 2).sum(dim=1, keepdim=True)
            e2 = (self.embedding.weight ** 2).sum(dim=1)
            return -z2 - e2.unsqueeze(0) + 2 * (z @ self.embedding.weight.t())
        z_n = F.normalize(z, dim=1)
        e_n = F.normalize(self.embedding.weight, dim=1)
        return z_n @ e_n.t()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2 and z.size(1) == self.embed_dim, \
            f"Expected [B, {self.embed_dim}], got {tuple(z.shape)}"
        return torch.argmax(self._compute_logits(z), dim=1)


class MediatorVQ(nn.Module):
    """Adapter + codebook: maps text-encoder embeddings to discrete code ids.

    forward(embeddings) returns the assigned code id per input, in [0, num_codes).
    """

    def __init__(self, encoder_dim: int = 4096, vq_dim: int = 512,
                 num_codes: int = 100000, distance: str = "cos",
                 adapter_hidden_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.vq_dim = vq_dim
        self.num_codes = num_codes
        self.adapter = EmbeddingAdapter(
            input_dim=encoder_dim, hidden_dim=adapter_hidden_dim,
            output_dim=vq_dim, dropout=dropout,
        )
        self.codebook = VQCodebook(num_codes=num_codes, embed_dim=vq_dim, distance=distance)

    # ------------------------------------------------------------------ core
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map [B, encoder_dim] embeddings to continuous [B, vq_dim] VQ-space vectors."""
        return self.adapter(embeddings.to(self.adapter.net[0].weight.dtype))

    def quantize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map [B, encoder_dim] embeddings to discrete code ids ([B], long)."""
        return self.codebook(self.encode(embeddings))

    def similarity_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine similarities between each input and every code — [B, num_codes]."""
        return self.codebook._compute_logits(self.encode(embeddings))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.quantize(embeddings)

    @property
    def code_embeddings(self) -> torch.Tensor:
        """The [num_codes, vq_dim] codebook vectors (raw stored weights)."""
        return self.codebook.embedding.weight

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, source: Union[str, Path],
                        device: str = "cpu") -> "MediatorVQ":
        """Load a released checkpoint from a local directory or a Hugging Face repo id.

        For Hub sources the repo must be accessible to your token (MEDIATOR is
        released under gated access — request it on the model page first).
        """
        path = Path(_resolve_ckpt_dir(str(source)))
        with open(path / "config.json") as f:
            cfg = json.load(f)

        model = cls(
            encoder_dim=cfg["encoder_dim"],
            vq_dim=cfg["vq_dim"],
            num_codes=cfg["num_codes"],
            distance=cfg.get("distance", "cos"),
            adapter_hidden_dim=cfg.get("adapter_hidden_dim", 1024),
        )
        state_dict = torch.load(path / "pytorch_model.bin",
                                map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        absent = [k for k in _REQUIRED_KEYS if k in missing]
        if absent:
            raise RuntimeError(f"Checkpoint is missing required keys: {absent}")
        return model.to(device).eval()


def _resolve_ckpt_dir(source: str) -> str:
    """Accept a local directory or a Hugging Face repo id."""
    if Path(source).exists():
        return source
    if snapshot_download is None:
        raise RuntimeError(
            f"'{source}' is not a local path and huggingface_hub is not installed. "
            "Install it with: pip install huggingface_hub"
        )
    return snapshot_download(repo_id=source)
