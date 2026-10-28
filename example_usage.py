"""Minimal end-to-end example for MEDIATOR.

Requires access to the gated checkpoint:
    huggingface-cli login   # after requesting access on the model page

Two usage modes are shown:
  (1) full pipeline — embed clinical terms with Qwen3-Embedding-8B, then quantize
  (2) quantize-only — if you already have precomputed Qwen3-Embedding-8B embeddings
"""

import torch

from mediator import MediatorVQ
from mediator.embeddings import QwenEmbedder

CKPT = "MahmoodLab/MEDIATOR"  # or a local directory containing the checkpoint


def main():
    # ── (1) Full pipeline: text → discrete code ids ──────────────────────
    embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-8B", device="cuda")
    model = MediatorVQ.from_pretrained(CKPT, device="cuda")

    terms = [
        "Type 2 Diabetes Mellitus",
        "Diabetes Mellitus, Non-Insulin-Dependent",   # synonym → same code
        "Myocardial Infarction",
        "Heart Attack",                                # synonym → same code
        "Breast Cancer",
    ]
    embeddings = embedder.embed(terms)                    # [5, 4096]
    code_ids = model.quantize(embeddings.to("cuda"))      # [5] long in [0, 100000)
    print("terms:   ", terms)
    print("code ids: ", code_ids.tolist())

    # ── (2) Quantize-only: precomputed Qwen3-Embedding-8B embeddings ─────
    # emb = torch.load("my_embeddings.pt")                # [B, 4096], float32
    # ids = model.quantize(emb.to("cuda"))

    # Codebook vectors (e.g. for training a downstream model over code ids)
    codes = model.code_embeddings                        # [100000, 512]
    print("codebook:", tuple(codes.shape))


if __name__ == "__main__":
    main()
