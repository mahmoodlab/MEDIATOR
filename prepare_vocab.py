"""Prepare a hospital/ontology vocabulary for a downstream (e.g. BEHRT-style)
model over MEDIATOR interlingua codes.

Equivalent to the pipeline used in the paper: given a vocabulary CSV with a
term column and an id column, this script
  1. embeds every term with Qwen3-Embedding-8B (training-time recipe),
  2. quantizes the embeddings to discrete code ids,
  3. writes
       <out>_codebook_emb.jsonl   — one line per code: {"<code_id+1>": [512 floats]}
       <out>_with_vq_indices.csv  — the input CSV plus a `vq_indices` column

Usage:
    python prepare_vocab.py \
        --ckpt MahmoodLab/MEDIATOR \
        --input_csv my_vocab.csv \
        --term_col description --label_col token_id \
        --output_dir out_dir \
        --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mediator import MediatorVQ
from mediator.embeddings import QwenEmbedder


def save_codebook_jsonl(out_path: str, codebook_weight: torch.Tensor):
    """1-indexed code embeddings, one JSON object per line (downstream format)."""
    codebook_weight = codebook_weight.detach().cpu()
    with open(out_path, "w", buffering=1024 * 1024) as f:
        for i in range(codebook_weight.shape[0]):
            emb = codebook_weight[i].cpu().numpy().tolist()
            f.write(json.dumps({str(i + 1): emb}, separators=(",", ":")) + "\n")
    print(f"Wrote {codebook_weight.shape[0]} code embeddings to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="MahmoodLab/MEDIATOR",
                    help="Local checkpoint dir or Hugging Face repo id")
    ap.add_argument("--input_csv", required=True, help="Vocabulary CSV")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--term_col", default="term", help="Column with the concept text")
    ap.add_argument("--label_col", default="conceptId", help="Column with the vocabulary id")
    ap.add_argument("--model_name", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = MediatorVQ.from_pretrained(args.ckpt, device=device)
    embedder = QwenEmbedder(model_name=args.model_name, device=device)

    # 1) codebook embeddings for the downstream model's input embedding layer
    save_codebook_jsonl(
        os.path.join(args.output_dir, "codebook_emb.jsonl"),
        model.code_embeddings.cpu(),
    )

    # 2) per-term code assignment
    df = pd.read_csv(args.input_csv)
    texts = df[args.term_col].astype(str).tolist()
    emb = embedder.embed(texts, batch_size=args.batch_size)     # [N, 4096]
    with torch.no_grad():
        ids = model.quantize(emb.to(device))                    # [N] long

    df["vq_indices"] = ids.cpu().tolist()
    out_csv = os.path.join(args.output_dir, "with_vq_indices.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved vocabulary with VQ indices to {out_csv}")


if __name__ == "__main__":
    main()
