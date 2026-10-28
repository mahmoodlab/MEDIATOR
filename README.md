<div align="center">

<a href="https://research.massgeneralbrigham.org/en/researchers-and-collaborators/mass-general-brigham-ai"><img src="assets/mediator_banner.png" width="100%" alt="MEDIATOR: Discrete Neural Interlingua for Ontology-Agnostic EHR Modeling. Mass General Brigham AI Institute."></a>

[![arXiv](https://img.shields.io/badge/arXiv-2610.27047-b31b1b.svg)](https://arxiv.org/abs/2610.27047)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-6847A3)](#citation)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-MEDIATOR-FFD21E)](https://huggingface.co/MahmoodLab/MEDIATOR)

Tong Ding\*, Caiwei Tian\*, Dongmin Bang, Ming Y. Lu, Sophia J. Wagner,  
Andrew Zhang, Rowland Pettit, Long Phi Le, Faisal Mahmood

<sub>\*Equal contribution · Mass General Brigham AI Institute · Harvard University · Massachusetts Institute of Technology</sub>

[**Paper**](https://arxiv.org/abs/2610.27047) · [**Model weights**](https://huggingface.co/MahmoodLab/MEDIATOR) · [**Quickstart**](#quickstart) · [**Citation**](#citation)

</div>

---

**MEDIATOR is a discrete, ontology-agnostic tokenizer for electronic health records.** It maps any clinical concept expressed as text (a diagnosis, medication, lab, procedure, local identifier, or free-text mention) to one of 100,000 discrete **interlingua codes** in a shared vector-quantized codebook, independent of any specific ontology (ICD, SNOMED CT, RxNorm, LOINC, ...). Concepts that mean the same thing land on the same code, so EHR harmonization becomes open-vocabulary tokenization without manual mappings or a fixed vocabulary.

> [!NOTE]
> This release covers **MEDIATOR inference** (text → interlingua code) and vocabulary preparation for downstream EHR models.

## How it works

A frozen text encoder embeds the concept description, a trainable projection maps it into the quantization space, and hard cosine-argmax assignment against a learned codebook yields the discrete code. Training combines same-concept contrastive objectives before and after quantization, multi-view assignment agreement, semantic teachers from enriched descriptions, code co-occurrence, and relation graphs, and reconstruction to the encoder embedding space.

<div align="center">
  <img src="assets/mediator_architecture.gif" width="100%" alt="MEDIATOR animation: clinical strings from different ontologies and free text are mapped to shared interlingua codes; a frozen text encoder, a projection and a 100,000-code VQ codebook, trained with contrastive equivalence, assignment agreement and medical knowledge signals">
  <br>
  <sub>Any clinical string lands on a shared interlingua code. Synonyms merge; medical knowledge arranges distinct concepts.</sub>
</div>

Used as the token interface of an EHR Transformer, MEDIATOR improves mean AUPRC over the strongest prior tokenizer by 4.0 points on 20 MIMIC-III tasks and by 8.6 points in zero-shot transfer to 14 eICU tasks, where closed-vocabulary tokenizers can use only 13% of events. See the paper for the full results.

## Installation & access

The checkpoint is hosted on Hugging Face under **gated access**. Request access on the [model page](https://huggingface.co/MahmoodLab/MEDIATOR), then install and log in:

```bash
pip install -r requirements.txt
hf auth login
```

## Quickstart

`mediator/` contains the inference code: `mediator/modeling.py` for the VQ model and `mediator/embeddings.py` for the Qwen3-Embedding-8B recipe used at training time (instruct prefix, left padding, last-token pooling, L2 normalization).

```python
import torch
from mediator import MediatorVQ
from mediator.embeddings import QwenEmbedder

embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-8B", device="cuda")
model = MediatorVQ.from_pretrained("MahmoodLab/MEDIATOR", device="cuda")

terms = [
    "Type 2 Diabetes Mellitus",
    "Diabetes Mellitus, Non-Insulin-Dependent",   # synonym -> same code
    "Myocardial Infarction",
    "Heart Attack",                                # synonym -> same code
    "Breast Cancer",
]

with torch.inference_mode():
    embeddings = embedder.embed(terms)                 # (5, 4096) float32
    code_ids = model.quantize(embeddings.to("cuda"))   # (5,) long in [0, 100000)
```

If you already have Qwen3-Embedding-8B embeddings computed with the same recipe, skip the embedder and call `model.quantize(emb)` directly. The codebook vectors are available as `model.code_embeddings` (shape 100000 × 512). See [example_usage.py](example_usage.py) for the full example.

The model has three parts:

| Component | Role |
|---|---|
| Text encoder | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), frozen. Maps a lowercased concept string to a 4096-d embedding. |
| Adapter | A learned MLP (4096 → 1024 → 512, GELU + LayerNorm) that projects the embedding into the interlingua space. |
| VQ codebook | 100,000 × 512 code vectors. Hard cosine-argmax assignment gives the discrete code id. |

> [!IMPORTANT]
> MEDIATOR expects embeddings produced with the exact recipe in `mediator/embeddings.py` (task instruction, left padding, last-token pooling, L2 normalization, lowercased input). Embeddings from a different pooling or prompt will not map to same meaningful codes.

## Preparing a hospital vocabulary for a downstream model

To train a downstream model over interlingua codes (as done in the paper), map your vocabulary to code ids and export the codebook vectors:

```bash
python prepare_vocab.py \
    --ckpt MahmoodLab/MEDIATOR \
    --input_csv your_vocab.csv \
    --term_col description --label_col token_id \
    --output_dir out_dir
```

This produces:

| File | Contents |
|---|---|
| `out_dir/codebook_emb.jsonl` | One 512-d vector per code, 1-indexed, one JSON object per line. Use it to initialize the downstream model's token embedding layer. |
| `out_dir/with_vq_indices.csv` | Your vocabulary plus a `vq_indices` column with the assigned code id per term. |

## License

The model weights are released under [CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial academic research only. MEDIATOR is not validated for clinical use.

## Citation

If you find MEDIATOR useful, please cite:

```bibtex
@misc{ding2026discreteneuralinterlinguaontologyagnostic,
      title={Discrete Neural Interlingua for Ontology-Agnostic EHR Modeling}, 
      author={Tong Ding and Caiwei Tian and Dongmin Bang and Ming Y. Lu and Sophia J. Wagner and Andrew Zhang and Rowland Pettit and Long Phi Le and Faisal Mahmood},
      year={2026},
      eprint={2610.27047},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.27047}, 
}
```

## Questions

Please open a [GitHub issue](https://github.com/mahmoodlab/MEDIATOR/issues) for questions about using MEDIATOR.
