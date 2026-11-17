# AT3② Technical Reproduction — YOLOv10n

**Course:** DDS001 数据科学前沿分析研究  
**Student:** 李嘉烯　`D26092110161`  
**Paper:** Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024  
**Selected from AT3①:** YOLO series literature set (recommended frontier paper)

## What this repo does

Two assignment-allowed paths:

| Path | What | Command |
|------|------|---------|
| **A** | Official `.pt` → inference + fixed-subset eval | `python scripts/01_official_pt_eval.py` |
| **B** | Small data + train until convergence (early stop) | `python scripts/02_finetune_few_epochs.py` |
| Summary | Merge metrics + paper comparison | `python scripts/03_summarize.py` |

### Fixed simplified protocol

- Model: `yolov10n.pt` (nano)
- Dataset: Ultralytics **coco8** (8 images) — small-data substitute for full COCO
- Image size: 640
- Path B: **max 200 epochs**, `patience=30` early-stopping（验证指标不再提升就停）
- Device in this run: **CPU** (RTX 5060 present, but current PyTorch build is CPU-only)

## Setup

```bash
pip install -r requirements.txt
# optional if OpenMP duplicate warning appears on Windows:
set KMP_DUPLICATE_LIB_OK=TRUE
```

## Reproduce

```bash
python scripts/01_official_pt_eval.py
python scripts/02_finetune_few_epochs.py
python scripts/03_summarize.py
```

Outputs:

- `results/official_pt/metrics.json`
- `results/finetune/metrics.json`
- `results/comparison.md`
- prediction / val plots under `results/figures/` and run folders

## Comparison to the original paper

Scoring asks for **result fidelity vs the paper**, not multi-model bake-offs.

| Setting | Dataset | mAP50-95 | Note |
|---------|---------|---------:|------|
| Paper YOLOv10-N | COCO val2017 (full) | ~38.5 | Official reported number |
| A Official `.pt` (ours) | coco8 | see `results/comparison.md` | No training |
| B Few-epoch FT (ours) | coco8, 5 epochs | see `results/comparison.md` | Simplified fine-tune |

**Why numbers differ from the paper (expected):**

1. coco8 ≠ full COCO val2017  
2. CPU latency ≠ paper GPU end-to-end latency  
3. Path B uses only 5 epochs on 8 images — protocol demo, not SOTA claim  

## Deliverables mapping

| Assignment ask | Here |
|----------------|------|
| Runnable code | this repo |
| Result screenshots | `results/figures/`, val plots |
| Brief write-up | `REPRODUCTION_NOTE.md` |
| vs paper | `results/comparison.md` |

## License / attribution

YOLOv10 weights/code via Ultralytics. Paper: Wang et al., NeurIPS 2024.
