# AT3② Result Comparison

## vs original paper

| Setting | Dataset | mAP50-95 | Note |
|---|---|---:|---|
| Paper YOLOv10-N | COCO val2017 full | 38.5% | Official paper number |
| A Official `.pt` (ours) | coco8.yaml | 67.92% | No train; device=cpu |
| B Fine-tune until convergence (ours) | coco8.yaml | 64.03% | ran=42, best≈22; patience=30; device=cpu |

## Interpretation

- Path A/B use **coco8** (8 images), so mAP is a **protocol check**, not a claim to match full-COCO paper mAP.
- Gap vs paper is expected: tiny subset + CPU + simplified setting.
- Path B shows the assignment-allowed *small data + few epoch* fine-tune path is runnable.
