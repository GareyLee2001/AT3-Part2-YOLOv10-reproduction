# AT3② Technical Reproduction — YOLOv10n

**Course:** DDS001 数据科学前沿分析研究  
**Student:** 李嘉烯　`D26092110161`  
**Paper:** Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024  

## What this repo does

| Path | What | Command |
|------|------|---------|
| **A** | Official `.pt` → eval + inference | `python scripts/01_official_pt_eval.py` |
| **B** | Small-data fine-tune (100 epochs) | `python scripts/02_finetune_few_epochs.py` |
| Summary | Merge metrics + paper comparison | `python scripts/03_summarize.py` |

### Fixed protocol

- Model: `yolov10n.pt`
- Dataset: **coco128**（~128 张；YOLO/Ultralytics 最常用的小集，比 coco8 多人用、又不太大）
- Image size: 640
- Path B: **100 epochs**, batch=8
- Device: **CPU**

## Setup

```bash
pip install -r requirements.txt
set KMP_DUPLICATE_LIB_OK=TRUE
```

## Reproduce

```bash
python scripts/01_official_pt_eval.py
python scripts/02_finetune_few_epochs.py
python scripts/03_summarize.py
```

## Latest results (coco128)

| Setting | mAP50-95 |
|---------|---------:|
| Paper YOLOv10-N (full COCO val) | 38.5% |
| A Official `.pt` | 45.71% |
| B 100-epoch fine-tune | 75.73% |

> coco128 ≪ full COCO；且 Path B 在同一小集上训/验，数字偏乐观，不能直接声称打赢论文。详见 `REPRODUCTION_NOTE.md`。

## License / attribution

YOLOv10 via Ultralytics. Paper: Wang et al., NeurIPS 2024.
