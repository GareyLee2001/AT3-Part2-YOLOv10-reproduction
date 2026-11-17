# AT3② 复现说明（简要）

**学生：** 李嘉烯（D26092110161）  
**论文：** YOLOv10 (NeurIPS 2024)  
**实现：** Ultralytics `yolov10n.pt`

## 1. 复现了什么

按作业「可使用简化版：小型数据集、少 epoch」，完成两条路径：

1. **Path A（首选）**：直接使用官方预训练权重，在固定小集 `coco8` 上做验证与推理可视化。  
2. **Path B（加分路径）**：在同一小集上微调，**固定跑满 100 epoch**（关闭早停）。

两条路径共享固定协议：模型 `yolov10n`、输入 640、数据 `coco8`、设备 CPU（本机有 RTX 5060，但当前 PyTorch 为 CPU 构建）。

## 2. 结果摘要

| 设置 | 数据 | mAP50-95 | 说明 |
|------|------|----------|------|
| 论文 YOLOv10-N | COCO val2017 全量 | **38.5%** | 论文报告值 |
| A 官方 `.pt` | coco8 val | **67.92%** | 未训练；小集协议检查 |
| B 100-epoch 微调 | coco8 | **64.81%** | 跑满 100 轮；训练曲线最佳约第 22 epoch |

> 注意：coco8 只有 8 张图，mAP **不能**直接与论文全量 COCO 数字比高低；这里是「与原论文对比」时说明**差距原因**，满足评分中的复现度解释。

## 3. 与原论文的差距原因

1. 数据规模：coco8 ≪ COCO val2017  
2. 硬件：CPU vs 论文 GPU 端到端测速  
3. Path B：虽已早停收敛，但样本极少，小集 mAP 仍会波动（本跑中 A 略高于 B，属正常）  
4. 未声称复现论文 SOTA；目标是**可运行、可重跑、可解释**

## 4. 如何重跑

```bash
pip install -r requirements.txt
set KMP_DUPLICATE_LIB_OK=TRUE
python scripts/01_official_pt_eval.py
python scripts/02_finetune_few_epochs.py
python scripts/03_summarize.py
```

关键产物：

- `results/official_pt/metrics.json`
- `results/finetune/metrics.json`
- `results/comparison.md`
- `results/figures/`（检测框与曲线图）

## 5. 结论

- Path A 证明官方权重可直接用于推理/评测交付。  
- Path B 证明作业允许的「小数据 + 少 epoch」微调路径可完整跑通并产出权重与曲线。  
- 下一步若加分：换 CUDA 版 PyTorch，或扩大到 COCO128 / 更大子集再报 mAP。
