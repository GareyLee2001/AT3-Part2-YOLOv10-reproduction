# -*- coding: utf-8 -*-
"""
AT3② Path B: small-data + few-epoch fine-tune of YOLOv10n.
Protocol: coco8 train, epochs=5, imgsz=640, then val + inference.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "finetune"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "yolov10n.pt"
DATA = "coco8.yaml"
IMGSZ = 640
EPOCHS = 5
BATCH = 4
DEVICE = "cpu"


def find_coco8_images() -> Path | None:
    datasets_dir = Path(SETTINGS.get("datasets_dir", Path.home() / "datasets"))
    for p in [
        datasets_dir / "coco8" / "images" / "val",
        datasets_dir / "coco8" / "images" / "train",
    ]:
        if p.exists():
            return p
    return None


def main() -> None:
    print(f"[B] Fine-tune {MODEL_NAME} on {DATA} for {EPOCHS} epochs")
    model = YOLO(MODEL_NAME)

    t0 = time.perf_counter()
    train_results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=str(OUT),
        name="train",
        exist_ok=True,
        patience=EPOCHS,
        pretrained=True,
        plots=True,
        verbose=True,
    )
    train_sec = time.perf_counter() - t0

    best = OUT / "train" / "weights" / "best.pt"
    last = OUT / "train" / "weights" / "last.pt"
    ckpt = best if best.exists() else last
    print(f"[B] Load fine-tuned checkpoint: {ckpt}")
    ft = YOLO(str(ckpt))

    t1 = time.perf_counter()
    metrics = ft.val(
        data=DATA,
        imgsz=IMGSZ,
        device=DEVICE,
        split="val",
        project=str(OUT),
        name="val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    val_sec = time.perf_counter() - t1

    img_dir = find_coco8_images()
    if img_dir is not None:
        ft.predict(
            source=str(img_dir),
            imgsz=IMGSZ,
            device=DEVICE,
            save=True,
            project=str(FIG),
            name="finetune_predict",
            exist_ok=True,
        )

    box = metrics.box
    summary = {
        "path": "B_finetune_few_epochs",
        "base_model": MODEL_NAME,
        "finetuned_checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.exists() else None,
        "data": DATA,
        "imgsz": IMGSZ,
        "epochs": EPOCHS,
        "batch": BATCH,
        "device": DEVICE,
        "map50_95": float(box.map),
        "map50": float(box.map50),
        "map75": float(getattr(box, "map75", float("nan"))),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "train_seconds": round(train_sec, 3),
        "val_seconds": round(val_sec, 3),
        "speed_ms_per_image": {
            "preprocess": float(metrics.speed.get("preprocess", 0.0)),
            "inference": float(metrics.speed.get("inference", 0.0)),
            "postprocess": float(metrics.speed.get("postprocess", 0.0)),
        },
        "note": "Simplified reproduction: tiny coco8 + 5 epochs. Not full COCO training.",
        "train_save_dir": str(getattr(train_results, "save_dir", OUT / "train")),
    }
    out_json = OUT / "metrics.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[B] Saved: {out_json}")


if __name__ == "__main__":
    main()
