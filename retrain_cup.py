"""Retrain the deployed classification head with 100 COCO cup images.

The 2017 COCO images enter training only. The original 127-image validation and
127-image test sets remain unchanged, so the before/after comparison is valid.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import CLASSES, OUT, ROOT, WEIGHTS, evaluate, extract, figure, labels, train_head


def original_paths(paths):
    result = []
    for path in paths:
        relative = Path("dataset") / Path(path).relative_to("assgnment 1 cv/dataset")
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
        result.append(str(relative))
    return result


def main():
    tf.keras.utils.set_random_seed(42)
    old = json.loads((OUT / "split.json").read_text())
    sources = json.loads((ROOT / "dataset" / "coco_cup_sources.json").read_text())
    added = [str(Path("dataset/cup") / item["file"]) for item in sources["images"]]
    if len(added) != 100 or len(set(added)) != 100:
        raise ValueError("Expected exactly 100 distinct COCO cup images")
    split = {part: original_paths(old[part]) for part in ("train", "val", "test")}
    split["train"].extend(added)
    if not all((ROOT / path).is_file() for path in added):
        raise FileNotFoundError("A COCO cup image is missing")
    if len(set().union(*[set(split[p]) for p in split])) != sum(map(len, split.values())):
        raise ValueError("Train, validation, and test sets overlap")
    (OUT / "cup_retrain_split.json").write_text(json.dumps({
        "seed": 42, "classes": CLASSES,
        "note": "Original validation/test unchanged; all 100 COCO cups added to train only",
        **split,
    }, indent=2) + "\n")

    print("Loading frozen EfficientNetV2-B0", flush=True)
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights=None, input_shape=(224, 224, 3))
    base.load_weights(WEIGHTS)
    base.trainable = False
    x = extract(base, split["train"])
    xv = extract(base, split["val"])
    xt = extract(base, split["test"])
    y, yv, yt = [labels(split[part]) for part in ("train", "val", "test")]

    previous = tf.keras.models.load_model(OUT / "no_augmentation.keras", compile=False)
    before = evaluate(previous, xt, yt)
    class_counts = np.bincount(y, minlength=len(CLASSES))
    class_weight = {i: len(y) / (len(CLASSES) * int(count))
                    for i, count in enumerate(class_counts)}
    candidates = {}
    for name, options in (
        ("cup_unweighted", {}),
        ("cup_balanced", {"class_weight": class_weight}),
        ("cup_finetuned", {"class_weight": class_weight,
                           "initial_weights": previous.get_weights(),
                           "learning_rate": 1e-4}),
    ):
        head, history = train_head(x, y, xv, yv, name, **options)
        candidates[name] = {
            "validation": evaluate(head, xv, yv),
            "epochs": len(history["loss"]),
            "history": {key: [float(v) for v in values]
                        for key, values in history.items()},
        }
        figure(history, name)
    selected = max(candidates, key=lambda name: (
        candidates[name]["validation"]["macro_f1"],
        candidates[name]["validation"]["accuracy"],
        -min(candidates[name]["history"]["val_loss"]),
    ))
    shutil.copyfile(OUT / f"{selected}.keras", OUT / "cup_retrained.keras")
    deployed = tf.keras.models.load_model(OUT / "cup_retrained.keras", compile=False)
    after = evaluate(deployed, xt, yt)
    metrics = {
        "dataset": {part: len(paths) for part, paths in split.items()},
        "added_coco_cups": 100,
        "train_class_counts": {klass: int(class_counts[index])
                               for index, klass in enumerate(CLASSES)},
        "test_set": "Original A2 fixed test partition; no COCO images",
        "selection_rule": "Highest validation macro-F1, then accuracy, then final validation loss",
        "selected": selected, "candidates": candidates,
        "before": before, "after": after,
    }
    (OUT / "cup_retrain_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"DONE before={before['accuracy']:.4f} after={after['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
