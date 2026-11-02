"""Reproducible A2 experiment on the original A1 train/holdout partition.

Run with: assgnment 1 cv/.venv/bin/python train.py
The pre-trained EfficientNetV2-B0 weight file is kept in weights/.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "assgnment 1 cv" / "dataset"
if not DATA.exists():
    DATA = ROOT.parent / "assgnment 1 cv" / "dataset"
WEIGHTS = ROOT / "weights" / "efficientnetv2-b0_notop.h5"
OUT = ROOT / "artifacts"
SEED = 42
SIZE = (224, 224)
CLASSES = ["bottle", "cup", "pen", "spoon"]
BATCH = 16


def make_manifest():
    # Keras' directory-indexing and seed reproduce the A1 *train/holdout* split.
    common = dict(validation_split=0.30, seed=SEED, image_size=SIZE,
                  batch_size=32, label_mode="int", shuffle=True)
    train_index = tf.keras.utils.image_dataset_from_directory(
        DATA, subset="training", **common)
    holdout_index = tf.keras.utils.image_dataset_from_directory(
        DATA, subset="validation", **common)
    assert train_index.class_names == CLASSES
    train = [str(Path(p).relative_to(ROOT)) for p in train_index.file_paths]
    holdout = [str(Path(p).relative_to(ROOT)) for p in holdout_index.file_paths]
    # A1 took 127 samples from a reshuffled tf.data stream on every iteration.
    # This fixed choice is reproducible and disjoint from its training partition.
    rng = random.Random(SEED)
    rng.shuffle(holdout)
    test, val = holdout[:127], holdout[127:]
    assert len(train) == 593 and len(val) == 127 and len(test) == 127
    assert len(set(train) | set(val) | set(test)) == 847
    manifest = {"seed": SEED, "classes": CLASSES,
                "note": "A1 Keras train/holdout partition; fixed seeded half of holdout for A2 test",
                "train": train, "val": val, "test": test}
    (OUT / "split.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def dataset(paths, augment=False, seed_offset=0):
    def load(rel):
        data = tf.io.read_file(tf.strings.join([str(ROOT), rel], separator="/"))
        image = tf.io.decode_jpeg(data, channels=3)
        image = tf.image.resize(image, SIZE)
        return tf.cast(image, tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=4)
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=SEED + seed_offset),
            tf.keras.layers.RandomRotation(0.12, seed=SEED + seed_offset + 1),
            tf.keras.layers.RandomZoom(0.12, seed=SEED + seed_offset + 2),
            tf.keras.layers.RandomTranslation(0.08, 0.08, seed=SEED + seed_offset + 3),
        ])
        ds = ds.batch(BATCH).map(lambda x: aug(x, training=True))
    else:
        ds = ds.batch(BATCH)
    return ds.prefetch(2)


def labels(paths):
    return np.array([CLASSES.index(Path(p).parent.name) for p in paths])


def extract(base, paths, augment=False, seed_offset=0):
    vectors = []
    started = time.perf_counter()
    for i, images in enumerate(dataset(paths, augment, seed_offset)):
        feature_maps = base(images, training=False)
        vectors.append(tf.reduce_mean(feature_maps, axis=(1, 2)).numpy())
        if (i + 1) % 10 == 0:
            print(f"features {i + 1}/{(len(paths) + BATCH - 1)//BATCH}", flush=True)
    result = np.concatenate(vectors).astype("float32")
    print(f"extracted {len(result)} in {time.perf_counter()-started:.1f}s", flush=True)
    return result


def train_head(x, y, xv, yv, name, *, class_weight=None,
               initial_weights=None, learning_rate=1e-3):
    tf.keras.utils.set_random_seed(SEED)
    head = tf.keras.Sequential([
        tf.keras.layers.Input((x.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(4, activation="softmax"),
    ], name=name)
    if initial_weights is not None:
        head.set_weights(initial_weights)
    head.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = head.fit(x, y, validation_data=(xv, yv), epochs=35,
                       batch_size=32, verbose=0, class_weight=class_weight,
                       callbacks=[tf.keras.callbacks.EarlyStopping(
                           monitor="val_loss", patience=7, restore_best_weights=True)])
    head.save(OUT / f"{name}.keras")
    return head, history.history


def evaluate(head, x, y):
    pred = np.argmax(head.predict(x, verbose=0), axis=1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "weighted_f1": float(f1_score(y, pred, average="weighted")),
            "confusion_matrix": confusion_matrix(y, pred, labels=range(4)).tolist(),
            "per_class": classification_report(y, pred, target_names=CLASSES,
                                                output_dict=True, zero_division=0)}


def figure(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(history[metric], label="training")
        ax.plot(history[f"val_{metric}"], label="validation")
        ax.set(xlabel="Epoch", ylabel=metric.title())
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_curves.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    split = make_manifest()
    print("loading EfficientNetV2-B0", flush=True)
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False, weights=None, input_shape=(224, 224, 3))
    base.load_weights(WEIGHTS)
    base.trainable = False
    x = extract(base, split["train"])
    xa1 = extract(base, split["train"], augment=True, seed_offset=0)
    xa2 = extract(base, split["train"], augment=True, seed_offset=100)
    xv = extract(base, split["val"])
    xt = extract(base, split["test"])
    y, yv, yt = labels(split["train"]), labels(split["val"]), labels(split["test"])
    np.savez_compressed(OUT / "features.npz", train=x, aug1=xa1, aug2=xa2,
                        val=xv, test=xt)
    results = {"dataset": {s: len(split[s]) for s in ("train", "val", "test")},
               "synthetic_share": 0, "pretrained": "ImageNet EfficientNetV2-B0",
               "split_note": split["note"]}
    for name, vectors, targets in [
        ("no_augmentation", x, y),
        ("with_augmentation", np.concatenate([x, xa1, xa2]), np.tile(y, 3)),
    ]:
        head, history = train_head(vectors, targets, xv, yv, name)
        figure(history, name)
        results[name] = evaluate(head, xt, yt)
        results[name]["epochs"] = len(history["loss"])
        results[name]["history"] = {k: [float(v) for v in vals]
                                     for k, vals in history.items()}
        print(name, results[name]["accuracy"], results[name]["macro_f1"], flush=True)
    # Re-evaluate A1 saved model on precisely the same fixed A2 test images.
    old = tf.keras.models.load_model(ROOT / "weights" / "a1_final_mobilenetv2.keras",
                                      compile=False)
    p = np.concatenate([old(images, training=False).numpy()
                        for images in dataset(split["test"])])
    pred = p.argmax(axis=1)
    results["a1_re_evaluated"] = {
        "accuracy": float(accuracy_score(yt, pred)),
        "macro_f1": float(f1_score(yt, pred, average="macro")),
        "weighted_f1": float(f1_score(yt, pred, average="weighted")),
        "confusion_matrix": confusion_matrix(yt, pred, labels=range(4)).tolist()}
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print("DONE", json.dumps({k: v.get("accuracy") for k, v in results.items()
                              if isinstance(v, dict)}), flush=True)


if __name__ == "__main__":
    main()
