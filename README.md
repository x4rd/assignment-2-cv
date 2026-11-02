# Assignment 2 CV — Household Object Recognition

**Tsoy Ruslan · SE-2420 · Astana IT University**

This project continues Assignment 1 with a four-class household-object classifier. It uses an ImageNet-pretrained EfficientNetV2-B0 feature extractor, a trained classification head, and Grad-CAM visual explanations. The local web app accepts uploads and camera frames, then shows class probabilities, a heatmap, and inference latency.

## Deliverables

- [Report (PDF, 7 pages)](output/pdf/Assignment2_Tsoy_Ruslan.pdf)
- [Demo video (3 minutes 30 seconds)](output/Assignment2_Tsoy_Ruslan_demo.mp4) — annotated, silent walkthrough using model outputs
- Local web app: `server.py`, `predict.py`, `web/index.html`
- Original experiment: `train.py`, `artifacts/metrics.json`, `artifacts/split.json`, training curves, and model weights
- Cup update: `import_coco_cups.py`, `retrain_cup.py`, `dataset/coco_cup_sources.json`, and `artifacts/cup_retrain_metrics.json`

## Run the app

On macOS, double-click `launch.command`. On the first launch it creates `.venv` and installs the Python packages, then starts the server and opens `http://127.0.0.1:8000`. The first launch needs an internet connection and may take a few minutes. Keep the Terminal window open. Do not open `web/index.html` directly, because the page needs the local model API.

For a fresh checkout with Python 3.13:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python server.py
```

The app includes four sample images and runs without the training dataset. It accepts JPEG, PNG and WebP files up to 10 MB. Camera capture requires browser permission.

## Dataset and experiment

The original self-collected dataset contains 847 real images: 200 bottles, 239 cups, 208 pens, and 200 spoons. The raw photos are excluded from version control because they can contain iPhone EXIF location data. The included app sample images are re-encoded without EXIF. No synthetic images were used.

To reproduce training, place the original files in `assgnment 1 cv/dataset/{bottle,cup,pen,spoon}/` inside this folder or its parent, preserving their names. Then run `python train.py` in the environment above. The saved split manifest fixes 593 training, 127 validation, and 127 test images. Both the saved A1 MobileNetV2 and the A2 variants were evaluated on the same fixed test set.

| Model | Accuracy | Macro-F1 | Correct / 127 |
|---|---:|---:|---:|
| A1 MobileNetV2, re-evaluated | 92.91% | 92.54% | 118 |
| A2 EfficientNetV2-B0 | **99.21%** | **99.24%** | **126** |
| A2 with two augmented views | 97.64% | 97.53% | 124 |

The augmentation ablation used horizontal flips, rotation, zoom and translation. These results and the PDF report describe the **original 847-image experiment**.

### 100-photo cup update

The local `dataset/cup/` now includes 100 distinct photos from the [COCO 2017 validation set](https://cocodataset.org/#download). Each image has an annotated `cup` instance at least 60 pixels wide and high. The largest cup instance was cropped with a 25% margin and re-encoded without EXIF. COCO's `cup` category includes mugs and drinking glasses. The 100 images, their original URLs, bounding boxes, and individual Flickr license links are listed in `dataset/coco_cup_sources.json`. The selection excludes licenses that prohibit derivatives; several included photos have noncommercial licenses, so check each source entry before redistribution.

The updated local dataset has **947 images**, including **339 cups**. All new photos were added only to training: 693 training, 127 validation, 127 test. The original validation and test partitions were preserved in `artifacts/cup_retrain_split.json`. `retrain_cup.py` trained three heads, selected the best by validation macro-F1, and saved the deployed head as `artifacts/cup_retrained.keras`. The frozen ImageNet EfficientNetV2-B0 extractor did not change.

| Deployed head on original 127-photo test set | Accuracy | Macro-F1 | Correct |
|---|---:|---:|---:|
| Before cup update | 99.21% | 99.24% | 126 |
| After cup update | 98.43% | 98.34% | 125 |

The new head still classifies all 34 original cup test photos correctly; one additional bottle is classified as a spoon. The old test set does not measure performance on unseen COCO photos. Detailed metrics and candidate validation scores are in `artifacts/cup_retrain_metrics.json`. The local app loads the new head.

To reproduce the cup update, download the official COCO `instances_val2017.json` annotations (SHA-256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`), then run:

```bash
./.venv/bin/python import_coco_cups.py /path/to/instances_val2017.json
./.venv/bin/python retrain_cup.py
```

## Limitations

The small test set contains visually related photos from some capture sessions, so it may overstate performance on new households. The classifier always chooses one of the four labels even for unknown objects. Grad-CAM is a coarse explanation, not a segmentation mask. This project performs image classification, so mAP@0.5 and IoU are not applicable without detection-box or mask annotations.

## References

- [EfficientNetV2 paper](https://arxiv.org/abs/2104.00298)
- [Keras EfficientNetV2 implementation](https://keras.io/api/applications/efficientnet_v2/)
- [Neurohive overview of the EfficientNet family](https://neurohive.io/en/news/efficientnet-achieving-state-of-the-art-results-by-systematically-scaling-deep-neural-networks/)
- [Grad-CAM paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)
