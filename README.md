# Assignment 2 CV — Household Object Recognition

**Tsoy Ruslan · SE-2420 · Astana IT University**

This project continues Assignment 1 with a four-class household-object classifier. It uses an ImageNet-pretrained EfficientNetV2-B0 feature extractor, a trained classification head, and Grad-CAM visual explanations. The local web app accepts uploads and camera frames, then shows class probabilities, a heatmap, and inference latency.

## Deliverables

- [Report (PDF, 7 pages)](output/pdf/Assignment2_Tsoy_Ruslan.pdf)
- [Demo video (3 minutes 30 seconds)](output/Assignment2_Tsoy_Ruslan_demo.mp4) — annotated, silent walkthrough using model outputs
- Local web app: `server.py`, `predict.py`, `web/index.html`
- Experiment code and results: `train.py`, `artifacts/metrics.json`, `artifacts/split.json`, training curves, and model weights

## Run the app

On macOS, double-click `launch.command`. It starts the Python server and opens `http://127.0.0.1:8000`. Keep the Terminal window open. Do not open `web/index.html` directly, because the page needs the local model API.

For a fresh checkout with Python 3.13:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python server.py
```

The app includes four sample images and runs without the training dataset. It accepts JPEG, PNG and WebP files up to 10 MB. Camera capture requires browser permission.

## Dataset and experiment

The original self-collected dataset contains 847 real images: 200 bottles, 239 cups, 208 pens, and 200 spoons. The raw photos are excluded from this repository because they total about 3 GB and can contain iPhone EXIF location data. The included sample images are re-encoded without EXIF. No synthetic images were used.

To reproduce training, place the original files in `assgnment 1 cv/dataset/{bottle,cup,pen,spoon}/` inside this folder or its parent, preserving their names. Then run `python train.py` in the environment above. The saved split manifest fixes 593 training, 127 validation, and 127 test images. Both the saved A1 MobileNetV2 and the A2 variants were evaluated on the same fixed test set.

| Model | Accuracy | Macro-F1 | Correct / 127 |
|---|---:|---:|---:|
| A1 MobileNetV2, re-evaluated | 92.91% | 92.54% | 118 |
| A2 EfficientNetV2-B0 | **99.21%** | **99.24%** | **126** |
| A2 with two augmented views | 97.64% | 97.53% | 124 |

The augmentation ablation used horizontal flips, rotation, zoom and translation. The app loads the best-performing head, `artifacts/no_augmentation.keras`. The reported metrics, confusion matrices, curves, latency, Grad-CAM examples, and ethics discussion are in the report.

## Limitations

The small test set contains visually related photos from some capture sessions, so it may overstate performance on new households. The classifier always chooses one of the four labels even for unknown objects. Grad-CAM is a coarse explanation, not a segmentation mask. This project performs image classification, so mAP@0.5 and IoU are not applicable without detection-box or mask annotations.

## References

- [EfficientNetV2 paper](https://arxiv.org/abs/2104.00298)
- [Keras EfficientNetV2 implementation](https://keras.io/api/applications/efficientnet_v2/)
- [Neurohive overview of the EfficientNet family](https://neurohive.io/en/news/efficientnet-achieving-state-of-the-art-results-by-systematically-scaling-deep-neural-networks/)
- [Grad-CAM paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)
