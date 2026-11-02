"""Import 100 distinct COCO 2017 validation cups into the local training set.

Usage: python import_coco_cups.py /path/to/instances_val2017.json
The annotation file must have the official SHA-256 digest recorded below.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "dataset" / "cup"
MANIFEST = ROOT / "dataset" / "coco_cup_sources.json"
ANNOTATIONS_SHA256 = "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f"
ALLOWED_LICENSES = {1, 2, 4, 5, 7, 8}  # Exclude NoDerivs licenses because images are cropped.


def candidates(data):
    images = {im["id"]: im for im in data["images"]}
    licenses = {lic["id"]: lic for lic in data["licenses"]}
    best = {}
    for ann in data["annotations"]:
        if ann["category_id"] != 47 or ann["iscrowd"]:
            continue
        x, y, width, height = ann["bbox"]
        if min(width, height) < 60:
            continue
        im = images[ann["image_id"]]
        if im["license"] not in ALLOWED_LICENSES:
            continue
        if ann["image_id"] not in best or ann["area"] > best[ann["image_id"]]["area"]:
            best[ann["image_id"]] = ann
    for image_id in sorted(best):
        im, ann = images[image_id], best[image_id]
        yield {
            "image_id": image_id,
            "annotation_id": ann["id"],
            "bbox_xywh": ann["bbox"],
            "source_url": "https://images.cocodataset.org/val2017/" + im["file_name"],
            "flickr_url": im["flickr_url"],
            "license": licenses[im["license"]]["name"],
            "license_url": licenses[im["license"]]["url"].replace("http:", "https:"),
            "file": f"coco_val2017_{image_id:012d}.jpg",
        }


def import_one(item):
    target = DEST / item["file"]
    if target.exists():
        with Image.open(target) as image:
            image.verify()
        return item
    result = subprocess.run(
        ["curl", "--location", "--fail", "--silent", "--show-error",
         "--retry", "2", "--max-time", "45",
         item["source_url"].replace("https://images.cocodataset.org/",
                                    "https://s3.amazonaws.com/images.cocodataset.org/")],
        capture_output=True, check=True,
    )
    with Image.open(io.BytesIO(result.stdout)) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        x, y, w, h = item["bbox_xywh"]
        margin = 0.25 * max(w, h)
        box = (max(0, round(x - margin)), max(0, round(y - margin)),
               min(source.width, round(x + w + margin)),
               min(source.height, round(y + h + margin)))
        image = source.crop(box)
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image.save(target, "JPEG", quality=90, optimize=True)
        item["crop_xyxy"] = list(box)
    return item


def main(annotation_path):
    raw = Path(annotation_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ANNOTATIONS_SHA256:
        raise ValueError("Annotation file differs from official COCO val2017 instances")
    data = json.loads(raw)
    DEST.mkdir(parents=True, exist_ok=True)
    selected = list(candidates(data))
    if len(selected) < 100:
        raise ValueError(f"Only {len(selected)} eligible distinct cup images")
    imported = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(import_one, item): item for item in selected}
        for future in as_completed(futures):
            try:
                imported.append(future.result())
            except Exception as exc:
                print(f"Skipped {futures[future]['image_id']}: {exc}", flush=True)
    # All successful downloads may exceed 100; retain a stable set of image IDs.
    imported.sort(key=lambda item: item["image_id"])
    keep = imported[:100]
    if len(keep) != 100:
        raise RuntimeError(f"Downloaded only {len(keep)} of 100 required cup images")
    kept_files = {item["file"] for item in keep}
    for path in DEST.glob("coco_val2017_*.jpg"):
        if path.name not in kept_files:
            path.unlink()
    MANIFEST.write_text(json.dumps({
        "dataset": "COCO 2017 validation", "class": "cup",
        "annotation_sha256": ANNOTATIONS_SHA256,
        "processing": "Largest cup bbox per image, 25% margin; RGB JPEG without metadata",
        "images": keep,
    }, indent=2) + "\n")
    print(f"Imported {len(keep)} distinct cup images", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python import_coco_cups.py instances_val2017.json")
    main(sys.argv[1])
